from fastai import *

from fastai.vision import *

from fastai.widgets import *

from fastai.callbacks.hooks import *



import time
## Pytorch seed for reproducibility

def random_seed(seed_value, use_cuda=True):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False
classes = ['r_regulus','r_ignicapilla']

path = Path('data/regulus')
def copy_download_files_from_urls(folder, file, path=path, max_pics=200):

    dest = path/folder

    dest.mkdir(parents=True, exist_ok=True)



    !cp ../input/* {path}/

    try:

        download_images(os.path.join("..", "input","bird-recognition-regulus-urls", file), dest, max_pics=max_pics)

    except:

        try:

            # try once again

            download_images(os.path.join("..", "input","bird-recognition-regulus-urls", file), dest, max_pics=max_pics)

        except:

            pass
copy_download_files_from_urls("r_regulus", "r_regulus.csv", max_pics=200)
copy_download_files_from_urls("r_ignicapilla", "r_ignicapilla.csv", max_pics=200)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
# you can manipulate with the seed

random_seed(333)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(max_warp=0), size=81, num_workers=0).normalize(imagenet_stats)


random_seed(323)

data.show_batch(rows=9, ds_type=DatasetType.Valid, figsize=(12,12))
def seed_and_fit_cycle(learner,*args, seed=123, **kwargs):

    random_seed(seed)

    learner.fit_one_cycle(*args, **kwargs)
random_seed(111)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

seed_and_fit_cycle(learn, 3, seed=42)
#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=150)
#ImageCleaner(ds, idxs, path)
size=225

np.random.seed(42)

data = ImageDataBunch.from_csv(path, valid_pct=0.3, csv_labels="../../../input/regulus-cleaned/cleaned(1).csv", ds_tfms=get_transforms(max_warp=0), size=225, num_workers=4).normalize(imagenet_stats)
random_seed(111)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

seed_and_fit_cycle(learn, 4)
learn.unfreeze()
random_seed(111)

learn.lr_find()

learn.recorder.plot()
seed_and_fit_cycle(learn, 6, max_lr=slice(2e-5,1e-3))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
wrong_labeled_valid_idx = []

for idx in range(len(data.valid_ds)):

    img,y = data.valid_ds[idx]

    pred_class,pred_idx,outputs = learn.predict(img)

    if (str(y) != str(pred_class)):

        wrong_labeled_valid_idx.append(idx)
def heatmap(idx):

    x,y = data.valid_ds[idx]

    m = learn.model.eval();



    def hooked_backward(cat=y):

        with hook_output(m[0]) as hook_a: 

            with hook_output(m[0], grad=True) as hook_g:

                preds = m(xb)

                preds[0,int(cat)].backward()

        return hook_a,hook_g



    xb,_ = data.one_item(x)

    xb_im = Image(data.denorm(xb)[0])

    xb = xb.cuda()



    def show_heatmap(hm):

        _,ax = plt.subplots()

        xb_im.show(ax)

        ax.set_title(label="True_label: " + str(y))

        return ax.imshow(hm, alpha=0.6, extent=(0,size,size,0),

                  interpolation='bilinear', cmap='magma')



    hook_a,hook_g = hooked_backward()

    acts  = hook_a.stored[0].cpu()

    avg_acts = acts.mean(0)

    return show_heatmap(avg_acts)
for idx in wrong_labeled_valid_idx:

    heatmap(idx)
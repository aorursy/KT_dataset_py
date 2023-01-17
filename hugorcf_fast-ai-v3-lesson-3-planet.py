%reload_ext autoreload

%autoreload 2

%matplotlib inline
import fastai

from fastai.vision import *
fastai.__version__
path = Path('/kaggle/input/')

path.ls()
df = pd.read_csv(path/'train_v2.csv')

df.head()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)

src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')

       .random_split_by_pct(0.2)

       .label_from_df(label_delim=' '))   # changed from original version!
data = (src.transform(tfms, size=128)

        .databunch(num_workers=0).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = create_cnn(data, arch, metrics=[acc_02, f_score], model_dir='/tmp/models')
learn.lr_find()
learn.recorder.plot()
lr = 0.01
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.save('stage-2-rn50')
data = (src.transform(tfms, size=256)

        .databunch(num_workers=0).normalize(imagenet_stats))



learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr=1e-2/2
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-256-rn50')
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.recorder.plot_losses()
learn.save('stage-2-256-rn50')
learn.export()
test = ImageItemList.from_folder(path/'test-jpg').add(ImageItemList.from_folder(path/'test-jpg-additional'))

len(test)
learn = load_learner(path, test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
thresh = 0.2

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
labelled_preds[:5]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
df.to_csv(path/'submission.csv', index=False)
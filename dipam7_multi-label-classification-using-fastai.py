from fastai.vision import *
path = Path('/kaggle/input/')

path.ls()
df = pd.read_csv(path/'train_v2.csv')

df.head()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)

src = (ImageItemList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')

       .random_split_by_pct(0.2)

       .label_from_df(sep=' '))
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
learn.recorder.plot_losses()
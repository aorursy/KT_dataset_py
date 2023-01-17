from fastai import *

from fastai.vision import *
classes = ['teddy','grizzly','black']
folder = 'black'

file = 'url_black.txt'
path = Path('data/bears')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images('../input/teddy-classifier-txt-manthan/data/bears/url_black.txt', dest, max_pics=200)
folder = 'grizzly'

file = 'url_grizzly.txt'
path = Path('data/bears')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images('../input/teddy-classifier-txt-manthan/data/bears/url_grizzly.txt', dest, max_pics=200)
folder = 'teddy'

file = 'url_teddie.txt'
path = Path('data/bears')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images('../input/teddy-classifier-txt-manthan/data/bears/url_teddie.txt', dest, max_pics=200)
path.ls()
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.lr_find(start_lr=1e-5, end_lr=1e-1)
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
from fastai.widgets import *
db = (ImageList.from_folder(path)

                   .split_none()

                   .label_from_folder()

                   .transform(get_transforms(), size=224)

                   .databunch()

     )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)



learn_cln.load('stage-2');
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
ImageCleaner(ds, idxs, path)
ds, idxs = DatasetFormatter().from_similars(learn_cln)
ImageCleaner(ds, idxs, path, duplicates=True)
learn.export()
path.ls()
defaults.device = torch.device('cpu')
img = open_image(path/'black'/'00000091.jpg')

img
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)

pred_class.obj
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(1, max_lr=0.5)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(1, max_lr=1e-1)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(1, max_lr=3e-5)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(5, max_lr=1e-5)
learn.recorder.plot_losses()
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=32, 

        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0

                              ),size=224, num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, ps=0, wd=0)

learn.unfreeze()
learn.fit_one_cycle(50, slice(1e-6,1e-4))
learn.recorder.plot_losses()
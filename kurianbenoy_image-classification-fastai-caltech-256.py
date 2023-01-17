%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
# Printing 256 object classes

path = Path('/kaggle/input/256_objectcategories/256_ObjectCategories')

path.ls()
tfms = get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3, max_zoom=1.01)
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=tfms,

                                  size=128,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet50


learn = cnn_learner(data, arch, metrics=accuracy, model_dir="/tmp/model/")



learn.lr_find()





learn.recorder.plot()



lr = 1e-01/2
learn.fit_one_cycle(5, slice(lr))

learn.save('stage1')
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=tfms,

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr = 1e-3/3
learn.fit_one_cycle(5, slice(lr))

img = data.train_ds[0][0]
learn.predict(img)
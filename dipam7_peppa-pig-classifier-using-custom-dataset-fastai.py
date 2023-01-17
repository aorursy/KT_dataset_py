from fastai import *

from fastai.vision import *
classes = ['peppa','others']
folder = 'peppa'

file = 'urls_peppa.txt'
path = Path('../working/')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
# resetting the path

!cp ../input/* {path}/
download_images(path/file, dest, max_pics=200)
folder = 'others'

file = 'urls_others.txt'
path = Path('../working/')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images(path/file, dest, max_pics=200)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
# creating a data bunch

# training set is in the current path

# since we don't have a validation set we set valid_pct = 0.2

# to use 20% data as validation

np.random.seed(42)

data = ImageDataBunch.from_folder(".", train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
# verifying the classes

data.classes
# viewing some of the images

data.show_batch(rows=3, figsize=(8,7))
# creating a resnet34

learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
# saving the weights so we don't have to retrain it each time

learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, max_lr=slice(1e-5,1e-3))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
# we see that our model is 100% accurate

interp.plot_confusion_matrix()
# from fastai.widgets import *
# ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
# ImageCleaner(ds, idxs, path)
# ds, idxs = DatasetFormatter().from_similars(learn)
# opening a random image and making a prediction

img = open_image(path/'peppa/00000041.jpg')

img
pred_class,pred_idx,outputs = learn.predict(img)

print(pred_class)
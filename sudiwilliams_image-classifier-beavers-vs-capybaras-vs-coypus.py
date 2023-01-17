from fastai import *

from fastai.vision import *
classes = ['beaver', 'capybara', 'coypu']
folder = 'beaver'

file = 'beavers.csv'
path = Path('data/rodent')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images(path/file, dest, max_pics=100)
folder = 'capybara'

file = 'capybaras.csv'
path = Path('data/rodent')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=100)
folder = 'coypu'

file = 'coypus.csv'
path = Path('data/rodent')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=100)
for c in classes:

     print(c)

     verify_images(path/c, delete=True, max_size=500)
#np.random.seed(42)

#data = ImageDataBunch.from_folder(path, train="", valid_pct=0.2,

#         ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
#data.classes, data.c, len(data.train_ds), len(data.valid_ds)
#data.show_batch(rows=3, figsize=(7,8))
#learn = cnn_learner(data, models.resnet50, metrics=error_rate)
#learn.fit_one_cycle(4)
#learn.save('stage-1')
#learn.unfreeze()
#learn.lr_find()
#learn.recorder.plot(suggestion=True)
#learn.fit_one_cycle(4, max_lr=slice(3e-6,3e-5))
#learn.save('stage-2')
#learn.load('stage-2');
#interp = ClassificationInterpretation.from_learner(learn)

#interp.plot_top_losses(9,figsize=(12,12))
#interp.plot_confusion_matrix()
#from fastai.widgets import *
#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
#ds, idxs = DatasetFormatter().from_similars(learn)
#ImageCleaner(ds, idxs, path, duplicates=True)
#np.random.seed(42)

#cleaned_data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',

#ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
#cleaned_data.classes, cleaned_data.c, len(cleaned_data.train_ds), len(cleaned_data.valid_ds)
#cleaned_data.show_batch(rows=3, figsize=(7,8))
#learn_cln = cnn_learner(cleaned_data, models.resnet50, metrics=error_rate)
#learn.fit_one_cycle(4)
#learn.save('stage-3')
#img = open_image(path/'beaver'/'00000010.jpg')

#img
#classes = ['beaver', 'capybara', 'coypu']
#data2 = ImageDataBunch.single_from_classes(path, classes,

#                                  ds_tfms=get_transforms(),

#                                  size=224).normalize(imagenet_stats)
#learn = cnn_learner(data2, models.resnet50).load('stage-3')
#pred_class,pred_idx,outputs = learn.predict(img)

#pred_class
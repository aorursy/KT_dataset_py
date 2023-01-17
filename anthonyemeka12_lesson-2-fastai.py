from fastai.vision import *
folder = 'black'

file = 'urls_black.csv'
folder = 'teddys'

file = 'urls_teddys.csv'
folder = 'grizzly'

file = 'urls_grizzly.csv'
path = Path('data/bears')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
path.ls()
classes = ['teddys','grizzly','black']
download_images(path/file, dest, max_pics=200)
# If you have problems download, try with `max_workers=0` to see exceptions:

download_images(path/file, dest, max_pics=20, max_workers=0)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
# If you already cleaned your data, run this cell instead of the one before

# np.random.seed(42)

# data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',

#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
# If the plot is not showing try to give a start and end learning rate# learn.lr_find(start_lr=1e-5, end_lr=1e-1)learn.recorder.plot()
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
# If you already cleaned your data using indexes from `from_toplosses`,

# run this cell instead of the one before to proceed with removing duplicates.

# Otherwise all the results of the previous step would be overwritten by

# the new run of `ImageCleaner`.



# db = (ImageList.from_csv(path, 'cleaned.csv', folder='.')

#                    .no_split()

#                    .label_from_df()

#                    .transform(get_transforms(), size=224)

#                    .databunch()

#      )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)



learn_cln.load('stage-2');
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
# Don't run this in google colab or any other instances running jupyter lab.# If you do run this on Jupyter Lab, you need to restart your runtime and# runtime state including all local variables will be lost.ImageCleaner(ds, idxs, path)
ds, idxs = DatasetFormatter().from_similars(learn_cln)
ImageCleaner(ds, idxs, path, duplicates=True)
learn.export()
defaults.device = torch.device('cpu')
img = open_image(path/'black'/'00000021.jpg')

img
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(1, max_lr=0.5)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(5, max_lr=1e-5)
learn.recorder.plot_losses()
learn = cnn_learner(data, models.resnet34, metrics=error_rate, pretrained=False)
learn.fit_one_cycle(1)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=32, 

        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0

                              ),size=224, num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, ps=0, wd=0)

learn.unfreeze()
learn.fit_one_cycle(40, slice(1e-6,1e-4))
# make sure latest version of fastai is installed

! conda install -c pytorch -c fastai fastai --yes
from fastai import *

from fastai.vision import *

vision.__version__
# let's make sure our code runs when we commit by using a loop...

folders = ['black','teddys','grizzly']

files = ['urls_black.txt','urls_teddys.txt','urls_grizzly.txt']

path = Path('bears')

url_path = Path('../input')



for i in range(0,len(folders)):

    dest = path/folders[i]

    dest.mkdir(parents=True, exist_ok=True)

    download_images(url_path/files[i], dest, max_pics=200)
# create ImageDataBunch

classes = ['teddys','grizzly','black']

for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_workers=8)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
# Training a model

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(2e-5,7e-4))
learn.save('stage-2')
# Interpretation

learn.load('stage-2')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
# Cleaning input data

from fastai.widgets import *



ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=30)
ImageCleaner(ds, idxs, path)
ds, idxs = DatasetFormatter().from_similars(learn)
ImageCleaner(ds, idxs, path, duplicates=True)
import pandas as pd

df = pd.read_csv(path/'cleaned.csv', header='infer')

df.head()
data = ImageDataBunch.from_df(path, df, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#data.classes

#data.show_batch(rows=3, figsize=(7,8))

data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-3')
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(2e-5,3e-4))
learn.save('stage-4')
# Putting the model in production

# use CPU only for prediction

defaults.device = torch.device('cpu')
img = open_image(path/'black'/'00000021.jpg')

img
classes = ['black', 'grizzly', 'teddys']

data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = create_cnn(data2, models.resnet34)

learn.load('stage-2')
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
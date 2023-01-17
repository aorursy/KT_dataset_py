%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from PIL import Image
img = Image.open('../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train/Acura Integra Type R 2001/00198.jpg')

width, height = img.size
width, height
path = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/'
np.random.seed(42)

data = ImageDataBunch.from_folder(path,valid_pct=0.2, 

                                  ds_tfms=get_transforms(), 

                                  size=300, bs=64, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(10,12))
learn = create_cnn(data, models.resnet50, metrics=[error_rate,accuracy], model_dir = '/kaggle/working/')
learn.fit_one_cycle(6)
learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6, 1e-5))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
learn.recorder.plot_losses()
interp.plot_top_losses(6, figsize=(30,26))
interp.most_confused(min_val=4)
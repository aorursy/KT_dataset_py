%cd ../input/food-101/food-101/
!ls food-101/
!ls food-101/images/
from collections import defaultdict
from shutil import copy
import os
import gc

# Helper method to split dataset into train and test folders
def prepare_data(filepath, src,dest):
  classes_images = defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')

  for food in classes_images.keys():
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")
%cd /
print("Creating train data...")
prepare_data('/kaggle/input/food-101/food-101/food-101/meta/train.txt', '/kaggle/input/food-101/food-101/food-101/images', 'train')
%cd /
print("Creating test data...")
prepare_data('/kaggle/input/food-101/food-101/food-101/meta/test.txt', '/kaggle/input/food-101/food-101/food-101/images', 'test')
print("Total number of samples in train folder")
!find train -type d -or -type f -printf '.' | wc -c

print("Total number of samples in test folder")
!find test -type d -or -type f -printf '.' | wc -c
from fastai.metrics import accuracy
from fastai.vision import *

BS = 16
SEED = 786
NUM_WORKERS = 6

from pathlib import Path
path = Path('/')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0., xtra_tfms=[cutout()])

src = (ImageList.from_folder(path/'train')
       .split_by_rand_pct(0.2)
       .label_from_folder())

data = (src.add_test_folder(test_folder = path/'test')              
         .transform(tfms, size=128)
         .databunch(num_workers=NUM_WORKERS,bs=BS)).normalize(imagenet_stats)
data.show_batch(3)
gc.collect()

learner = cnn_learner(
    data,
    models.resnet50,
    path=path,
    metrics=[accuracy],
    ps = 0.5
)
learner.lr_find()
learner.recorder.plot()
lr = 1e-02
learner.fit_one_cycle(1, slice(lr))
learner.save('FOOD101_stage-1')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(2, slice(1e-6, lr/5))
learner.save('FOOD101_stage-2')
data = (src.add_test_folder(test_folder = path/'test')              
         .transform(tfms, size=256)
         .databunch(num_workers=NUM_WORKERS,bs=BS)).normalize(imagenet_stats)

learner.data = data
learner.freeze()
learner.lr_find()
learner.recorder.plot()
lr=1e-2
learner.fit_one_cycle(1, slice(lr))
learner.save('FOOD101_stage-1-256')
learner.unfreeze()
learner.fit_one_cycle(2, slice(1e-6, 1e-5
                               /5))
learner.recorder.plot_losses()
learner.save('FOOD101_stage-2-256')
learner.export()
data = (src.add_test_folder(test_folder = path/'test')              
         .transform(tfms, size=512)
         .databunch(num_workers=NUM_WORKERS,bs=BS)).normalize(imagenet_stats)

learner.data = data

learner.freeze()
learner.lr_find()
learner.recorder.plot()
lr=1e-5/2
learner.fit_one_cycle(1, slice(lr))
learner.save('FOOD101_stage-1-512')
learner.unfreeze()
learner.fit_one_cycle(2, slice(1e-6, 1e-5/2))
learner.recorder.plot_losses()
learner.save('FOOD101_stage-2-512')
learner.export()
import warnings
warnings.filterwarnings("ignore")

src = (ImageDataBunch.from_folder(path = path, train = 'train', valid = 'test')
       .split_by_rand_pct(0.2)
       .label_from_folder())

data = (src.transform(tfms, size=512)
         .databunch(num_workers=NUM_WORKERS,bs=BS)).normalize(imagenet_stats)
learner.fit_one_cycle(1)
def predict_class(model, images, show = True):
    for img in images:
        im = open_image(img)
        pred,_,_ = model.predict(im)
        if show:
            plt.imshow(image2np(im.data))                      
            plt.axis('off')
            print(pred)
            plt.show()
import numpy as np
import os
import matplotlib.pyplot as plt

# list all files in dir
files = [y for x in os.listdir("/test/") for y in os.listdir(os.path.join("/test/", x)) if os.path.isfile(y)]

# select 0.1 of the files randomly 
random_files = np.random.choice(files, int(len(files)*.1))
predict_class(learner, random_files,True )


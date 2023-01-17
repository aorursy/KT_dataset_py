%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.metrics import error_rate

import numpy

import cv2 
path = "../input/fruits/data"
Train_path = path+"/train" 

test_path = path+'/test'

fnames = get_image_files(Train_path)

fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(Train_path, fnames, pat, ds_tfms=get_transforms(), size=320, bs=64

                                  ).normalize(imagenet_stats)

test_data = ImageDataBunch.from_name_re(test_path, fnames, pat, ds_tfms=get_transforms(), size=320, bs=64

                                  ).normalize(imagenet_stats)
data.show_batch(rows=2, figsize=(9,8))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(1)
learn.model_dir='/kaggle/working'

learn.save('stage-1')
learn.load('stage-1');
image = test_data.test_ds

img =cv2.imread(test_path+"/Apple-Braeburn_1.jpg")

img2 = open_image(test_path+"/Apple-Braeburn_1.jpg")

result =learn.predict(img2)

plt.imshow(img)

plt.xlabel(result[0].__str__)

plt.show()



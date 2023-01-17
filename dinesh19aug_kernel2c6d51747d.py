%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
path = untar_data(URLs.MNIST); path
path_training = path/'training'

path_testing = path/'testing'

path_training.ls()
fnames2=get_image_files(path_training/'1')





'''

You cannot do this

for x in range(0,10):

  files = get_image_files(path_training/str(x))

  fnames.append(files) ----> This will create 10 items with each item being a list in iteself list of lists

'''

print(fnames2[:5])
fnames=[]

for x in range(0,10):

  files = get_image_files(path_training/str(x))

  for y in (files):

    fnames.append(y)

print(len(fnames))
np.random.seed(2)

#pat = r'/([^/]+)_\d+.jpg$'

pat = r'(?<=training/)\d'

path_training.ls()
import re

 # not using 

def get_labels(file_path):

    pattern=re.compile(pat) 

    res=pattern.search(str(file_path))

    return res.group(0)

data = ImageDataBunch.from_name_func(path_training, fnames, label_func=get_labels,ds_tfms=get_transforms(do_flip=False),  size=224).normalize(imagenet_stats)

data.classes

data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
data_resnet = ImageDataBunch.from_name_func(path_training, fnames, label_func=get_labels,ds_tfms=get_transforms(do_flip=False),  size=224).normalize(imagenet_stats)

data_resnet.classes
learn_resnet = cnn_learner(data_resnet, models.resnet50, metrics=error_rate)
learn_resnet.lr_find()

learn_resnet.recorder.plot()
learn_resnet.fit_one_cycle(10)
learn_resnet.save('stage-1-50')

interp = ClassificationInterpretation.from_learner(learn_resnet)

interp.most_confused(min_val=2)
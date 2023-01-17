%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *

from fastai.metrics import error_rate
# Data location /kaggle/input/dataset/train

path_train=pathlib.PosixPath('/kaggle/input/dataset/train')

path_test=pathlib.PosixPath('/kaggle/input/dataset/test')

path_train.ls()

fnames=[]

gender=['man', 'woman']

for x in gender:

  files = get_image_files(path_train/x)

  for y in (files):

    fnames.append(y)
print(len(fnames))

fnames[:5]
np.random.seed(2)

#pat = r'/([^/]+)_\d+.jpg$'

pat = r'(?<=train\/)\w+'

path_train.ls()
import re

# not using 

def get_labels(file_path):

  pattern=re.compile(pat) 

  res=pattern.search(str(file_path))

  return res.group(0)

data = ImageDataBunch.from_name_func(path_train, fnames, label_func=get_labels,ds_tfms=get_transforms(do_flip=False),  size=224).normalize(imagenet_stats)

data.classes
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.model

learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)



interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#!mkdir /kaggle/working/models

learn.model_dir='/kaggle/working/models/'

learn.save('stage-1')
learn.unfreeze()

learn.fit_one_cycle(1)
learn.load('stage-1');

learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-2))

learn.save('stage-2')
data_resnet = ImageDataBunch.from_name_func(path_train, fnames, label_func=get_labels,ds_tfms=get_transforms(do_flip=False),  size=224).normalize(imagenet_stats)

data_resnet.classes



learn_resnet = cnn_learner(data_resnet, models.resnet50, metrics=error_rate)

learn_resnet.model_dir='/kaggle/working/models/'

learn_resnet.lr_find()

learn_resnet.recorder.plot()
learn_resnet.fit_one_cycle(10)
learn_resnet.save('stage-2')

learn_resnet.unfreeze()

learn_resnet.fit_one_cycle(5, max_lr=slice(1e-6,1e-1))
learn_resnet.save('stage-3')
interp = ClassificationInterpretation.from_learner(learn_resnet)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
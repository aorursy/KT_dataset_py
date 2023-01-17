%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate
help(untar_data)
URLs.PETS
path = untar_data(URLs.PETS)

path
path.ls()
path_anno = path/'annotations'

path_img = path/'images'

#path/ is a path object
fnames = get_image_files(path_img)

fnames[:5]
#regular expression to get the label from the text

np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img,fnames,pat,ds_tfms=get_transforms(),size=224)

data.normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(7,6))
print(data.classes)

len(data.classes),data.c
#just needs two things: data and what is your model

#many ways to make a CNN but resnet is very good for almost all things. The main decision is 34 vs 50, where 34 is faster

#also gives 

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

#pretrains the model. 
#now we want to fit it

learn.fit_one_cycle(4)

#number decides how many times to go through the training set. If it goes through too many times it'll overfit
#saves model weights

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(15,11))

# We'll learn what we were most confident about but got the most wrong
doc(interp.plot_top_losses)

#allows you to check the docs as well as what the help function tells you
#could plot a confusion matrix

interp.plot_confusion_matrix(figsize=(12,12),dpi=60)
#what could be more helpful:

interp.most_confused(min_val=2)

#will pull the most mixed up of the confusion matrix

#you can then interpret - is it making mistakes that make sense?
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()

#shows the fastest speed we can have our model learn at without blowing it up completely
learn.recorder.plot()

#shows the learn rate plot vs. loss
#we want to retrain at the learning rate we want but we don't need to retrain all of the layers

learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

#max_lr slicing means start at the first learning rate and end at the end learning rate, distributing it across the layers
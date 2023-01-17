"""

Lets start by writing a few magic methods so as to present our graphs and ensure that any edits that we make to the library are

reloaded automatically. Also, any charts or images which you wish to display are shown in this notebook itself

"""

%reload_ext autoreload

%autoreload 2

%matplotlib inline
#Let us import the fast ai libraries

from fastai import *

from fastai.vision import *



#To access the dataset files

import os



#To do some RegeX magic

import re



import numpy as np



"""

It turns out, that the folder structure has some arbitray alphanumeric characters before the name of the dog-breed. So we cannot use the names of the folders as is, for

labelling our data. As it turns out, we have a special function in the fastai library(ImageDataBunch.from_name_re), which will do the work for us. We will need to supply the regex to extract the labels

from the folder name.

"""

#Let's see how this function works

help(ImageDataBunch.from_name_re)

#The main path of the dataset can be defined as

root_path = "../input/stanford-dogs-dataset/images/Images/"



#We need something that will iterate through our dataset and store the path of every image. Below code shows how it's done.

f_names = []

for dirname, _, filenames in os.walk(root_path):

    for filename in filenames:

        f_names.append(os.path.join(dirname, filename))

f_names[:5]

#Now for the RegeX pattern, we will need to group the labels, hence we will define the pattern as

pattern_regex = re.compile(r'\d-(\w+-*\w*)')



# We can also specify the size of the image, and the batch size

dims = 224

batch_size = 64 



#Setting the numpy random seed, to get the same results everytime

np.random.seed(1)
data = ImageDataBunch.from_name_re(path=root_path, fnames=f_names, pat=pattern_regex, ds_tfms=get_transforms(), size=dims, bs=batch_size, num_workers=0).normalize(imagenet_stats)

print(f"{data.classes}\nTotal Number of breeds = {len(data.classes)}")
data.show_batch(rows=3)
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
output_directory = r'/kaggle/working'

learn.model_dir = output_directory

learn.save('stage-1-rerun')
learn.export(file = Path("/kaggle/working/exportmodel.pkl"))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.most_confused(min_val=2)
learn.model_dir = r'../input/dog-classifier-weights/'

print(learn.model_dir)

learn.load(r'stage-1')
learn.model_dir=r'/kaggle/working/'

learn.lr_find()
learn.recorder.plot()
#Unfreezing the rest of the layers to train on the entire NN

learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-5))
#Saving our weights to use them later

learn.export(file = Path("/kaggle/working/export.pkl"))
#Loading the weights to check if the weights are saved properly

learn = load_learner(deployed_path, test=ImageList.from_folder(test_path))
interp.plot_top_losses(9, figsize=(15,11))
dog_breed = list(data.classes)

data2 = ImageDataBunch.single_from_classes(r'./input/sample-doggos/', dog_breeds, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn2 = create_cnn(data2, models.resnet34)

learn2 = load_learner('/kaggle/working/')
#Testing out our sample images

img = open_image('../input/sample-doggos/samoyed.jpg')

pred_class, pred_idx, outputs = learn.predict(img)

print(pred_class)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
#Training our Resnet-50 Model

learn.fit_one_cycle(4)
#Saving our model weights

learn.export(file = Path("/kaggle/working/exportmodel_resnet50.pkl"))
learn.model_dir = r'/kaggle/working/'

learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6, 1e-5))
learn.export(file = Path("/kaggle/working/exportmodelresnet50_unfrozen.pkl"))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
# first we import the libraries



from fastai.vision import * 

from fastai import *



import warnings

warnings.filterwarnings('ignore')
path = "../input/african-wildlife" # path to data set in kaggle



np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

#imagedatabunch wraps the dataset and transforms is into the require format for training our model
#list classes

data.classes
data.show_batch(rows=4, figsize=(7,8))
#the model uses tranfer learning using the resnet34

learn = cnn_learner(data, models.resnet34, metrics=accuracy, callback_fns=ShowGraph)
learn.fit_one_cycle(5) # we set the model to run for 5 epochs
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
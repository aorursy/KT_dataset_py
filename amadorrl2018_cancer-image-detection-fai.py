# Goal: Cancer image classification

# no_cancer: 0, cancer: 1
%reload_ext autoreload

%autoreload 2

%matplotlib inline 
# Import libraries 



from fastai import *

from fastai.vision import *

import pandas as pd

# Creating  paths and display a list

train_dir = ('../input/train')

test_dir = ('../input/test')

path = Path('../input')

path.ls()
# Reading the labels  



df = pd.read_csv(path/'../input/train_labels.csv')

df.head(10)
# View the top five train files 

fnames = get_image_files(train_dir)

fnames[:5]
# Create an image data bunch

# ImageDataBunch: contains all the data to build a model(Train, Validation, and Test(optional))

# from_csv: Extract labels 

# get_transforms(): Will return tuples Train, Validation and test 

# size: size of image 96x96

# normalize:changes the range of pixels (0 - 255) and removes the noise

'''Note: if pixels are below 0 becomes 0, if they are above 255 they

become 255. Making data the same size, same mean and

same standard deviation. RGB channels becomes mean of 0

and standard deviation of 1.  If data is not normalize,

training the model will be difficult'''



np.random.seed(42) #making sure we get the same dataset 

data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               suffix=".tif", test = test_dir, size = 96, ds_tfms = get_transforms())

data.path = pathlib.Path('.')

data.normalize(imagenet_stats)
'''AUC, or Area Under Curve, is a metric for binary classification. 

It’s probably the second most popular one, after accuracy.  However,

I will be using AUC over accuracy since it is often preferred as 

it provides a “broader” view of the performance '''

# y_pred: calculates output

# y_true: target data

# tens = True: values will all be multiples of 1/10



from sklearn.metrics import roc_auc_score



def auc_score(y_pred,y_true,tens=True):

    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score=tensor(score)

    else:

        score=score

    return score
# Display some images 3x3



data.show_batch(rows = 3, figsize = (7,6))
# Display all the labels(2) and categories(2) 



print(data.classes)

len(data.classes), data.c
# Model Training

'''The model will take images as input and predicted

probability will the output for each of the categories.

CNN will be use as a backbone and a fully connected

head with a single hidden layer as a classifier'''

# models.resnet34: is the architecture 34(34 layers) which will train fast

'''Note: Basically, is downloading resnet34 pre-trained

weights. This means that this model has already been trained

for a particular task.  This model knows something.'''

# other option: resnet50 (50 layers)

# auc_score: for binary classification 



learn = cnn_learner(data, models.resnet34, metrics = auc_score)


# Let's plot learn to find the best learning rate 

# Select where is decending before is going up

# This is a key hyperparameter to train a neural network 

'''Determines how quickly (or slowly) we want to update the weights 

after each iteration''' 

'''Note: If the learning rate is too low: loss function does not improve.

If decending: optimal learning rate range (a quick drop in the loss function).

If learning rate too high: begins to diverge. '''



learn.lr_find()

learn.recorder.plot()


# run it for four cycles 

# error_rate 2% and 98% accuracy

learn.fit_one_cycle(4, max_lr = slice(3e-4,3e-2))
# Results

# the learn is pass throught and here we have a classification object



interp = ClassificationInterpretation.from_learner(learn)
# Display the top losses of the prediction

# Note: This will tell us how good is the prediction



interp.plot_top_losses(9, figsize = (15,11))
# confusion matrix 

# This plot is showing the right number of predictions in blue



interp.plot_confusion_matrix()

#Validation Prediction



preds,y=learn.get_preds()

pred_score= auc_score(preds,y)

pred_score
# We have 98% accuracy, but could we make it better?

# Let's try TTA

# Test Time Augmentation



'''TTA takes data augmentations at random as 

well as the un-augmented original.  Then calculate predictions 

for all these images, take the average, 

and make that our final prediction. This is only 

for validation set and/or test set.'''



preds,y=learn.TTA()

pred_score_tta=auc_score(preds,y)

pred_score_tta
# Target output

y[:5]
# predicted output

preds[:5]
# Test Prediction 



preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)

# target output

y_test[:5]
# predicted output

preds_test[:5]


preds_test_tta,y_test_tta=learn.TTA(ds_type=DatasetType.Test)

# target output

y_test_tta[:5]
# predicted output

preds_test_tta[:5]
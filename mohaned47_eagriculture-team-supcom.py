'''Problem Statement



Misdiagnosis of the many diseases impacting agricultural crops can lead to misuse of chemicals leading to 

the emergence of resistant pathogen strains, increased input costs, and more outbreaks with significant

economic loss and environmental impacts. Current disease diagnosis based on human scouting is time-consuming

and expensive, and although computer-vision based models have the promise to increase efficiency, the great

variance in symptoms due to age of infected tissues, genetic variations, and light conditions within trees

decreases the accuracy of detection

this is a classification computer vision problem

. '''

#####running this notebook locally  requirments:



#installing fastai

#downloading the dataset from this link https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data

#you can also use our pretrained model to skip the training time wait (50min-90min based on specs)



#####a  better alternative is run a this notebook on kaggle using this link https://www.kaggle.com/c/plant-pathology-2020-fgvc7/notebooks 



#####credits:



##created with love & coffee by SUPCOM E-AGRICULTURE TEAM: 



#mohaned abid 

#ahmed tlili

#Dhia Eddine Daikhi

#Mohannad Lassioued

#Abdelwahed Rebhi



##framed by:

#madame Asma BEN LETAIFA

#madame Leila NAJJAR
####first we import all the  libraries that we are going to use 

#libraries for preproccesing and linar algebra

import numpy as np

import pandas as pd 

import os

#libraries for deep learning :we are going to use FASTAI as a framework on top of pytorch 

import torch

from fastai.vision import *

from fastai.metrics import error_rate

#visualization library

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots
####phase1:IMPORTATION,EXPLORATORY DATA ANALYSIS AND PRE PROCESSING  

traindf = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

traindf.head()
#checking train dataset columns and data types

traindf.info()
#some basic statistics on train dataset 

traindf.describe()
#exloring the data 

classdata = (traindf.healthy + traindf.multiple_diseases+

             traindf.rust + traindf.scab)

classdata.head()
any(classdata > 1)

#---->this means that is problem is is not a multiclassification problem 

#since all examples falls under only one of the 4 classes
#adding .jpg to help us load images later on 

traindf["image_id"] =traindf["image_id"].astype("str") + ".jpg"

traindf.head()
#now  lets define our classes to be:

# 0 for healthy

# 1 multiple_diseases

# 2 rust

# 3 scab

traindf["label"] = (0*traindf.healthy + 1*traindf.multiple_diseases+

             2*traindf.rust + 3*traindf.scab)

traindf.drop(columns=["healthy","multiple_diseases","rust","scab"],inplace=True)

traindf.head()
##some visual EDA to understand our data more 

#checking class unbalance

train_data = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

fig = go.Figure([go.Pie(labels=train_data.columns[1:],

           values=train_data.iloc[:, 1:].sum().values)])

fig.update_layout(title_text="Pie chart of targets", template="simple_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.show()
#distribution of healthy class

train_data["Healthy"] = train_data["healthy"].apply(bool).apply(str)

fig = px.histogram(train_data, x="Healthy", title="Healthy distribution", color="Healthy",\

            color_discrete_map={

                "True": px.colors.qualitative.Plotly[0],

                "False": px.colors.qualitative.Plotly[1]})

fig.update_layout(template="simple_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig



#scab class distribution 

train_data["Scab"] = train_data["scab"].apply(bool).apply(str)

fig = px.histogram(train_data, x="Scab", color="Scab", title="Scab distribution",\

            color_discrete_map={

                "True": px.colors.qualitative.Plotly[1],

                "False": px.colors.qualitative.Plotly[0]})

fig.update_layout(template="simple_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig
#rust distribution

train_data["Rust"] = train_data["rust"].apply(bool).apply(str)

fig = px.histogram(train_data, x="Rust", color="Rust", title="Rust distribution",\

            color_discrete_map={

                "True": px.colors.qualitative.Plotly[1],

                "False": px.colors.qualitative.Plotly[0]})

fig.update_layout(template="simple_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig
#multiple deseases distribution

train_data["Multiple diseases"] = train_data["multiple_diseases"].apply(bool).apply(str)

fig = px.histogram(train_data, x="Multiple diseases", color="Multiple diseases", title="Multiple diseases distribution",\

            color_discrete_map={

                "True": px.colors.qualitative.Plotly[1],

                "False": px.colors.qualitative.Plotly[0]})

fig.update_layout(template="simple_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig
#Creation of transformation object this object help

#augment our data by making transormations like flipping images and rotation...

#since more data means better results ;) (usually)

transformations = get_transforms(do_flip = True,

                                 flip_vert=True, 

                                 max_lighting=0.1, 

                                 max_zoom=1.05,

                                 max_warp=0.,

                                 max_rotate=15,

                                 p_affine=0.75,

                                 p_lighting=0.75

                                )
#this object is an encapsulation of our data it is a necessay step for the data to fit in a model under FASTAI

pathofdata = "/kaggle/input/plant-pathology-2020-fgvc7/"

data  = ImageDataBunch.from_df(path=pathofdata, 

                               df=traindf, 

                               folder="images",

                               label_delim=None,

                               valid_pct=0.2,

                               seed=100,

                               fn_col=0, 

                               label_col=1, 

                               suffix='',

                               ds_tfms=transformations, 

                               size=512,

                               bs=64, 

                               val_bs=32,

                               )
#some images and their correspanding classes

data.show_batch(rows=3, figsize=(10,7))
#normalizing is a necessay pre proccessing step to make the model generalize better on the data

#it carries out a simple function on eacch image:subtract the mean of pixels and divide by the variance

data = data.normalize()
####PHASE 2 :MODELING AND TRAINNING

#our model in a CNN architecture specificly a resnet34 a commun architecture used in computer vision tasks 

learner = cnn_learner(data, 

                      models.resnet34, 

                      pretrained=True

                      ,metrics=[error_rate, accuracy],).to_fp16()

learner.model_dir = '/kaggle/working/models'
#we set the hyperparameter leraning rate to be 0.002 (a value signifying how much we should update the weights at each iteration)

#after trying out a bunch of values 

#this value seems to work the best

#also we set epochs to be 10 due to time constraint 

#epachs is how many times does the model go through the whole dataset

lr = 0.002

epochs=10

#now we fit the resnet34 to our data 

#it takes about 50 minutes on kaggle (please activate GPU accelerator if running on kaggle)

learner.fit_one_cycle(epochs, lr)

#this saves the weights of the   model so that you can use it later on without trainning the model again

learner.save('mini_train')
#uncomment these lines to load pretrained model 

#learner.export()

#learner = load_learner(path="/kaggle/working/models")
#this shows a batch of the model predictions on the train dataset 

learner.show_results()
#### PHASE 3 RUNNING THE MODEL ON THE TEST DATASET 

#FIRST we import the test dataset

testdf = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

testdf.head()
#now lets get the paths of all test dataset images

pathofdata = "/kaggle/input/plant-pathology-2020-fgvc7/"

testdata= ImageList.from_folder(pathofdata+"images")

testdata.filter_by_func(lambda x: x.name.startswith("Test"))
#reading an image and predict its  classe with our model

img1 = open_image(testdata.items[0])

learner.predict(img1)

#--->the result is 3 which is the label of the scab desease
#### PHASE 4 PREPARING SUBMISSION FILE TO CHECK  OUR ACCURACY ON UNSEEN DATA 

#SUBMISSION DATAFRAME

resultlist = []

for item in testdata.items:

    img = open_image(item)

    predval = learner.predict(img)[2].tolist()

    predval.insert(0,item.name[:-4:])

    resultlist.append(predval)

resultdf = pd.DataFrame(resultlist)

resultdf.columns = sampsubmit.columns

resultdf.set_index("image_id",inplace=True)

resultdf = resultdf.loc[sampsubmit.image_id,:]

resultdf.reset_index(inplace=True)

resultdf.head()
#PREPARE SUBMISSION CSV YOU CAN FIND IT IN OUTPUT FOLDER ON THE RIGHT 

resultdf.to_csv("submit.csv",index=False)

#AFTER SUBMITING OUR RESULTS TO KAGGLE THE MODEL 

#PERFORMED OUTSTANDINGLY WELL SCORING 0.94405% OF ACCURACY ON UNSEEN TEST DATASET



'''THANK YOU 

END OF THE KERNEL'''
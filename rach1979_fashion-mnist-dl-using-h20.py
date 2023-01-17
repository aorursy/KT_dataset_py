# Objectives

# Image data multiclass classification and object identification - apparel and clothing

# Using H2O Deep Learning 

# 2 Parts - Part I without reducing dimensionality; Part II reducing dimensionality using random projection

# Dataset : Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. 

# Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from time import time



from sklearn.preprocessing import StandardScaler



import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
os.listdir("../input")
## H20 initiation



h2o.init()
## Read the dataset - both train and test using H20



train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")

test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
# Data exploration

train.shape

train.head(5)
#First convert into Pandas Dataframe



#Take any one row and first column till last column¶

#Reshape into 28x28 matrix

#use imshow to display the image
tmp_df = train.as_data_frame()
tmp_df.label.unique()
tmp_r = tmp_df.values[5,1:]

tmp_r.shape
tmp_r = tmp_r.reshape(28,28)
tmp_r.shape
plt.imshow(tmp_r)

plt.show()
##Data Modelling using H20 ##

##This data is a balanced set and no need to use SMOTE

##Separate out target and predictors from train data
y_target = train.columns[0]

y_target
X_predictors = train.columns[1:785]
train["label"] = train["label"].asfactor()
train["label"].levels()
# 4.1 Instantiate model
model_h2o = H2ODeepLearningEstimator(

                distribution="multinomial",

                activation="RectifierWithDropout",

                hidden=[50,50,50],

                input_dropout_ratio=0.2,

                standardize=True,

                epochs=1000

                )
##  Train model

start = time()

model_h2o.train(X_predictors, y_target, training_frame= train)

end = time()

(end-start)/60
result = model_h2o.predict(test)
result.shape

result.as_data_frame().head(5)
re = result.as_data_frame()
re["predict"]
re["actual"] = test["label"].as_data_frame().values
out = (re["predict"] == re["actual"])

np.sum(out)/out.size
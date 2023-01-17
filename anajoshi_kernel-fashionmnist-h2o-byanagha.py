### Labels



## Each training and test example is assigned to one of the following labels:



#0 T-shirt/top

#1 Trouser

#2 Pullover

#3 Dress

#4 Coat

#5 Sandal

#6 Shirt

#7 Sneaker

#8 Bag

#9 Ankle boot 



### Objective: Is to learn modeling with H2ODeeplearning  

#########
# Call libraries

# basic libraries

import pandas as pd

import numpy as np

import os

# For plotting

import matplotlib.pyplot as plt



# For measuring time elapsed

from time import time



# Model building

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator





# Change ipython options to display all data columns

pd.options.display.max_columns = 300
os.chdir("../input")

print(os.listdir("../input"))

#Start h2o

h2o.init()
train=h2o.import_file("fashion-mnist_train.csv")

test = h2o.import_file("fashion-mnist_test.csv")

#Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to

#     H2o dataframe

X_columns = train.columns[1:785]        # Only column names. No data

  # C1 to C786
y_columns = train.columns[0]

y_columns
train["label"]=train["label"].asfactor()
train['label'].levels()
model = H2ODeepLearningEstimator(

                             distribution="multinomial",

                             activation = "RectifierWithDropout",

                             hidden = [32,32,32],

                             input_dropout_ratio=0.2,  

                             standardize=True,

                             epochs = 500

                             )

start = time()

model.train(X_columns,

               y_columns,

               training_frame = train)





end = time()

(end - start)/60
result = model.predict(test[: , 1:785])
result.shape       # 10000 X 11
result.as_data_frame().head(10)   # Class-wise predictions
#      Convert H2O frame back to pandas dataframe

xe = test['label'].as_data_frame()

xe['result'] = result[0].as_data_frame()

xe.head()

xe.columns
#Accuracy

out = (xe['result'] == xe['label'])

np.sum(out)/out.size
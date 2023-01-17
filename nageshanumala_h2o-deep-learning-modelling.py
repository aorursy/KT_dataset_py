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



#Read file

train = pd.read_csv("fashion-mnist_train.csv")











train.shape       # (60000,785)
train.head(1)
train['label'].value_counts() # balanced classes 
# Get the first row excluding first column

#    First column contains class labels and 

#    other columns contain pixel-intensity values

abc = train.values[1, 1:]

abc.shape    # (784,)

abc = abc.reshape(28,28)   # Reshape to 28 X 28
# And plot it

plt.imshow(abc)

plt.show()
#Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to

#     H2o dataframe

X_columns = train.columns[1:786]        # Only column names. No data

   



y_columns = train.columns[0]

y_columns
train["label"].unique()


h2o.init()
train = h2o.import_file("../input/fashion-mnist_train.csv", destination_frame="train")

test = h2o.import_file("../input/fashion-mnist_test.csv", destination_frame="test")
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

result.shape       # 5730 X 3

result.as_data_frame().head(10)   # Class-wise predictions
#  Ground truth

#      Convert H2O frame back to pandas dataframe

xe = test['label'].as_data_frame()

xe['result'] = result[0].as_data_frame()

xe.head()

xe.columns
#Accuracy

out = (xe['result'] == xe['label'])

np.sum(out)/out.size
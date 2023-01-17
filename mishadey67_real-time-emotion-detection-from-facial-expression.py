# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/data-2/test.csv", dst = "../working/test.csv")

copyfile(src = "../input/data-2/train.csv", dst = "../working/train.csv")

copyfile(src = "../input/data1/Data_Preprocessing.py", dst = "../working/Data_Preprocessing.py")

copyfile(src = "../input/data4/CNN_Model.py", dst = "../working/CNN_Model.py")

copyfile(src = "../input/data1/Utils_funX.py", dst = "../working/Utils_funX.py")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from Utils_funX import *

from Data_Preprocessing import *

from CNN_Model import *

from keras.layers import *
train_data = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")
print("\nThe shape of the Traning samples = {} \n".format(train_data.shape))

train_data
# Description of the train dataset

train_data.describe()
#information of the training data set

train_data.info()
# Looking For the no. of null values in the Training dataset

train_data[train_data.columns].isna().sum()
# Looking for the Duplicate rows in the Test Dataset

train_data[train_data.duplicated()]
print("\nThe shape of the Testing samples = {} \n".format(test_data.shape))

test_data
test_data.describe()
test_data.info()
# Looking For the no. of null values in the Testing dataset

test_data[test_data.columns].isnull().sum()
# Looking for the Duplicate rows in the Test Dataset

test_data[test_data.duplicated()]
#Since we have no Null values in the dataset--> we only remov the Duplicates

train_data = remove_duplicates(train_data)

test_data = remove_duplicates(test_data)

print("\nThe shape of the Traning samples after Data Preprocessing is {} \n".format(train_data.shape))

print("\nThe shape of the Testing samples after after Data Preprocessing is {} \n".format(test_data.shape))
dictionary = {

    0:"Angry",

    1:"Disgust",

    2:"Fear",

    3:"Happy",

    4:"Sad",

    5:"Surprise",

    6:"Neutral" }

temp_dataset=train_data["emotion"].replace(dictionary)

temp_dataset=pd.DataFrame(temp_dataset)

Bar_Plots_For_Features(temp_dataset,"emotion")
Pie_Plots_For_Features(temp_dataset,"emotion")
x_train,y_train,x_test,y_test = Data_Preparation(train_data,test_data)
print("\nThe train and test data Shapes are :",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
x_train,x_test=Data_Normalization(x_train,x_test)
x_train,y_train,x_test,y_test
num_features = 64

n_classes = 7

batch_size = 64

epochs = 100

width = 48

height = 48
x_train = x_train.reshape(x_train.shape[0],width,height,1)

x_test = x_test.reshape(x_test.shape[0],width,height,1)
x_train.shape,y_train.shape
from CNN_Model import *
model= CNN_Model_Initialize(height,width,n_classes)
model.summary()
CNN_model_visualize(model)
x_train.shape,y_train.shape
from keras.utils import to_categorical
y_train = to_categorical(y_train,7)

print(y_train.shape)

y_train
y_test = to_categorical(y_test,7)

print(y_test.shape)

y_test
model,history=CNN_model_Compile_and_Train(model,x_train,y_train,1)
model.save('CNN_Model1.h5', include_optimizer=False)
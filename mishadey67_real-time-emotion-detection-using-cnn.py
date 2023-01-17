# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/data-2/test.csv", dst = "../working/test.csv")

copyfile(src = "../input/data-2/train.csv", dst = "../working/train.csv")

copyfile(src = "../input/cnn-ver-2/Data_Preprocessing.py", dst = "../working/Data_Preprocessing.py")

copyfile(src = "../input/cnn-ver-6/CNN_Model.py", dst = "../working/CNN_Model.py")

copyfile(src = "../input/cnn-ver-2/Utils_funX.py", dst = "../working/Utils_funX.py")

copyfile(src = "../input/model-ver1/CNN_Model1.h5", dst = "../working/CNN_Model1.h5")

copyfile(src = "../input/modelver2/CNN_Model2.h5", dst = "../working/CNN_Model2.h5")

copyfile(src = "../input/model-ver-3/CNN_Model1 3.h5", dst = "../working/CNN_Model3.h5")
!pip install plantcv
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
x_train = x_train.reshape(x_train.shape[0],width,height,)

x_test = x_test.reshape(x_test.shape[0],width,height,)
x_train.shape
from keras.utils import to_categorical



y_train = to_categorical(y_train,7)

print(y_train.shape)

y_train
y_test = to_categorical(y_test,7)

print(y_test.shape)

y_test
x_train[0]
np.argmax(y_train[0])
import matplotlib.pyplot as plt 

plt.style.use("dark_background")

for i in range(20):

    plt.imshow(x_train[i])

    plt.title(Decode_Y_Val(y_train[i]))

    plt.show()

# adding two type of data 1 -> vertically flipped and horizontally fliped

# therefore After augmentation: sizeof(Train_data) = sizeof(Train_data)*2*2

x_train,y_train=Data_Augmentation(x_train,y_train)
x_train.shape,y_train.shape
num_features = 64

n_classes = 7

batch_size = 64

epochs = 100

width = 48

height = 48
from CNN_Model import *
model= CNN_Model_Initialize(height,width,n_classes)
model.summary()
CNN_model_visualize(model)
x_train.shape,y_train.shape
from keras import *
#model,history=CNN_model_Compile_and_Train(model,x_train,y_train,3,200)

#model.save('CNN_Model1.h5', include_optimizer=False)
#model = models.load_model('CNN_Model1.h5')

#model,history=CNN_model_Compile_and_Train(model,x_train,y_train,3,180)

#model,history=CNN_model_Compile_and_Train(model,x_train,y_train,4,300)
#model = models.load_model('CNN_Model2.h5')

#model,history=CNN_model_Compile_and_Train(model,x_train,y_train,4,350)

#model.save('CNN_Model3.h5', include_optimizer=False)
model = models.load_model('CNN_Model3.h5')

model,history=CNN_model_Compile_and_Train(model,x_train,y_train,4,300)

model.save('CNN_Model4.h5', include_optimizer=False)
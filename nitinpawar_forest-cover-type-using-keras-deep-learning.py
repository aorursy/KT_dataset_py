# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
pd.options.display.max_rows = None

pd.options.display.max_columns = None
import keras

from keras import layers
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_train_feature = df_train.iloc[:,1:-1]
df_train_feature.head()
df_train_label = df_train.iloc[:,-1:]
df_train_label.head()
#correlation of features

correlation = df_train_feature.corr()
correlation
#as we can see Soil_Type7 and Soil_Type15 are having NaN values ,we can drop these columns

df_train_feature = df_train_feature.drop(columns={'Soil_Type7','Soil_Type15'})
#ensuring Soil_Type7 and Soil_Type15 are dropped :)

df_train_feature.head()
# We can see that for features from 'Elevation' to 'Horizontal_Distance_To_Fire_Points' are numerical

#and having values of different scales.

#will use StandardScalar to bring values to same scale

from sklearn.preprocessing import StandardScaler



StdScl = StandardScaler()



df_train_feature.iloc[:,0:10] = StdScl.fit_transform(df_train_feature.iloc[:,0:10])
#Define Keras sequential classifier

classifier = keras.Sequential()
#input layer

classifier.add(layers.Dense(128,activation='relu',input_shape=(52,)))



#dropout layer for input layer

# classifier.add(layers.Dropout(0.3))
#hidden layer 1

classifier.add(layers.Dense(64,activation='relu'))

#Drop out for hidden layer 1

# classifier.add(layers.Dropout(0.3))
#hidden layer 2

classifier.add(layers.Dense(32,activation='relu'))

#drop out for hidden layer2

# classifier.add(layers.Dropout(0.3))
#output layer

classifier.add(layers.Dense(8,activation='softmax'))
#summary report of classifier we have just built

print("Summary report of Keras classifier:") 

classifier.summary()
#Complie the classifier with mentioned hyper-parameters

#we have choosen loss function as 'sparse_categorical_crossentropy' because

# we are having more than 2 classes and no need to perform any lable encoding 



classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
keras_result = classifier.fit(df_train_feature,

                              df_train_label,

                              epochs=50,

                              batch_size=5) 
np.mean(keras_result.history['acc'])
#Its time to perform predictinos on test data

df_test = pd.read_csv('../input/test.csv')
df_test.head()
#will capture IDs as test labels

df_test_label = df_test.iloc[:,:1]
df_test_label.head()
#drop ID, Soil_Type7 and Soil_Type15

df_test_feature = df_test.drop(columns={'Id','Soil_Type7','Soil_Type15'})
#as performed standardization on train data ,will do similar for test data as well

df_test_feature.iloc[:,0:10] = StdScl.transform(df_test_feature.iloc[:,0:10])
df_test_feature.head()
df_test_feature.shape
#its time for predictionsss

class_predict = classifier.predict_classes(df_test_feature)
class_predict
class_predict = pd.DataFrame(class_predict,columns={'Cover_Type'})
class_predict.head()
keras_submission = df_test_label.join(class_predict)
keras_submission.head()
keras_submission.to_csv('keras_submission_new.csv',index=False)
ls
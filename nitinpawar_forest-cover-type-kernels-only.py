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
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_train_feature = df_train.iloc[:,1:-1]
df_train_feature.head()
df_train_label = df_train.loc[:,['Cover_Type']]
df_train_label.head()
from sklearn.preprocessing import StandardScaler
StdScl = StandardScaler()
df_train_feature.iloc[:,0:11].head()
df_train_feature.iloc[:,0:10] = StdScl.fit_transform(df_train_feature.iloc[:,0:10])
df_train_feature.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score,classification_report
RFC = RandomForestClassifier()
RFC.fit(df_train_feature,df_train_label)
RFC
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test_label = df_test.loc[:,['Id']]
df_test_feature = df_test.iloc[:,1:]
df_test_feature.iloc[:,0:10] = StdScl.transform(df_test_feature.iloc[:,0:10])
df_test_feature.head()
prediction = RFC.predict(df_test_feature)
prediction = pd.DataFrame(prediction,columns={'Cover_Type'})
prediction.head()
submission = df_test_label.join(prediction)
submission.head()
submission.to_csv('sample_upload_forest.csv',index=False)
import keras

from keras import layers
corr = df_train_feature.corr() 
#Feature importance

imp = pd.DataFrame(RFC.feature_importances_)
df_train_feature.shape
classifier = keras.Sequential()
#input layer 1

classifier.add(layers.Dense(128,activation='relu',input_shape=(54,)))
#dropout layer for input layer

classifier.add(layers.Dropout(0.3))
#hidden layer 1

classifier.add(layers.Dense(64,activation='relu'))

classifier.add(layers.Dropout(0.3))
#hidden layer 2

classifier.add(layers.Dense(32,activation='relu'))

# classifier.add(layers.Dropout(0.3))
#output layer

classifier.add(layers.Dense(8,activation='sigmoid'))
classifier.summary()
classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
df_train.head()
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train , y_test = train_test_split(df_train_feature,df_train_label,test_size=0.3,random_state=1)
keras_result = classifier.fit(X_train,y_train,epochs=50,batch_size=5,validation_data=(X_test,y_test)) 
np.mean(keras_result.history['acc'])
classifier.fit(df_train_feature,df_train_label,epochs=50,batch_size=5)
class_predict = classifier.predict_classes(df_test_feature)
class_predict = pd.DataFrame(class_predict,columns={'Cover_Type'})
class_predict.head()
keras_submission = df_test_label.join(class_predict)



# drop keras_submission
keras_submission.head()
keras_submission.to_csv('keras_submission',index=False)
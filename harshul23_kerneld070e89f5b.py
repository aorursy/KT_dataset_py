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
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("../input/train_V2.csv")
df1 = pd.read_csv("../input/test_V2.csv")



x_train=df.iloc[:,3:28].values
y_train=df.iloc[:,28].values
x_test=df1.iloc[:,3:28].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
x_train[:,12]=le.fit_transform(x_train[:,12])
x_test[:,12]=le.fit_transform(x_test[:,12])


onehot=OneHotEncoder(categorical_features=[12])
x_train=onehot.fit_transform(x_train).toarray()
x_test=onehot.fit_transform(x_test).toarray()
x_train[:,12]
print(x_train.shape)
for i in range(0,40):
    x_train[:,i]=x_train[:,i]/max(x_train[:,i])
for i1 in range(0,40):
    x_test[:,i1]=x_test[:,i1]/max(x_test[:,i1])    
print(x_train[:,12])
y_train=y_train.reshape(-1,1)
from sklearn.preprocessing import Imputer
im=Imputer(missing_values='NaN',strategy='mean',axis=0)
im=im.fit(y_train)
y_train=im.transform(y_train)
y_train=y_train.reshape(4446966)
print(y_train)
y_train=y_train.astype('float32')
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# lda=LDA(n_components=20)
# x_train=lda.fit_transform(x_train,y_train)
# x_test=lda.transform(x_test)


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=0)

print(x_train.shape)
model=Sequential()
model.add(Dense(512,activation='relu',input_dim=x_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))
model.compile(optimizer='Adam',loss='mean_squared_error',metrics=['mean_absolute_error'])
model.fit(x_train,y_train,epochs=5,batch_size=1024)
model.summary()
predictions= model.predict(x_test)
print(predictions.shape)
predictions=predictions.reshape(1934174 )
print(predictions)
my_submission = pd.DataFrame({'Id':df1.Id,'winPlacePerc':predictions})
my_submission.to_csv('submission.csv',index=False)
print('A submission file has been made')


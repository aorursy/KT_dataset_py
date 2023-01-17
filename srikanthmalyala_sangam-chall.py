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
Train_data=pd.read_csv("../input/Train.csv")

Test_data=pd.read_csv("../input/Test.csv")



Train_data.columns
Train_data=Train_data.drop(columns=['date_time','weather_type'],axis=1)

Train_data=pd.get_dummies(Train_data,columns=['is_holiday','weather_description'])

Train_data.columns

y=Train_data.traffic_volume

X=Train_data.drop(['traffic_volume'],axis=1)
from sklearn.model_selection import cross_val_score, train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)
print('plotting_feature importance')

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=20, max_features='sqrt')

clf = clf.fit(X_train, y_train)

features = pd.DataFrame()

features['feature'] = X_train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25,25))

plt.show()
from keras.layers import Dense, Softmax, Dropout

from keras.models import Sequential 

model_nn = Sequential()

from keras.layers.normalization import BatchNormalization

# layers

model_nn.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 60))

model_nn.add(BatchNormalization())

model_nn.add(Dropout(0.4))

model_nn.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))

model_nn.add(BatchNormalization())

model_nn.add(Dropout(0.4))

model_nn.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

model_nn.add(BatchNormalization())

model_nn.add(Dropout(0.4))

model_nn.add(Dense(units = 1, kernel_initializer = 'uniform',activation='sigmoid' ))



# Compiling the ANN

model_nn.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# Train the ANN

model_nn.fit(X_train, y_train, batch_size =64, epochs = 100)
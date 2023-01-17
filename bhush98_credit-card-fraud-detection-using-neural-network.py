#importing libraries

import pandas as pd
import numpy as np
# Reading and cleaning dataset

dataset = pd.read_csv('../input/creditcard.csv')
dataset.dropna()
dataset.head()
#Getting the no of columns

dataset.columns
# Getting the X_data

X_data = dataset.iloc[:,0:-1].values
X_data[0:2]
#Getting the Y_data

Y_data = dataset.iloc[:,-1].values
Y_data[0:5]
# Getting no of instances of each unique class

unique , counts = np.unique(Y_data,return_counts = True)
print(unique,counts)
# here 0 means not spam and 1 represents spam
# importing StandardScaler used for scaling

from sklearn.preprocessing import StandardScaler
sclaer = StandardScaler()
# Scaling X_data

X_data = sclaer.fit_transform(X_data)
X_data[0:2]
# Splitting the data into training and testing  

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_data,Y_data,test_size=0.3)
# Libraries that we will need to create ANN

from keras.models import Sequential
from keras.layers import Dense
# building our Neural Network

classifier = Sequential()
classifier.add(Dense(40 , input_dim = 30 , activation = 'relu'))
classifier.add(Dense(30 , input_dim = 40 , activation = 'relu'))
classifier.add(Dense(20 , input_dim = 30 , activation = 'relu'))
classifier.add(Dense(10 , input_dim = 20 , activation = 'relu'))
classifier.add(Dense(6 , input_dim = 10 , activation = 'relu'))
classifier.add(Dense(4 , input_dim = 6 , activation = 'relu'))
classifier.add(Dense(1, input_dim = 4 , activation = 'sigmoid'))
# Specifying our loss and optimizer

classifier.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
# Fitting our data to ANN

classifier.fit( x_train , y_train , epochs = 200 , batch_size = 500 )
# Getting the results

classifier.evaluate(x_test,y_test)
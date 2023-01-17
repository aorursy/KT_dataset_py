# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## LOADING LIBRARIES 

import os

import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.models import Sequential, Model

from keras.layers import Dense, Input
#### Loading the data

data = pd.read_csv("../input/BackOrders.csv",header=0)

data.head()
# Understand the Data  See the No. row and columns

data.shape
# Display the columns

data.columns
# Display the index

data.index
#Shows a quick statistic summary of your data using describe

data.describe(include='all')
# Display data type of each variable

data.dtypes
for col in ['sku', 'potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']:

    data[col] = data[col].astype('category')
# Cheeking the Variable int 

data.dtypes
# Delete sku attribute by cheeking the count of it

np.size(np.unique(data.sku, return_counts=True)[0])

data.drop('sku', axis=1, inplace=True)
data.isnull().sum()
# Observing the number of records before and after missing value records removal

print(data.shape)
# Since the number of missing values is about 5%. For initial analysis we ignore all these records

data = data.dropna(axis=0)

data.isnull().sum()

categorical_Attributes = data.select_dtypes(include=['category']).columns


data = pd.get_dummies(columns=categorical_Attributes, data=data, prefix=categorical_Attributes, prefix_sep="_",drop_first=True)
pd.value_counts(data['went_on_backorder_Yes'].values)
#Performing train test split on the data

X, y = data.loc[:,data.columns!='went_on_backorder_Yes'].values, data.loc[:,'went_on_backorder_Yes'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
#To get the distribution in the target in train and test

print(pd.value_counts(y_train))

print(pd.value_counts(y_test))
perceptron_model = Sequential()

perceptron_model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid', kernel_initializer='normal'))
perceptron_model.compile(loss='binary_crossentropy', optimizer='adam')
perceptron_model.fit(X_train, y_train, epochs=100)
test_pred=perceptron_model.predict_classes(X_test)

train_pred=perceptron_model.predict_classes(X_train)
confusion_matrix_test = confusion_matrix(y_test, test_pred)

confusion_matrix_train = confusion_matrix(y_train, train_pred)



print(confusion_matrix_train)

print(confusion_matrix_test)
TNR_Train = confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])

TPR_Train = confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train TNR: ",TNR_Train)

print("Train TPR: ",TPR_Train)

TNR_Test = confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])

TPR_Test = confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])



print("Test TNR: ",TNR_Test)

print("Test TPR: ",TPR_Test)
# The size of encoded and actual representations

encoding_dim = 16 

actual_dim = X_train.shape[1]
# Input placeholder

input_attrs = Input(shape=(actual_dim,))



# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='relu')(input_attrs)



# "decoded" is the lossy reconstruction of the input

decoded = Dense(actual_dim, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction

autoencoder = Model(input_attrs, decoded)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=100)
## Create a separate encoder model

# this model maps an input to its encoded representation

encoder = Model(input_attrs, encoded)
print(encoder.summary())
X_train_nonLinear_features = encoder.predict(X_train)

X_test_nonLinear_features = encoder.predict(X_test)
X_train = np.concatenate((X_train, X_train_nonLinear_features), axis=1)

X_test = np.concatenate((X_test, X_test_nonLinear_features), axis=1)
perceptron_model = Sequential()



perceptron_model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
perceptron_model.compile(loss='binary_crossentropy', optimizer='adam')
perceptron_model.fit(X_train, y_train, epochs=10)
test_pred=perceptron_model.predict_classes(X_test)

train_pred=perceptron_model.predict_classes(X_train)
confusion_matrix_test = confusion_matrix(y_test, test_pred)

confusion_matrix_train = confusion_matrix(y_train, train_pred)



print(confusion_matrix_train)

print(confusion_matrix_test)
Accuracy_Train=(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

TNR_Train= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])

TPR_Train= confusion_matrix_train[1,1]/(confusion_matrix_train[1,0]+confusion_matrix_train[1,1])



print("Train TNR: ",TNR_Train)

print("Train TPR: ",TPR_Train)

print("Train Accuracy: ",Accuracy_Train)
Accuracy_Test=(confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

TNR_Test= confusion_matrix_test[0,0]/(confusion_matrix_test[0,0] +confusion_matrix_test[0,1])

TPR_Test= confusion_matrix_test[1,1]/(confusion_matrix_test[1,0] +confusion_matrix_test[1,1])



print("Test TNR: ",TNR_Test)

print("Test TPR: ",TPR_Test)

print("Test Accuracy: ",Accuracy_Test)
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
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/Churn_Modelling.csv')

dataset.describe()

dataset.head(10)
#The first model, which include all the features

X1 = dataset.iloc[:,3:13].values

y  = dataset.iloc[:,13].values

#The subdataset for the second classifier, which has high salaries

X2 = dataset[['CreditScore','Age','EstimatedSalary','IsActiveMember']].values

#The third subdataset for the third classifier, which are lazy to change or close the bank accounts

X3 = dataset[['CreditScore','Age','Balance','IsActiveMember']].values
# Encoding categorical data,

# Here only Country and Gender feature need to be encode

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X1[:, 1] = labelencoder_X_1.fit_transform(X1[:, 1])

labelencoder_X_2 = LabelEncoder()

X1[:, 2] = labelencoder_X_2.fit_transform(X1[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])

X1 = onehotencoder.fit_transform(X1).toarray()

X1 = X1[:, 1:] #Avoid dummy variable traps
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

X2_train, X2_test = train_test_split(X2, test_size = 0.2, random_state = 0)

X3_train, X3_test = train_test_split(X3, test_size = 0.2, random_state = 0)
# Feature Scaling

# As we can see from the data, there are difference in scale of data in each catagory in the set.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X1_train = sc.fit_transform(X1_train)

X1_test = sc.transform(X1_test)

X2_train = sc.fit_transform(X2_train)

X2_test = sc.transform(X2_test)

X3_train = sc.fit_transform(X3_train)

X3_test = sc.transform(X3_test)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense
#Our first classifier

def build_classifier1():

    classifier1 = Sequential()

    classifier1.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

    classifier1.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    classifier1.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier1
def build_classifier2():

    classifier2 = Sequential()

    classifier2.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))

    classifier2.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))

    classifier2.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier2
#Now we going to see the results of the first model!

classifier1 = KerasClassifier(build_fn = build_classifier1, batch_size = 10, epochs = 100)

accuracies1 = cross_val_score(estimator = classifier1, X = X1_train, y = y_train, cv = 10, n_jobs = -1)

mean1 = accuracies1.mean()

variance1 = accuracies1.std()
#Next we going to see the results of the second model!

classifier2 = KerasClassifier(build_fn = build_classifier2, batch_size = 10, epochs = 100)

accuracies2 = cross_val_score(estimator = classifier2, X = X2_train, y = y_train, cv = 10, n_jobs = -1)

mean2 = accuracies2.mean()

variance2 = accuracies2.std()

#Last we going to test the results of the third model!

classifier3 = KerasClassifier(build_fn = build_classifier2, batch_size = 10, epochs = 100)

accuracies3 = cross_val_score(estimator = classifier2, X = X3_train, y = y_train, cv = 10, n_jobs = -1)

mean3 = accuracies3.mean()

variance3 = accuracies3.std()
print('The accuracy of our first model is:  ',mean1)

print('The vairance of accuracy is:         ',variance1)

print('The accuracy of our second model is: ',mean2)

print('The vairance of accuracy is:         ',variance2)

print('The accuracy of our third model is:  ',mean3)

print('The vairance of accuracy is:         ',variance3)
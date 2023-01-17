# Data Preprocessing

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/Data.csv')



# Creating Matrix of the features(independent Variables)

X = dataset.iloc[:, :-1].values



# Creating The dependent Variable Vector

y = dataset.iloc[:, 3].values



# Taking care of missing data (replacing with the mean)

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values ='NaN', strategy ="mean", axis = 0)



# Fitting the imputer object to the matrix of features X

imputer = imputer.fit(X[:, 1:3])



# Replacing the missing data by the mean of the column

X[:, 1:3] = imputer.transform(X[:, 1:3])
# Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])



#Dummy Encoding

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()



# Encoding Categorical data

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)
# Splitting the Dataset into the training Set and Test set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling(Standardisation and Normalisation)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

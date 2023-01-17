# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing titanic datasets(training and test sets)

titanic= pd.read_csv('../input/titanic/train.csv')

titanic_test= pd.read_csv('../input/titanic/test.csv')

print(titanic_test.head())
# Coverting certain features to categorical datatypes in traing and test datasets

titanic[['Pclass', 'Sex']]= titanic[['Pclass','Sex']].astype('category')

print(titanic.info())

titanic_test[['Pclass','Sex']]= titanic_test[['Pclass','Sex']].astype('category')

print(titanic_test.info())
# Extracting matrix of features from training set

X= titanic.iloc[:,[2,4,5,6,7,9]].values

y= titanic.iloc[:,1].values



# converting test data into array

X_test_sample= titanic_test.iloc[:,[1,3,4,5,6,8]].values

# Imputing values in training and test dataset

from sklearn.impute import SimpleImputer

imputer= SimpleImputer()

imputer.fit(X[:,[2]])

X[:,[2]]= imputer.transform(X[:,[2]])

imputer.fit(X_test_sample[:,[2,5]])

X_test_sample[:,[2,5]]= imputer.transform(X_test_sample[:,[2,5]])
# Encoding categorical variables in training and test dataset

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

ct= ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')

X  = np.array(ct.fit_transform(X))



ct_test= ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')

X_test_sample= np.array(ct_test.fit_transform(X_test_sample))



le= LabelEncoder()

X[:,3]= le.fit_transform(X[:,3])

X_test_sample[:,3]= le.fit_transform(X_test_sample[:,3])
# Avoiding dummy variable

X=X[:,1:]

X_test_sample= X_test_sample[:,1:]
# Splitting the training dataset into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2 , random_state=0)
# Feature scaling traing and test datasets

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train= sc.fit_transform(X_train)

X_test= sc.transform(X_test)

X_test_sample= sc.transform(X_test_sample)
# Importing Tensorflow

import tensorflow as tf



# Building Artificial Neural Network

ann= tf.keras.models.Sequential()



# Building input layer

ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

ann.add(tf.keras.layers.Dense(units=4, activation='relu'))

# Building output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# Compiling ann

ann.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])



# Fitting ann on training set

ann.fit(X_train, y_train, batch_size=10, epochs=100)
# Predicting results on training dataset

y_pred= ann.predict(X_test)

y_pred= (y_pred>0.5)



# confusion matrix

from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test, y_pred)

print(cm)
# Predicting values in training and test datasets

y_pred_sample= ann.predict(X_test_sample)

y_pred_sample= (y_pred_sample>0.5)



y_pred_sample= (1*y_pred_sample).ravel()



output= pd.DataFrame({"PassengerId": titanic_test.PassengerId, "Survived": y_pred_sample})

print(output)



# Printing results to a csv file

output.to_csv('Submission2.csv', index=False)
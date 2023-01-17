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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_train.describe()
df_train.columns
df_train.head()
df_test.columns
all_interesting_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 

          'Parch', 'Fare', 'Embarked', 'Survived']

df_train_no_missing_values = df_train[all_interesting_cols]

# Elimino variables con missing values, más adelante buscar otro enfoque. Sólo elimino las filas que

# tienen las columnas que me interesan (si hay columnas que vienen con missing values que no uso 

# no las saco)

df_train_no_missing_values = df_train_no_missing_values.dropna()



X_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 

          'Parch', 'Fare', 'Embarked']

X_train = df_train_no_missing_values[X_cols]

Y_train = df_train_no_missing_values.Survived



X_test = df_test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

#X_test = df_test

# Elimino variables con missing values, más adelante buscar otro enfoque

#X_test = X_test.dropna()
# Get list of categorical variables

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))



OH_cols_train.head()
# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_test.index = X_test.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_test = X_test.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
OH_X_test.PassengerId.dtype
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets

model.fit(OH_X_train,Y_train)
cols_with_missing = [col for col in OH_X_test.columns if OH_X_test[col].isnull().any()]

cols_with_missing
from sklearn.impute import SimpleImputer



# hago Imputación a columnas missing en el test dataset

my_imputer = SimpleImputer()

OH_X_test_with_imputed_values = pd.DataFrame(my_imputer.fit_transform(OH_X_test))

OH_X_test_with_imputed_values.columns = OH_X_test.columns
OH_X_test_with_imputed_values.PassengerId = OH_X_test.PassengerId
OH_X_test_with_imputed_values.head()
OH_X_test_noPassengerId  = OH_X_test_with_imputed_values.drop("PassengerId", axis=1).copy()



Y_test = model.predict(OH_X_test_noPassengerId)
OH_X_test_new_data.describe()
# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'PassengerId': OH_X_test_with_imputed_values.PassengerId,

                       'Survived': Y_test})

output.to_csv('submission.csv', index=False)
output.describe()
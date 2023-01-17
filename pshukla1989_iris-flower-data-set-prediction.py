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
iris_file_path ='/kaggle/input/iris-flower-dataset/IRIS.csv'

#uses one hot encoding to convert categorical into numerical data.

iris_df =pd.read_csv(iris_file_path)

iris_df.columns
iris_df.head()

pd.get_dummies(iris_df['species'])
iris_df.describe()
# y cannot be object dtype it should be numeric value i.e dummies wil be used.

y =pd.get_dummies(iris_df['species'])

iris_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X =iris_df[iris_features]

X.describe()
from sklearn.tree import DecisionTreeRegressor

iris_model= DecisionTreeRegressor(random_state =1)

iris_model.fit(X,y)
prediction = iris_model.predict(X.tail())

print("Model uses One-Hot Encoding.")

print("The predictions is : \n Iris-setosa Iris-versicolor Iris-virginica")

print(prediction)
# check whether the prediction of species from featueres is correct or not.

iris_df.head()

#Model Validation or calculating  MAE(Mean Absolute Error) to find quality of our model.

from sklearn.metrics import mean_absolute_error



predicted_species = iris_model.predict(X)

mean_absolute_error(y, predicted_species)
# split data into training and validation data, for both features and target

# method is used to observe the prediction for same data as well as data which is not included in training data 

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model

iris_model = DecisionTreeRegressor()

# Fit model

iris_model.fit(train_X, train_y)



# get predicted values on validation data

# calculating MAE to check the efficiency of our model 

val_predictions = iris_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
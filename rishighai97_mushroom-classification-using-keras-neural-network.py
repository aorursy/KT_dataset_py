'''

Dataset analysis and summary of approach

-----------------------------------------

1. On observing the dataset, we can easily conclude that the entire dataset is categorical data.

2. As we know, ML models only work on numeric data, so the first task that comes in hand is to convert the data to numeric format

3. The desired ouput classitication is binary (class-p:poisonous,e:edible)

   So this data can be converted into 0,1.

4. But in case of independent variables, label enocoding the data (e.g: cap-shape to 0,1,2 etc),

   would mean that the variables are ranked from 0-n which is wrong as cap-shape are categories.

5. So we perform one hot encoding on these variables.

   eg: cap_shape_x will have 0 for absence and 1 for presence of x shape

   So now the data is translated in the format that can be easily understood by ML models.

6. As a part of feature engineering, we calculate correlation matrix and select features having correlation with dependent variable(result)>0,5

7. We use neural network to perform classification.

'''
# imports

import pandas as pd

import numpy as np

import keras

from numpy.random import seed

from scipy.stats import pearsonr

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split
# viewing the dataset

dataset = pd.read_csv("../input/mushroom-classification/mushrooms.csv")

display(dataset.head())
# check for any null values in dataset

display(dataset.isnull().any())
# splitting dataset in independent(data) and dependent variables(y)

y = dataset.iloc[:,0]

data = dataset.iloc[:,1:]

# Converting categorical data to numeric

y = y.replace(to_replace=["e","p"],value=[0,1])

data = pd.get_dummies(data)

# combining data and y anc calculating correlation matrix

data["result"] = y

correlation_matrix = data.corr()

display(correlation_matrix.head())
# getting columns whose correlation with dependent variable(result)>0.5 

columns=[]

for column,value in zip(correlation_matrix.loc["result"].index,correlation_matrix.loc["result"]):

    if(abs(value)>0.5):

        columns.append(column)

data = data[columns]
# getting X and y from data and performing train test split

# 33 percent data is used for testing

X = data.iloc[:,:-1]

y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# creating a neurel net with 

# 2 layers with 50 neurons

# relu layers as they are less computational expensive and more efficient in intermediate layers

# sigmoid is used in output as it classifies data based on continuous values between 0 and 1 based on a threshold

model = Sequential()

model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)
# evaluating model accuracy by fitting test data

_, accuracy = model.evaluate(X_test, y_test)

print('Accuracy: %.2f' % (accuracy*100))
'''

I hope using the model you pick up fresh mushrooms for your dishes :P

Do comment your views  on the kernel.

'''
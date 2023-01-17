#Necessary packages imported

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import ensemble, model_selection

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score

from yellowbrick.regressor import ResidualsPlot



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Read the csv file and create a data frame

train_df = pd.read_csv('../input/ph-recognition/ph-data.csv')

train_df.head()
#Create the target variable and the features to train

y = train_df.label



X = train_df[['blue', 'green', 'red']]
#Lets see the description of data

X.describe()
#Visualisation of data with respect to ph values

plt.figure(figsize=(15,5))

#plotting blue spectrum with ph

plt.subplot(1,3,1)

plt.scatter(X.blue,y)

plt.xlabel('Blue color')

plt.ylabel('ph value')



#plotting red spectrum with ph

plt.subplot(1,3,2)

plt.scatter(X.red,y)

plt.xlabel('Red color')

plt.ylabel('ph value')



#plotting green spectrum with ph

plt.subplot(1,3,3)

plt.scatter(X.green,y)

plt.xlabel('Green color')

plt.ylabel('ph value')
#Splitting the data into train and test

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,

                                                                    test_size=0.25, random_state=42)
X_train.shape
#Initiate a random forest model and train it

rf_model = ensemble.RandomForestRegressor(random_state=12)

rf_model.fit(X_train, y_train)
#Prediction on the test data

y_pred = rf_model.predict(X_test)

#Calculation of Mean Absolute Error

mae = mean_absolute_error(y_test, y_pred)

#Calculation of R Squared value

r2_val = r2_score(y_test, y_pred)

print('Mean Absolute Error of the model is: ', mae)

print('R Squared value is: ', r2_val)
#Visualizing the model predictions and errors

fig, ax = plt.subplots(figsize=(10,5))

roc_viz = ResidualsPlot(rf_model)

roc_viz.fit(X_train, y_train)

roc_viz.score(X_test, y_test)

roc_viz.show()
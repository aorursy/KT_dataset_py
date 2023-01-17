import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from datetime import datetime

from sklearn.metrics import accuracy_score,mean_absolute_error
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read data

train_data = pd.read_csv('../input/Confirm_datasets_clean (4).csv')
#testing dataset

test_data = []

for i in range(161,223):

    test_data.append(i)

test_data = pd.DataFrame(test_data)
#training dataset

print(train_data.columns.tolist())

#data we use for training

X = train_data['Date']

X = pd.DataFrame(X)

#target

y = train_data['Number of cases']

y = pd.Series(y)
#model

lr =  LinearRegression(copy_X = False, n_jobs = -1)

#fit with training dataset

lr.fit(X,y)

#predict

submit = lr.predict(test_data)
#score methods (MAE,MLE, cross validation, ...)

from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(lr, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores.mean())
#submission

print(test_data.columns.tolist())

submission = pd.DataFrame({'Date': test_data[0].tolist(),

                           'Number of cases': submit.tolist()})

submission.to_csv("../../kaggle/working/Confirm_prediction.csv", index=False)
submission.head()
#plots

plt.scatter(test_data[0].tolist(), submit,  color='black')

plt.plot(test_data[0].tolist(), submit, color='blue', linewidth=3)

plt.show()

    
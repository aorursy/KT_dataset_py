# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/financial-data/Data-1.xlsx')

df.head()
# Import the libraries

from random import randint

from sklearn.linear_model import LinearRegression



TRAIN_INPUT = list()

TRAIN_OUTPUT= list()



for i in range(len(df['GDP'])):

    a = df['short_lr_n'][i]

    b = df['CONS'][i]

    c = df['GDP'][i]

    op = df['cpi_inf_arm'][i]

    TRAIN_INPUT.append([a,b,c])

    TRAIN_OUTPUT.append(op)



predictor = LinearRegression(n_jobs=-1) #Create a linear regression object NOTE n_jobs = the number of jobs to use for computation, -1 means use all processors

predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)  #fit the linear model (approximate a target function)



X_TEST = [[10,20,30]]

outcome = predictor.predict(X=X_TEST)



coefficients = predictor.coef_  #The estimated coefficients for the linear regression problem.



print('Outcome: {} \n Coefficients: {}'.format(outcome, coefficients))
s, predictions = [], []

for i in range(len(df['GDP'])):

    a = df['short_lr_n'][i]

    b = df['CONS'][i]

    c = df['GDP'][i]

    real = predictor.predict(X=[[a, b, c]])

    predicted = df['cpi_inf_arm'][i]

    predictions.append(predicted)

    s.append(abs(predicted-real)/predicted * 100)

    

print('Accurace rate: ', 100 - sum(s)/len(df['GDP']), '%')
plt.figure(dpi=200)

plt.plot(df['data'], df['cpi_inf_arm'])

plt.xticks([])

plt.title("Real Data")

plt.xlabel("Years from 2000 to 2018")

plt.ylabel("CPI rate")
plt.figure(dpi=200)

plt.plot(df['data'], predictions, color='C1')

plt.xticks([])

plt.title("Predicted Data")

plt.xlabel("Years from 2000 to 2018")

plt.ylabel("CPI rate")
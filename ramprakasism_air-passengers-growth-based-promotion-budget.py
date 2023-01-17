from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

air=pd.read_csv("/kaggle/input/AirPassengers.csv")

air.shape
air.columns
air.describe()
#Find the correlation between number of passengers and promotional budget.

np.corrcoef(air.Passengers, air.Promotion_Budget)
#Find the correlation between number of passengers and Service_Quality_Score.

np.corrcoef(air.Passengers, air.Service_Quality_Score)
plt.scatter(air.Passengers, air.Promotion_Budget)
plt.scatter(air.Passengers, air.Service_Quality_Score)
import statsmodels.api as sm

y=air[['Passengers']]

X=air[['Promotion_Budget']]

#adding constant to make beeter pridiction

x1=sm.add_constant(X)

x1
#Modelling using ordinary least squaring method (OLS).

model = sm.OLS(y,x1)

#Calculating all correlated constant and model generatoin

fitted1=model.fit()

fitted1.summary()
#Regression Line

plt.scatter( air.Promotion_Budget,air.Passengers)

plt.plot( air.Promotion_Budget,fitted1.fittedvalues,c="g")
fitted1.predict([1,650000])
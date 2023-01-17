import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

import seaborn as sns
Data = pd.read_csv("../input/SolarPrediction.csv")

Data.head(10)
Data.info()
X = Data.iloc[:, 4:8]    

Y = Data['Radiation']

X.head()
X_train, X_test, y_train, y_test = train_test_split(

  X, Y, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape )

print(y_train.shape,y_test.shape)
corr = Data.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.show()
X1 = Data['Temperature'] 

X2 = Data['Pressure']

print ('Temperature and Radiation',np.corrcoef(X1,Y),'\n')

print ('Pressure and Radiation',np.corrcoef(X2,Y),'\n')
from sklearn import linear_model

reg = linear_model.BayesianRidge()

reg.fit(X_train ,y_train)

reg.score(X_test, y_test)
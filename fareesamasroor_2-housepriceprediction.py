# Importing the libraries

import numpy as np

#import matplotlib.pyplot as plt

import pandas as pd

import pickle



dataset = pd.read_csv('../input/train_hp.csv')



dataset = dataset.dropna()





X = dataset.iloc[:, : 29]



y = dataset.iloc[:, -1]





#Splitting Training and Test Set

#Since we have a very small dataset, we will train our model with all availabe data.



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()



#Fitting model with trainig data

regressor.fit(X, y)



print(regressor.predict([[60,65,8450,7,5,2003,2003,706,0,150,856,856,854,1710,1,0,2,1,3,1,8,2003,2,548,0,0,0,2,2008]]))
import pandas as pd

train_hp = pd.read_csv("../input/train_hp.csv")
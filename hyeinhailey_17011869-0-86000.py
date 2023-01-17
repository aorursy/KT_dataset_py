import torch

import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler()



torch.manual_seed(111)

xy=pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv', skiprows=1, header=None, usecols=range(1,10))

xtest=pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv', skiprows=1, header=None, usecols=range(1,9))

submit=pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')

x=xy.drop(9, axis=1)

y=xy[9]



sc.fit(x)

x=sc.transform(x)

xtest=sc.transform(xtest)



#x[0:5]
regressor=KNeighborsRegressor(n_neighbors=12, weights='distance')

regressor.fit(x,y)



guesses=regressor.predict(xtest)

guesses= (guesses >= 0.5) * 1.0

guesses
for i in range(len(guesses)):

  submit['Label'][i]=guesses[i]



submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header=True, index=False)
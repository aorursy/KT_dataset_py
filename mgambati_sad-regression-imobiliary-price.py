import requests

import pandas as pd

import seaborn as sns

import numpy as np

from sklearn import linear_model

from sklearn import preprocessing

import os
print(os.listdir("../input"))
dataFrame = pd.read_csv("../input/vivareal-listings-cpa.csv")

dataFrame.head()
PricePerAreaFrame = dataFrame[['price', 'usableAreas']]

PricePerAreaFrame.head()
sns.pairplot(data=PricePerAreaFrame, kind="reg")
labelEncoder = preprocessing.LabelEncoder()

regressionModel = linear_model.LinearRegression(normalize=True)
X = np.array(PricePerAreaFrame['usableAreas']).reshape(-1, 1)

y = labelEncoder.fit_transform(PricePerAreaFrame['price'])

regressionModel.fit(X, y)
tamanho = 270

print('Valor: ',regressionModel.predict(np.array(tamanho).reshape(-1, 1)))
bedroomsFrame = dataFrame[['price', 'bedrooms', 'usableAreas']]

bedroomsFrame.head()
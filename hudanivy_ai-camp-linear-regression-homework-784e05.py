import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import matplotlib.pyplot as plt
import math 
from patsy import dmatrices

data = pd.read_csv("../input/kc_house_data.csv")
data.head()
data.corr()['price']

plt.title('grade vs price')
plt.scatter(data['grade'],data['price'])
plt.show()

plt.title('sqft_living vs price')
plt.scatter(data['sqft_living'],data['price'])
plt.show()

plt.title('bedrooms vs price')
plt.scatter(data['bedrooms'],data['price'])
plt.show()

plt.title('bathrooms vs price')
plt.scatter(data['bathrooms'],data['price'])
plt.show()

plt.title('condition vs price')
plt.scatter(data['condition'],data['price'])
plt.show()
Y, X = dmatrices('price~C(bedrooms)+C(bathrooms)+C(condition)+sqft_living+floors+waterfront+view+grade+sqft_above+sqft_basement', data, return_type='dataframe')
xtrain, xvali, ytrain, yvali = train_test_split(X, Y, test_size=0.3, random_state=0)
xtrain = np.asmatrix(xtrain)
xvali = np.asmatrix(xvali)
ytrain = np.ravel(ytrain)
yvali = np.ravel(yvali)
model = LinearRegression()
model.fit(xtrain, ytrain)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
pred=model.predict(xtrain)
print("train set relative MSE ",(abs(pred-ytrain)/ytrain).sum() / len(ytrain))
pred=model.predict(xvali)
print("vali set relative MSE ",(abs(pred-yvali)/yvali).sum() / len(yvali))

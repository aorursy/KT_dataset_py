import pandas as pd

import numpy as np 

from sklearn import linear_model

import matplotlib.pyplot as plt
df = pd.read_csv(r'../input/kc_house_data.csv')

df.head()
%matplotlib inline

plt.xlabel('area')

plt.ylabel('price')

plt.scatter(df.sqft_lot,df.price,color='blue',marker='+')
x = df[['bedrooms','bathrooms','sqft_lot','floors']]

x.head()
y = df.price

y.head()
# Create linear regression object

reg = linear_model.LinearRegression()

reg.fit(x,y)
reg.predict([[3,1.00,5650,1.0]])
reg.coef_
reg.intercept_
p = reg.predict(x)

p
final = df[['bedrooms','bathrooms','sqft_lot','floors']]

final.head()
final['price'] = p

final.head()
'''We have completed .... In a Simple Cleaner Code'''
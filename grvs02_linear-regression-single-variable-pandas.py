'''Linear regression is for "prediction" else other are for 'classification'.linear regression is for for single variable as well as multiple variable both.

basic line of predction is: y = mx + c i.e price = m x area + b , where m = slope(gradient) , b = Y intercept '''
import pandas as pd

import numpy as np

from sklearn import linear_model

import matplotlib.pyplot as plt



df = pd.read_csv(r'../input/kc_house_data.csv')

df.head()
%matplotlib inline

plt.xlabel('Square_Foot')

plt.ylabel('Price')

plt.scatter(df.sqft_living,df.price,color='red',marker='+')
area = df[['sqft_living']]

area.head()
price = df.price

price.head()
# Create linear regression object

reg = linear_model.LinearRegression()

reg.fit(area,price)
reg.predict([[3000]])
reg.coef_
reg.intercept_
'''Y = m * X + b (m is coefficient and b is intercept)'''
3000*280.6235679 + (-45380.7430944728)
df_area = df.sqft_living

df_area.head()

df_area.values.reshape(-1,1)

p = reg.predict(df_area) #Reshaped to 2D Array Even Though Not Working..Sorry.

p.head()
%matplotlib inline

plt.xlabel('area', fontsize=20)

plt.ylabel('price', fontsize=20)

plt.scatter(df.sqft_living,df.price,color='red',marker='+')

plt.plot(df.sqft_living,reg.predict(df[['sqft_living']]),color='blue')
import pandas as pd

import numpy as np
df=pd.read_csv("../input/headbrain-lr/headbrain.csv")
df.shape
from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()
x=df.iloc[:,0:1].values #range to convert data from one d array to 2d array

y=df.iloc[:,1].values

lr_model.fit(x,y)
#Since y=mx+c:

m=lr_model.coef_  #slope

c=lr_model.intercept_ #intercept
m,c
y_predict=(m*x)+c
y_predict #predicted value of y
import matplotlib.pyplot as plt
plt.scatter(x,y) #blue dot=actual value

plt.plot(x,y_predict,c="red") #red line=predicted value

plt.show()
y_predict=lr_model.predict(x) #y_predict=mx+c
#suppose

head=4002

bweight=lr_model.predict([[head]])

plt.scatter(x,y)  #actual data with blue dots

plt.plot(x,y_predict,c="red") #prediction line

plt.scatter([head],bweight,c="yellow") #predicted value for input data

plt.show()
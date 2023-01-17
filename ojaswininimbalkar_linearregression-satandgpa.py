import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
df=pd.read_csv("../input/SATandGPA_LinearRegression.csv")

df.head()
df
df.shape  #prints (number of rows,number of columns)
df.keys()  #displaying column names and types


y=np.array(df['SAT']).reshape(-1, 1) #reshape(-1,1) means unknown rows(till end of column) and 1 means 1 column

x=np.array(df['GPA']).reshape(-1, 1) 

model=LinearRegression()   
model.fit(x,y)   #passing the data to be trained

SAT=model.predict([[4]])  #example to show our model is actually working

SAT
plt.scatter(x,y) #original data plot

y_predict=model.predict(x) #y=mx+c

plt.plot(x,y_predict) #plotting our trained model
#original data as scattered points and predicted model with best fit line shown with red color

plt.scatter(x,y)

plt.plot(x,y_predict,c="red")
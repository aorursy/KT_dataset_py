import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score,r2_score
from ipywidgets import interact
df = pd.read_csv('../input/Admission_Predict.csv')
df = df[['CGPA','Chance of Admit ']]

df.rename(columns = {'CGPA':'CGPA','Chance of Admit ':'Chance of Admit '},inplace=True)
x = df.iloc[:,0:1].values
y = df['Chance of Admit ']

model = LinearRegression()
model.fit(x,y)
m = model.coef_
c = model.intercept_
y_predict = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_predict,c="red")
r2_score(y_predict,y)

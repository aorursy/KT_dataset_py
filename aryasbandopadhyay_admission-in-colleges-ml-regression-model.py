import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('Admission_Predict_Ver1.1.csv')
data.head()
fig,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.05 ,cmap="magma")
plt.show()
data.dropna(inplace=True)
y=data.iloc[:,:-1].values
x=data.iloc[:,1:8].values
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=94)
lin.fit(xtrain,ytrain)
ypred=lin.predict(xtest)
from sklearn.metrics import r2_score
r2_score(ytest,ypred)
xtrain1,xtest1,ytrain1,ytest1=train_test_split(x,y,test_size=0.2,random_state=14)
from sklearn.ensemble import RandomForestRegressor
RFG=RandomForestRegressor(n_estimators=100)
RFG.fit(xtrain,ytrain)
y_pred1=RFG.predict(xtest)
r2_score(ytest,y_pred1)



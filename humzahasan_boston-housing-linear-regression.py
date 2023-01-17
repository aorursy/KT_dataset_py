#Import required libraries along with the toy dataset of Boston Housing already presnt in sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = load_boston()
boston
boston.data
boston.target
boston.feature_names
x = pd.DataFrame(data=boston.data,columns=boston.feature_names)
x
y = pd.DataFrame(data=boston.target,columns=['PRICE'])
y
df = pd.concat([x,y],axis=1)
df
df.info()
df.describe()
df.corr()
sns.heatmap(df.corr(),annot=True)
sns.pairplot(df)

y=boston.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.22,random_state=42)
x_train.shape
x_test.shape
lr = LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_train,y_train) #Accuracy Score of 75%
predlr=lr.predict(x_test)
y_test
predlr

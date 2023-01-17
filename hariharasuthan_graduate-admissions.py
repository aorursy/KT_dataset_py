import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

data.drop(['Serial No.'],axis=1,inplace=True)
data.info()
data.corr()
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(data.corr(),annot=True,fmt='.1f',linewidth=0.5,ax=ax)

plt.show()
data.head()
plt.figure(figsize=(10,10))

sns.barplot(x=data['University Rating'],y=data['GRE Score'],palette=sns.cubehelix_palette(len(data['University Rating'].unique())))

plt.xlabel('University Rating')

plt.ylabel('GRE Score')

plt.title('Uni. Rating v GRE')
plt.figure(figsize=(10,10))

sns.barplot(x=data['University Rating'],y=data['TOEFL Score'],palette=sns.cubehelix_palette(len(data['University Rating'].unique())))

plt.xlabel('University Rating')

plt.ylabel('TOEFL Score')

plt.title('Uni. Rating v TOEFL')
UR=data['University Rating'].value_counts()
f,ax=plt.subplots(figsize=(10,10))

ax.pie(UR.values, labels=UR.index, autopct='%1.1f%%')

#ax.axis('equal')

plt.show()
sns.countplot(data['University Rating'],hue=data['Research'])

plt.show()
sns.distplot(data['GRE Score'],hist=True,kde=False,color='orange',hist_kws={'edgecolor':'black'})

plt.ylabel('Occurence')
sns.distplot(data['TOEFL Score'],hist=True,kde=False,color='blue',hist_kws={'edgecolor':'black'})

plt.ylabel('Occurence')
sns.lmplot(x='University Rating',y='CGPA',data=data)

plt.xlabel('Uni. Rating')

plt.ylabel('CGPA')
sns.lmplot(x='GRE Score',y='Chance of Admit ',data=data)

plt.xlabel('GRE Score')

plt.ylabel('Chance of Admit')
sns.pairplot(data.drop(['Research','University Rating'],axis=1))
y=data['Chance of Admit ']

x=data.drop(['Chance of Admit '],axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error

linear=LinearRegression()

linear.fit(xtrain,ytrain)

lrpredict=linear.predict(xtest)

lr2=r2_score(ytest,lrpredict)

lmse=np.sqrt(mean_squared_error(ytest,lrpredict))

print("r2:",lr2)

print("rmse:",lmse)
from sklearn.preprocessing import PolynomialFeatures

p=PolynomialFeatures(degree=2)

xp=p.fit_transform(xtrain)

xt=p.fit_transform(xtest)

plinear=LinearRegression()

plinear.fit(xp,ytrain)

plrpredict=plinear.predict(xt)

plr2=r2_score(ytest,plrpredict)

pmse=np.sqrt(mean_squared_error(ytest,plrpredict))

print("r2:",lr2)

print("rmse:",pmse)
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100)

rf.fit(xtrain,ytrain)

rfpredict=rf.predict(xtest)

rr2=r2_score(ytest,rfpredict)

rrmse=np.sqrt(mean_squared_error(ytest,rfpredict))

print("r2:",rr2)

print("rmse:",rrmse)
from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(xtrain,ytrain)

dtpredict=dt.predict(xtest)

dtr2=r2_score(ytest,dtpredict)

dtmse=np.sqrt(mean_squared_error(ytest,dtpredict))

print("r2:",dtr2)

print("rmse:",dtmse)
errors=pd.DataFrame({'RMSE':[lmse,pmse,rrmse,dtmse],'R2':[lr2,plr2,rr2,dtr2]},index=['LR','PLR','RF','DT'])
errors.plot(kind='bar')
plt.bar(['Linear','Polynomial','RandomForest','DecisionTree'],np.array([lr2,plr2,rr2,dtr2]))

plt.xlabel("Type of Regressor")

plt.ylabel("R2 Score")

plt.show()
plt.bar(['Linear','Polynomial','RandomForest','DecisionTree'],np.array([lmse,pmse,rrmse,dtmse]))

plt.xlabel("Type of Regressor")

plt.ylabel("RMSE Score")

plt.show()
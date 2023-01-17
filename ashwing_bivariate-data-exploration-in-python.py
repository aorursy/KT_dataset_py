import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
ds = pd.read_csv("../input/train.csv")
ds.info()
ds.describe()
ds[['Sex','Survived','PassengerId']].groupby(['Sex','Survived']).count()
fig,ax = plt.subplots(2,2,figsize=(12,12))

sns.barplot(x="Sex",y='Survived',data=ds,estimator = (lambda x: sum(x==1)/len(x)*100),ci=0,ax=ax[0,0])

sns.barplot(x="Pclass",y='Survived',data=ds,estimator = (lambda x: sum(x==1)/len(x)*100),ci=0,ax=ax[0,1])

sns.barplot(x="Embarked",y='Survived',data=ds,estimator = (lambda x: sum(x==1)/len(x)*100),ci=0,ax=ax[1,0])

sns.barplot(x="Parch",y='Survived',data=ds,estimator = (lambda x: sum(x==1)/len(x)*100),ci=0,ax=ax[1,1])
fig,ax = plt.subplots(1,2,figsize=(8,5))

sns.stripplot(x='Survived',y='Age',data=ds,jitter=True,ax=ax[0])

sns.boxplot(x="Survived", y="Age",data=ds,ax=ax[1])
fig,ax = plt.subplots(1,2,figsize=(8,5))

sns.stripplot(x='Survived',y='Fare',data=ds,jitter=True,ax=ax[0])

sns.boxplot(x="Survived", y="Fare",data=ds,ax=ax[1]);
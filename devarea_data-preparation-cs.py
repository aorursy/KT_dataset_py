import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb
df = pd.read_csv('../input/pupils-ex.csv')
df.info()
df.head()
df.Height.fillna(df.Height.mean(),inplace=True)
df.loc[df.Age > 12,'Age'] = 12
df.loc[df.Height.isnull(),'Height'] = df.Age*12
ls = [2,4,6,8,10]*10

for k,v in df.iterrows():

    if np.isnan(v.Height):

            df.loc[k,'Height'] = ls[v.Age]
df.drop(df.loc[df.Age <= 8].index,inplace=True)
df.drop(['type'],axis=1,inplace=True)
df['gen'].replace({'M':1 , 'F':2},inplace=True)
df['namelen']=df.Name.apply(len)
def myop(inc):

    if inc < 10000:

        return 1

    else:

        if inc < 30000:

            return 2

        else:

            return 3
df['new_inc'] = df.income.apply(myop)
def myop2(r):

    return r.Age*r.income
df['new2'] = df.apply(myop2,axis=1)
np.digitize(df.Height, [110,130,150,160], right=True)
pd.get_dummies(df.type).head()
pd.cut(df.Height,bins=3,labels=['S','M','L'])
pd.qcut(df.Height,q=[0,0.1,0.8,1],labels=['S','M','L'])
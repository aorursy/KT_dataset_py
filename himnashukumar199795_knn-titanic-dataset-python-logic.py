import pandas as pd

import numpy as np

import math

import operator

import seaborn as sb

from bs4 import BeautifulSoup
tnc=sb.load_dataset('titanic')

tnc.head(10)
tnc.dropna(axis=0,how='any',subset=['age','fare'],inplace=True)
test=tnc[-200:]

train=tnc[:500]

for k in range(1,50):    

    for key,val in test.iterrows():

        dist=[]

        for key1,val1 in train.iterrows():

            dist.append(abs(val1['age']-val['age']))

        train['dist']=dist

        train.sort_values(by=['dist'],inplace=True)

        test.loc[key,'pred_fare'+str(k)]=train.loc[train['dist'].isin(train['dist'].unique()[:k]),'fare'].mean()



mae=[]

rmse=[]

for key,val in test.loc[:,'pred_fare1':'pred_fare49'].iteritems():

    tmp=abs(test['fare']-val)

    mae.append(tmp.mean())

    tmp=(test['fare']-val)**2

    rmse.append(tmp.mean()**(1/2))

print(mae)

print(rmse)
sb.lineplot(x=list(range(1,50)),y=mae)
sb.lineplot(x=list(range(1,50)),y=rmse)
sb.scatterplot(train['fare'],train['age'],hue=train['fare'])
import os

import json

from pathlib import Path

from glob import glob

import matplotlib.pyplot as plt

%matplotlib inline
from statsmodels.formula.api import quantreg

import pandas as pd
train = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/train.csv' )

test  = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/test.csv' )

train['traintest'] = 0

test ['traintest'] = 1

submission  = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )

submission['Weeks']   = submission['Patient_Week'].apply( lambda x: int(x.split('_')[-1]) )

submission['Patient'] = submission['Patient_Week'].apply( lambda x: x.split('_')[0] ) 

train.head(4)
test.head(2)
submission.head(5)
train = pd.concat( (train,test) )

train.sort_values( ['Patient','Weeks'], inplace=True )

import seaborn as sns

corrmat = train.corr() 

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
train['SmokingStatus'].value_counts()

z=train.groupby(['SmokingStatus','FVC'])['Weeks'].count().to_frame().reset_index().head()

z.style.background_gradient(cmap='Blues') 
z=train.groupby(['SmokingStatus','Weeks'])['FVC'].count().to_frame().reset_index().head()

z.style.background_gradient(cmap='Oranges') 
plt.figure(figsize=(16, 6))

a = sns.countplot(data=train, x='SmokingStatus', hue='Sex')



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')



plt.title('Gender wise SmokingStatus', fontsize=16)

sns.despine(left=True, bottom=True);

train['Sex']           = pd.factorize( train['Sex'] )[0]

train['SmokingStatus'] = pd.factorize( train['SmokingStatus'] )[0]
model1 = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train).fit( q=0.15 )

model2  = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train).fit( q=0.50 )

model3 = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train).fit( q=0.85 )
import numpy as np

train['ypred1'] = model1.predict( train ).values

train['ypred2']  = model2.predict( train ).values

train['ypred3'] = model3.predict( train ).values

train['ypredstd'] = 0.5*np.abs(train['ypred3'] - train['ypred2'])+0.5*np.abs(train['ypred2'] - train['ypred1'])

train.head(2)
dt = train.loc[ train.traintest==1 ,['Patient','Percent','Age','Sex','SmokingStatus']]

test = pd.merge( submission, dt, on='Patient', how='left' )

test.sort_values( ['Patient','Weeks'], inplace=True )

test['ypred1'] = model1.predict( test ).values

test['FVC']    = model2.predict( test ).values

test['ypred3'] = model3.predict( test ).values

test['Confidence'] = np.abs(test['ypred3'] - test['ypred1']) / 2

test[['Patient_Week','FVC','Confidence']].to_csv('submission.csv', index=False)

df=pd.read_csv("submission.csv")
df.head(5)
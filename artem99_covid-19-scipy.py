

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

path_train='/kaggle/input/covid19-global-forecasting-week-1/train.csv'

path_test='/kaggle/input/covid19-global-forecasting-week-1/test.csv'

path_submission='/kaggle/input/covid19-global-forecasting-week-1/submission.csv'
train=pd.read_csv(path_train,sep=',')

test=pd.read_csv(path_test,sep=',')

submission=pd.read_csv(path_submission,sep=',')
train.head()
train['day/month']=[x.replace(x[:5],'') for x in train['Date']]
train.head()
test['day/month']=[x.replace(x[:5],'') for x in test['Date']]
train['day/month'].unique()
test['day/month'].unique()
import matplotlib.pyplot as plt

fatal_global=train.pivot_table('Fatalities',columns=['day/month'],aggfunc=sum)

plt.figure(figsize=(20,10))

plt.bar(fatal_global.columns,fatal_global.values[0])

plt.rc('xtick', labelsize=5)
ConfirmedCases_global=train.pivot_table('ConfirmedCases',columns=['day/month'],aggfunc=sum)

plt.figure(figsize=(20,10))

plt.bar(ConfirmedCases_global.columns,ConfirmedCases_global.values[0])

plt.rc('xtick', labelsize=5)
train['month']=[x[5:7] for x in train['Date']]

train.head()
train['day']=[x[8:] for x in train['Date']]

train.head()
month=train.pivot_table('ConfirmedCases',columns=['month'],aggfunc=sum)

month.plot(kind='bar', figsize=(15, 8), grid=False)

plt.rc('xtick', labelsize=10)
month=train.pivot_table('Fatalities',columns=['month'],aggfunc=sum)

month.plot(kind='bar', figsize=(15, 8), grid=False)

plt.rc('xtick', labelsize=10)
Fatalities=train.pivot_table('Fatalities',columns=['day/month'],aggfunc=sum)

Fatalities=Fatalities[['03-01','03-02','03-03','03-04','03-05','03-06','03-07','03-08','03-09','03-10',

'03-11','03-12','03-13','03-14','03-15','03-16','03-17','03-18','03-19','03-20','03-21','03-22','03-23','03-24']]

Fatalities.plot(kind='bar', figsize=(12, 8), grid=False)

plt.rc('xtick', labelsize=10)
ConfirmedCases=train.pivot_table('ConfirmedCases',columns=['day/month'],aggfunc=sum)

ConfirmedCases=ConfirmedCases[['03-01','03-02','03-03','03-04','03-05','03-06','03-07','03-08','03-09','03-10',

'03-11','03-12','03-13','03-14','03-15','03-16','03-17','03-18','03-19','03-20','03-21','03-22','03-23','03-24']]

ConfirmedCases.plot(kind='bar', figsize=(12, 8), grid=False)

plt.rc('xtick', labelsize=10)
z=0

k=63

ConfirmedCases_list=[]

import scipy as sp

for i in range(284):

    x=np.array(list(range(63)))

    y=train['ConfirmedCases'][z:k]

    e,residuals,rank,sv,rcond=sp.polyfit(x,y,14,full=True)

    fp=sp.poly1d(e)

    x1=np.array(list(range(51,94)))

    ConfirmedCases_list.append(fp(x1))

    z+=63

    k+=63
z=0

k=63

Fatalities_list=[]

for i in range(284):

    x=np.array(list(range(63)))

    y1=train['Fatalities'][z:k]

    e,residuals,rank,sv,rcond=sp.polyfit(x,y1,12,full=True)

    fp=sp.poly1d(e)

    x1=np.array(list(range(51,94)))

    Fatalities_list.append(fp(x1))

    z+=63

    k+=63
Fatalities=[]

for i in range(len(Fatalities_list)):

    for j in range(len(Fatalities_list[i])):

        if Fatalities_list[i][j] <0:

            Fatalities_list[i][j]=0

        Fatalities.append(int(Fatalities_list[i][j]))
ConfirmedCases=[]

for i in range(len(ConfirmedCases_list)):

    for j in range(len(ConfirmedCases_list[i])):

        if ConfirmedCases_list[i][j] <0:

            ConfirmedCases_list[i][j]=0

        ConfirmedCases.append(int(ConfirmedCases_list[i][j]))


submission['Fatalities']=Fatalities

submission['ConfirmedCases']=ConfirmedCases

submission.to_csv ('submission.csv', index =False)
submission
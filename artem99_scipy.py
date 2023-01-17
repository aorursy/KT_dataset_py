# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path_train='/kaggle/input/covid19-global-forecasting-week-4/train.csv'

path_test='/kaggle/input/covid19-global-forecasting-week-4/test.csv'

path_submission='/kaggle/input/covid19-global-forecasting-week-4/submission.csv'
train=pd.read_csv(path_train,sep=',')

test=pd.read_csv(path_test,sep=',')

submission=pd.read_csv(path_submission,sep=',')
train.head()
train.shape
train['day/month']=[x.replace(x[:5],'') for x in train['Date']]
train.head()
test['day/month']=[x.replace(x[:5],'') for x in test['Date']]
train['day/month'].unique()
len(train['day/month'].unique())
test['day/month'].unique()
len(test['day/month'].unique())
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

Fatalities=Fatalities[['04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '04-07',

       '04-08', '04-09','04-10','04-11', '04-12', '04-13', '04-14']]

Fatalities.plot(kind='bar', figsize=(12, 8), grid=False)

plt.rc('xtick', labelsize=10)
ConfirmedCases=train.pivot_table('ConfirmedCases',columns=['day/month'],aggfunc=sum)

ConfirmedCases=ConfirmedCases[['04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '04-07',

       '04-08', '04-09','04-10','04-11', '04-12', '04-13', '04-14']]

ConfirmedCases.plot(kind='bar', figsize=(12, 8), grid=False)

plt.rc('xtick', labelsize=10)
z=0

k=84

ConfirmedCases_list=[]

import scipy as sp

for i in range(313):

    x=np.array(list(range(84)))

    y=train['ConfirmedCases'][z:k]

    e,residuals,rank,sv,rcond=sp.polyfit(x,y,8,full=True)

    fp=sp.poly1d(e)

    x1=np.array(list(range(71,114)))

    ConfirmedCases_list.append(fp(x1))

    z+=84

    k+=84
z=0

k=84

Fatalities_list=[]

for i in range(313):

    x=np.array(list(range(84)))

    y1=train['Fatalities'][z:k]

    e,residuals,rank,sv,rcond=sp.polyfit(x,y1,6,full=True)

    fp=sp.poly1d(e)

    x1=np.array(list(range(71,114)))

    Fatalities_list.append(fp(x1))

    z+=84

    k+=84
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
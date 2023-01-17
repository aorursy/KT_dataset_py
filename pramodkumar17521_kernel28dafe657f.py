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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

test = pd.read_csv("../input/chennai-house-pricing-/test.csv")

train = pd.read_csv("../input/chennai-house-pricing-/train.csv")
train.describe(include='all')
train.isnull().sum()
train.dtypes

train['INT_SQFT'].plot.hist(bins=50)

plt.xlabel('Square ft.',fontsize=12)

plt.ylabel('Frequency',fontsize=12)
train['DIST_MAINROAD'].plot.hist(bins=100)

plt.xlabel('Distance',fontsize=20)
train['COMMIS'].plot.hist()

plt.xlabel('Commision',fontsize=15)
train['SALES_PRICE'].plot.hist(bins=20)

plt.xlabel('price',fontsize=20)
temp=pd.DataFrame(index=train.columns)  

temp['data_types']=train.dtypes    

temp['Null_Counts']=train.isnull().sum()  

temp['Unique_count']=train.nunique() 

temp
train['N_BEDROOM'].value_counts()/len(train)*100
train['N_BATHROOM'].value_counts().plot(kind='bar')
train['AREA'].value_counts().plot.bar() 
train['PARK_FACIL'].value_counts().plot.bar()
train.isnull().sum()
train['N_BEDROOM'].fillna(value=(train['N_BEDROOM'].mode()[0]),inplace=True)
train.pivot_table(index=train['N_BATHROOM'],columns=train['N_BEDROOM'])
train.loc[train['N_BATHROOM'].isnull()==True]

for i in range(0,len(train)):

    if pd.isnull(train['N_BATHROOM'][i])==True:

        if train['N_BEDROOM'][i]==1.0:

            train['N_BATHROOM'][i]=1.0

        else:

            train['N_BATHROOM'][i]=2.0
train['N_BATHROOM'].isnull().sum()
train[['QS_ROOMS','QS_BEDROOM','QS_BATHROOM','QS_OVERALL']].head()
temp=(train['QS_ROOMS']+train['QS_BEDROOM']+train['QS_BATHROOM'])/3

pd.concat([train['QS_ROOMS'],train['QS_BEDROOM'],train['QS_BATHROOM'],temp],axis=1).head(10)
for i in range(0,len(train)):

    if pd.isnull(train['QS_OVERALL'][i])==True:

        temp=(train['QS_ROOMS'][i]+train['QS_BEDROOM'][i]+train['QS_BATHROOM'][i])/3

        train['QS_OVERALL'][i]=temp
train['QS_OVERALL'].isnull().sum()
train.dtypes
train=train.astype({'N_BEDROOM':'object','N_ROOM':'object','N_BATHROOM':'object'})

train[['N_BEDROOM','N_ROOM','N_BATHROOM']].dtypes
train['QS_ROOMS']
temp=list(train.columns)

for i in temp:

    print(' value count in ',i,' ')

    print(train[i].value_counts())

    print(" ")
train['PARK_FACIL'].replace({'Noo':'No'},inplace=True)

train['PARK_FACIL'].value_counts()
train['AREA'].replace({'Chrompt':'Chrompet','Chrmpet':'Chrompet','Chormpet':'Chrompet','Karapakam':'Karapakkam','Ana Nagar':'Anna Nagar','Anna NAgar':'Anna Nagar','KKNagar':'KK Nagar','TNagar':'T Nagar','Velchery':'Velachery','Adyr':'Adyar'},inplace=True)

train['AREA'].value_counts()
train['SALE_COND'].replace({'AsjLand':'AdjLand','Adj Land':'AdjLand','Ab Normal':'AbNormal','Partiall':'Partial','PartiaLl':'Partial'},inplace=True)

train['SALE_COND'].value_counts()
train['PARK_FACIL'].replace({'Noo':'No'},inplace=True)

train['PARK_FACIL'].value_counts()
train['UTILITY_AVAIL'].replace({'All Pub':'AllPub'},inplace=True)

train['UTILITY_AVAIL'].value_counts()
train['STREET'].replace({'NoAccess':'No Access'},inplace=True)

train['STREET'].value_counts()
train['BUILDTYPE'].replace({'Comercial':'Commercial','Other':'Others'},inplace=True)

train['BUILDTYPE'].value_counts()
train.plot.scatter('INT_SQFT','SALES_PRICE')
fig,ax=plt.subplots()

colors={'Commercial':'red','House':'blue','Others':'green'}

ax.scatter(train['INT_SQFT'],train['SALES_PRICE'],c= train['BUILDTYPE'].apply(lambda x:colors[x]))
train.pivot_table(values='SALES_PRICE', index='N_BEDROOM', columns='N_BATHROOM',aggfunc='median')
train.plot.scatter('QS_OVERALL','SALES_PRICE')
fig,axs=plt.subplots(2,2)

fig.set_figheight(10)

fig.set_figwidth(10)

axs[0,0].scatter(train['QS_BEDROOM'],train['SALES_PRICE'])

axs[0,0].set_title('QS_BEDROOM')

axs[0,1].scatter(train['QS_BATHROOM'],train['SALES_PRICE'])

axs[0,1].set_title('QS_BATHROOM')

axs[1,0].scatter(train['QS_ROOMS'],train['SALES_PRICE'])

axs[1,0].set_title('QS_ROOMS')

axs[1,1].scatter(train['QS_OVERALL'],train['SALES_PRICE'])

axs[1,1].set_title('QS_OVERALL')
test.describe(include='all')
train.isnull().sum()
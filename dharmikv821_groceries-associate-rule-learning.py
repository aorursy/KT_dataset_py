# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
df = pd.read_csv('../input/groceries-dataset/Groceries_dataset.csv')

df.head()
df.isna().sum()
df.nunique()
df.shape
df.info()
df.Date = pd.to_datetime(df.Date) 
df['Year'] = df.Date.apply(lambda x : x.year)
df['Month'] = df.Date.apply(lambda x : x.month)

df['Days of Week'] = df.Date.apply(lambda x : x.dayofweek)
df.nunique()
sns.countplot(df.Year)

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(df.Month)

plt.show()

plt.figure(figsize=(12,4))

sns.countplot(df.Month,hue=df.Year)

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(df['Days of Week'])

plt.show()

plt.figure(figsize=(12,4))

sns.countplot(df['Days of Week'],hue=df.Year)

plt.show()
df.groupby('Month').count().plot(legend=False,figsize=(12,4))

plt.xticks([i for i in range(1,13)])

plt.show()
df[df['Year']==2014].groupby('Month').count()['Date'].plot(label=2014,figsize=(12,4))

df[df['Year']==2015].groupby('Month').count()['Date'].plot(label=2015)

plt.xticks([i for i in range(1,13)])

plt.legend()

plt.show()
df[df['Year']==2014].groupby('Days of Week').count()['Date'].plot(label=2014,figsize=(12,4))

df[df['Year']==2015].groupby('Days of Week').count()['Date'].plot(label=2015)

plt.xticks([i for i in range(0,7)])

plt.legend()

plt.show()
days = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

for i in range(0,7):

    df[df['Days of Week']==i].groupby('Month').count().plot(legend=False,figsize=(12,4))

    plt.xticks([i for i in range(1,13)])

    plt.title(days[i].upper())

    plt.show()
df.itemDescription.value_counts().head(50).plot(kind='bar',figsize=(15,6))

plt.show()
pd.DataFrame(df.itemDescription.value_counts()/len(df)).head()
dummies = pd.get_dummies(df.itemDescription)

dummies.head()
df = df.join(dummies)

df.head()
item = df.itemDescription.unique()
df = df.groupby(['Member_number','Date'])[item[:]].sum()

df.head(10)
df = df.reset_index()
df.head(10)
df = df.drop(['Member_number','Date'],axis=1)
df.head(10)
# Converting true values to its columns names



temp = df.copy()



for i in range(len(temp)):

    for j in (temp.columns):

        if temp.loc[i,j]>0:

            temp.loc[i,j]=j

temp.head(10)
temp = temp.values

transactions = []

for i in range(14693):

    x=[]

    for j in range(167):

        if temp[i,j]!=0:

            x.append(temp[i,j])

    transactions.append(x)

transactions
!pip install apyori

from apyori import apriori
results = apriori(transactions,min_support=0.0003,min_confidence=0.02,min_lift=3,min_length=2,target='rules')
results = list(results)
len(results)
results[0]
def inspect(results):

    lhs         = [tuple(result[2][0][0])[0] for result in results]

    rhs         = [tuple(result[2][0][1])[0] for result in results]

    supports    = [result[1] for result in results]

    confidences = [result[2][0][2] for result in results]

    lifts       = [result[2][0][3] for result in results]

    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDF = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
resultsinDF.sort_values(by='Lift',ascending=False)
resultsinDF.drop(['Confidence','Lift'],axis=1).sort_values(by='Support',ascending=False)
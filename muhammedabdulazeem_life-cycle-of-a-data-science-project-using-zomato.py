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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from scipy.stats import norm,skew

from statsmodels.graphics.gofplots import qqplot
data = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')
data.head()
data.shape
data.dtypes
data.isnull().sum()
data.drop(['url','address','phone'],axis=1,inplace=True)
data.describe(include='all')
data[['rate','outof']] = data['rate'].str.split('/',expand=True)
data.head()
data.shape
data.drop('outof',axis=1,inplace=True)
data.dtypes
data['rate'].unique()
data.drop(data[data['rate'] == '-'].index,inplace=True)
data.shape
data[data['rate'] == 'NEW']
data[data['rate'] == 'NEW'].shape
data['rate'] = data['rate'].replace('NEW',np.nan)
data[data['votes'] == 0].shape
data['rate'].astype('float')
sns.distplot(data['rate'],fit=norm)
#from sklearn.impute import KNNImputer
#imputer = KNNImputer(n_neighbors=3)
data['location'].unique()
plt.figure(figsize=(20,20))

sns.countplot(y='location',data=data,palette = "Set1",order=data['location'].value_counts().iloc[:50].index)
plt.figure(figsize=(20,20))

sns.countplot(y='rest_type',data=data,palette = "Set1",order=data['rest_type'].value_counts().iloc[:30].index)
plt.figure(figsize=(20,20))

sns.countplot(y='dish_liked',data=data,palette = "Set1",order=data['dish_liked'].value_counts().iloc[:50].index)
plt.figure(figsize=(20,20))

sns.countplot(y='cuisines',data=data,palette = "Set1",order=data['cuisines'].value_counts().iloc[:60].index)
data.head()
data['approx_cost(for two people)'].dtypes
data['approx_cost(for two people)']
data.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality','listed_in(type)': 'restaurant_type'}, inplace=True)
data['average_cost'].unique()
data['average_cost'] = data['average_cost'].astype(str).apply(lambda x: x.replace(',',''))

data['average_cost'] = data['average_cost'].astype('float')

fig, ax = plt.subplots(figsize=[16,4])

sns.distplot(data['average_cost'],ax=ax)

ax.set_title('average_cost')
data['online_order'] = data['online_order'].map({'Yes':1,'No':0})
online_orders=pd.crosstab(data['rate'],data['online_order'])
online_orders
plt.figure(figsize=(20,15))

online_orders.plot.bar(stacked=True)

plt.legend(title='Online orders V/S rating')
book_table=pd.crosstab(data['rate'],data['book_table'])

plt.figure(figsize=(20,15))

book_table.plot.bar(stacked=True)

plt.legend(title='Book table V/S rating')
data['location']
data['rate'].dtypes
data['rate'] = data['rate'].astype('float')
location = data.groupby(['location'])['rate'].mean().reset_index()

location.head(50).sort_values('rate', ascending=False).style.background_gradient(cmap='Greens')
plt.figure(figsize=(15,20))

sns.barplot(x='location',y='rate',data=location[:50])

plt.xticks(rotation=90)
restrauent_type = data.groupby(['restaurant_type'])['rate'].mean().reset_index()

restrauent_type.head(50).sort_values('rate', ascending=False).style.background_gradient(cmap='Greens')
location_with_price = data.groupby(['location'])['average_cost'].mean().reset_index()

location_with_price.head(50).sort_values('average_cost', ascending=False).style.background_gradient(cmap='Greens')
plt.figure(figsize=(15,20))

sns.barplot(x='location',y='average_cost',data=location_with_price[:50])

plt.xticks(rotation=90)
rest_type_with_price = data.groupby(['rest_type'])['average_cost'].mean().reset_index()

rest_type_with_price.head(50).sort_values('average_cost', ascending=False).style.background_gradient(cmap='Greens')
plt.figure(figsize=(15,20))

sns.barplot(x='rest_type',y='average_cost',data=rest_type_with_price[:50])

plt.xticks(rotation=90)
data.head()
restrauent_type_with_price = data.groupby(['restaurant_type'])['average_cost'].mean().reset_index()

restrauent_type_with_price.head(50).sort_values('average_cost', ascending=False).style.background_gradient(cmap='Greens')
X= data.drop_duplicates(subset='name',keep='first')
X.shape
type(X)
newdf=X[['name','average_cost','locality','rest_type','cuisines','rate','restaurant_type','online_order','book_table','dish_liked','cuisines']].groupby(['average_cost'], sort = True)

#newdf=newdf.sort_values(by=['average_cost'])
X= data.drop_duplicates(subset='name',keep='first')

# dups_name = X1.pivot_table(index=['name'],aggfunc='size')

newdf=X[['name','average_cost','locality','rest_type','cuisines','restaurant_type','online_order','book_table','dish_liked','cuisines']].groupby(['average_cost'], sort = True)

newdf=newdf.filter(lambda x: x['average_cost'].mean() <= 1500)

newdf=newdf.sort_values(by='average_cost')



newdf_expensive=X[['name','average_cost','locality','rest_type','cuisines','restaurant_type','online_order','book_table','dish_liked','cuisines']].groupby(['average_cost'], sort = True)

newdf_expensive=newdf_expensive.filter(lambda x: x['average_cost'].mean() >= 3000)

newdf_expensive=newdf_expensive.sort_values(by='average_cost')
newdf
newdf_rate=X[['name','rate']].groupby(['rate'], sort = True)

newdf_rate=newdf_rate.filter(lambda x: x['rate'].mean() >= 4.5)

newdf_rate=newdf_rate.sort_values(by=['rate'])

X.rate.value_counts()

X.rate.unique()

X.nunique()
s1 = pd.merge(newdf,newdf_rate,how='inner',on=['name'])
s1.head(60)
s2 = pd.merge(newdf_expensive,newdf_rate,how='inner',on=['name'])
s2
newdf_voting = X[['name','average_cost','votes','locality','rest_type','cuisines','restaurant_type','online_order','book_table','dish_liked','cuisines']]
newdf_voting=X[['name','average_cost','votes','locality','rest_type','cuisines','restaurant_type','online_order','book_table','dish_liked','cuisines']].groupby(['votes'], sort = True)

newdf_voting=newdf_voting.filter(lambda x: x['votes'].mean() >= 200)
newdf_voting=newdf_voting.sort_values(by=['votes'])
s3 = pd.merge(pd.merge(newdf_voting,newdf_rate,on='name'),newdf,on='name')
s3.iloc[:,:12]
s4 = pd.merge(pd.merge(newdf_voting,newdf_rate,on='name'),newdf_expensive,on='name')
s4.iloc[:,:12]
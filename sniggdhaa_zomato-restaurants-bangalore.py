import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import ast

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import os

print(os.listdir("../input"))
data=pd.read_csv('../input/zomato.csv')

data.head()
data.isnull().sum()
data.drop(columns=['url','address','phone','listed_in(city)','dish_liked'], axis=1,inplace=True)
data.isnull().sum()
data.rate.unique()
data.rate.replace(('NEW','-'),np.nan,inplace =True)

data.rate = data.rate.astype('str')

data.rate = data.rate.apply(lambda x: x.replace('/5','').strip())

data.rate = data.rate.astype('float')

data.rate.unique()
data.reviews_list = data.reviews_list.apply(lambda x: ast.literal_eval(x))

data.reviews_list[0]
x = [float(i[0].replace('Rated','').strip()) for i in data.reviews_list[0]]

x = round(sum(x)/len(x),1)

x
def get_rate(x):

    if not x or len(x) < 1:

        return None

    rate = [float(i[0].replace('Rated','').strip())  for i in x if type(i[0])== str]

    return round((sum(rate)/len(rate)),1)



data['review_rate']= data.reviews_list.apply(lambda x : get_rate(x))
nan_index = data.query('rate != rate & review_rate == review_rate').index

for i in nan_index:

    data.loc[i,'rate'] = data.loc[i,'review_rate'] 

data.drop(columns='review_rate',axis=1,inplace=True)
data.rename(columns={'approx_cost(for two people)': 'avg_cost', 'listed_in(type)': 'meal_type'}, inplace=True)
data.dropna(subset=['rate','avg_cost'],inplace=True)
data.isna().sum()
data.online_order.replace(('Yes','No'),(True,False),inplace =True)

data.book_table.replace(('Yes','No'),(True,False),inplace =True)
data.avg_cost= data.avg_cost.apply(lambda x: int(x.replace(',','')))
data.avg_cost.unique()
data.name = data.name.apply(lambda x: x.title())

data.name.unique()
((data.isna().sum()/data.shape[0])*100).round(3)
data.shape
data.head()
sns.heatmap(data.corr())
plt.figure(figsize=(14,6))

data.location.value_counts()[:10].plot('bar',grid=True)

plt.title('Locations of restaurants in bangalore',weight='bold')

plt.xlabel('Resturant\'s Location')

plt.ylabel('Number of Restaurants')
plt.rcParams['figure.figsize'] = (14, 6)

data.groupby(['location','online_order']).size().unstack().plot(kind='bar',stacked=True)

plt.title('Online orderings',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Location')

plt.show()
plt.rcParams['figure.figsize'] = (14, 6)

data.groupby(['location','book_table']).size().unstack().plot(kind='bar',stacked=True)

plt.title('Table Bookings',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Location')

plt.show()
plt.figure(figsize=(14,6))

data.online_order.value_counts().plot('bar')

plt.title('Online ordering restaurants in bangalore',weight='bold')

plt.xlabel('Online Delivery Available')

plt.ylabel('Number of Restaurants')
plt.figure(figsize=(14,6))

data.book_table.value_counts().plot('bar')

plt.title('Table Booking restaurants in bangalore',weight='bold')

plt.xlabel('Table Booking Available')

plt.ylabel('Number of Restaurants')
plt.rcParams['figure.figsize'] = (14, 6)

Y = pd.crosstab(data['rate'], data['book_table'])

Y.div(Y.sum(1), axis = 0).plot(kind = 'bar', stacked = True,color=['red','blue'])

plt.title('Table booking vs rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
plt.rcParams['figure.figsize'] = (14, 6)

Y = pd.crosstab(data['rate'], data['online_order'])

Y.div(Y.sum(1), axis = 0).plot(kind = 'bar', stacked = True,color=['red','blue'])

plt.title('Online ordering vs rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
plt.figure(figsize=(14,6))

data.rest_type.value_counts()[:10].plot('bar',grid=True)

plt.title('Types of restaurants in bangalore',weight='bold')

plt.xlabel('Restaurant type')

plt.ylabel('Number of Restaurants')
data['location'][data.rate==5].value_counts()
plt.rcParams['figure.figsize'] = (14, 6)

data['location'][data.rate==5].value_counts().plot('bar')

plt.title('Locations of 5-star rated restaurants',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Location')

plt.show()
data['name'].value_counts()[:10]
plt.rcParams['figure.figsize'] = (14, 6)

data['name'].value_counts()[:10].plot('bar')

plt.title('Top 10 Restaurant chains in Bangalore',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Name')

plt.show()
data['name'][data.rate==5].value_counts()[:10]
plt.rcParams['figure.figsize'] = (14, 6)

data['name'][data.rate==5].value_counts()[:10].plot('bar')

plt.title('5-star rated Restaurant chains in Bangalore',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Name')

plt.show()
plt.rcParams['figure.figsize'] = (14, 6)

data.nlargest(25,'avg_cost')['name'].value_counts().plot(kind='bar')

plt.title('Most expensive restaurant chains in Bangalore',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Name')

plt.show()
plt.rcParams['figure.figsize'] = (14, 6)

data.nlargest(25,'avg_cost')['rate'].value_counts().plot(kind='bar')

plt.title('Rating of most expensive restaurant chains in Bangalore',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Name')

plt.show()
plt.rcParams['figure.figsize'] = (14, 6)

data.nlargest(25,'avg_cost')['location'].value_counts().plot(kind='bar')

plt.title('Location of most expensive restaurant chains in Bangalore',fontweight='bold')

plt.ylabel('Number of Restaurants')

plt.xlabel('Location')

plt.show()
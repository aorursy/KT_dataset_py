import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams
data=pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')

data.head()
print("Percentage null or na values in df")

((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
data.rate = data.rate.replace("NEW", np.nan)

data.dropna(how ='any', inplace = True)



del data['address']

del data['phone']

del data['location']

data.rename(columns={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality','listed_in(type)': 'restaurant_type'}, inplace=True)

data.head()
X = data

X.rate = X.rate.astype(str)

X.rate = X.rate.apply(lambda x: x.replace('/5',''))

X.rate = X.rate.apply(lambda x: float(x))

X.head()
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="locality",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Order Count Per Locality',size = 20)
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="rest_type",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Number of per type of Resturant',size = 20)
plt.rcParams['figure.figsize'] = (3, 4)

plt.style.use('_classic_test')



X['online_order'].value_counts().plot.bar(color = 'cyan')

plt.title('Online orders', fontsize = 20)

plt.ylabel('Number of orders', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

x = pd.crosstab(X['rate'], X['online_order'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('online order rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
plt.rcParams['figure.figsize'] = (7, 9)

plt.style.use('_classic_test')



X['book_table'].value_counts().plot.bar(color = 'cyan')

plt.title('Table booking', fontsize = 20)

plt.ylabel('Number of bookings', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

x = pd.crosstab(X['rate'], X['book_table'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('table booking rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
import seaborn as sns

sns.barplot(data.groupby('online_order').count().head()['url'].index,data.groupby('online_order').count().head()['url'])
X.head()

X.average_cost = X.average_cost.apply(lambda x: x.replace(',',''))

X.average_cost = X.average_cost.astype(int)

fig, ax = plt.subplots(figsize=[16,4])

sns.distplot(X['average_cost'],ax=ax)

ax.set_title('Cost Distrubution for all restaurants')
restaurantTypeCount=data['restaurant_type'].value_counts().sort_values(ascending=True)

slices=[restaurantTypeCount[0],

        restaurantTypeCount[1],

        restaurantTypeCount[2],

        restaurantTypeCount[3],

        restaurantTypeCount[4],

        restaurantTypeCount[5],

        restaurantTypeCount[6]]

labels=['Pubs and bars','Buffet','Drinks & nightlife','Cafes','Desserts','Dine-out','Delivery ']

colors = ['#3333cc','#ffff1a','#ff3333','#c2c2d6','#6699ff','#c4ff4d','#339933']

plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)

fig = plt.gcf()

plt.title("Percentage of Restaurants according to their Type", bbox={'facecolor':'2', 'pad':5})



fig.set_size_inches(12,12)

plt.show()
data=data[data['dish_liked'].notnull()]

data.index=range(data.shape[0])

import re

likes=[]

for i in range(data.shape[0]):

    splited_array=re.split(',',data['dish_liked'][i])

    for item in splited_array:

        likes.append(item)



sns.barplot(pd.DataFrame(likes)[0].value_counts().head(10),pd.DataFrame(likes)[0].value_counts().head(10).index,orient='h')
plt.figure(figsize=(12,5))

sns.barplot(data['rest_type'].value_counts().head(8).index,data['rest_type'].value_counts().head(8))
X= X.drop_duplicates(subset='name',keep='first')

newdf=X[['name','average_cost','locality','rest_type','cuisines']].groupby(['average_cost'], sort = True)

newdf=newdf.filter(lambda x: x.mean() <= 1500)

newdf=newdf.sort_values(by=['average_cost'])



newdf_expensive=X[['name','average_cost','locality','rest_type','cuisines']].groupby(['average_cost'], sort = True)

newdf_expensive=newdf_expensive.filter(lambda x: x.mean() >= 3000)

newdf_expensive=newdf_expensive.sort_values(by=['average_cost'])
newdf_rate=X[['name','rate']].groupby(['rate'], sort = True)

newdf_rate=newdf_rate.filter(lambda x: x.mean() >= 4.5)

newdf_rate=newdf_rate.sort_values(by=['rate'])

X.rate.value_counts()

X.rate.unique()

X.nunique()
s1 = pd.merge(newdf, newdf_rate, how='inner', on=['name'])



s2= pd.merge(newdf_expensive, newdf_rate, how='inner', on=['name'])



print("Cheap restaurants with low cost,high rating \n")

s1
print("Expensive restaurants with high cost,high rating \n")

s2
newdf_votes=X[['name','votes']].groupby(['votes'], sort = True)

newdf_votes=newdf_votes.filter(lambda x: x.mean() >= 175)

newdf_votes=newdf_votes.sort_values(by=['votes'])
s = pd.merge(s1, newdf_votes, how='inner', on=['name'])

s=s.sort_values(by=['average_cost'])

print("Cheap restaurants,high rating,high votes")

s
s = pd.merge(s2, newdf_votes, how='inner', on=['name'])

s=s.sort_values(by=['average_cost'])

s
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
zdata = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
zdata.head()
zdata.info()
zdata.isnull().sum()
#Checking if we have duplicate records

dup_rec = zdata[zdata.duplicated()]
print(dup_rec.shape)

#We dont have any duplicate records
#Function to label in plots

def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,

                '%d' % int(height),

                ha='center', va='bottom')
#resturnts with maximum chain

temp_series = zdata.name.value_counts()[:10]

labels = np.array(temp_series.index)

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots()

rects = ax.bar(ind, np.array(temp_series), width=width, color='y')

ax.set_xticks(ind+((width)/2.))

ax.set_xticklabels(labels, rotation='vertical')

ax.set_ylabel("Count")

ax.set_title("Top 10 resturants")

autolabel(rects)

plt.show()
#percent of resturants accepting online orders

zdata['online_order'].value_counts().plot(kind='pie',autopct='%1.1f%%')
#percent of resturants having table booking

zdata['book_table'].value_counts().plot(kind='pie',autopct='%1.1f%%')
#online Order vs Booking table option

sns.countplot(x='online_order',data=zdata,hue='book_table')
#Distribution of ratings

plt.figure(figsize=(6,5))

rating=zdata['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()

sns.distplot(rating)
temp_df=zdata[['rate','approx_cost(for two people)','online_order','rest_type','name']].dropna()

temp_df['rate']=temp_df['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)

temp_df['approx_cost(for two people)']=temp_df['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
#Online Order option against Cost for two people

sns.scatterplot(x="rate",y='approx_cost(for two people)',hue='online_order',data=temp_df)

plt.show()
#Cont of each Resturant Types

rest=zdata['rest_type'].value_counts()[:20]

sns.barplot(rest,rest.index)

plt.title("Restaurant types")

plt.xlabel("count")
def mywish_rest(cost,rating):

    budget=temp_df[(temp_df['approx_cost(for two people)']<=cost) & (temp_df['rate']>rating)]

    return(budget['name'].unique())
mywish_rest(400,4)
#Popular Cusines of Blore

plt.figure(figsize=(7,7))

cuisines=zdata['cuisines'].value_counts()[:20]

sns.barplot(cuisines,cuisines.index)

plt.xlabel('Count')

plt.title("List of popular Cusines")
#Most Liked Dish

dish=zdata['dish_liked'].value_counts()[:20]

sns.barplot(dish,dish.index)

plt.xlabel('Count')

plt.title("Most Liked dish")
#Top rated dish

newdf_rate=temp_df[['name','rate']].groupby(['rate'], sort = True)

newdf_rate=newdf_rate.filter(lambda x: x['rate'].mean() >= 4.5)

newdf_rate=newdf_rate.sort_values(by=['rate'])

newdf_rate
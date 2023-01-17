import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
zomato= pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")

zomato.head()
zomato.info()
del zomato['url']

del zomato['address']

del zomato['phone']
zomato.head()
# Replacing restaurents with there ratings given as new to NAN and dopping them finally

zomato['rate']=zomato['rate'].replace('NEW',np.NAN)

zomato['rate']=zomato['rate'].replace('-',np.NAN)

zomato.dropna(how='any',inplace=True)
zomato['rate']=zomato.loc[:,'rate'].replace('[ ]','',regex=True)

zomato['rate']=zomato['rate'].astype(str)

zomato['rate']=zomato['rate'].apply(lambda r:r.replace('/5',''))

zomato['rate']=zomato['rate'].apply(lambda r:float(r))
# now converting cost from string to integer 

zomato['approx_cost(for two people)']=zomato['approx_cost(for two people)'].str.replace(',','')

zomato['approx_cost(for two people)']=zomato['approx_cost(for two people)'].astype(int)
zomato.head()
from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets
location=['Banashankari', 'Basavanagudi', 'Jayanagar', 'Kumaraswamy Layout',

       'Rajarajeshwari Nagar', 'Mysore Road', 'Uttarahalli',

       'South Bangalore', 'Vijay Nagar', 'Bannerghatta Road', 'JP Nagar',

       'BTM', 'Wilson Garden', 'Koramangala 5th Block', 'Shanti Nagar',

       'Richmond Road', 'City Market', 'Bellandur', 'Sarjapur Road',

       'Marathahalli', 'HSR', 'Old Airport Road', 'Indiranagar',

       'Koramangala 1st Block', 'East Bangalore', 'MG Road',

       'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor',

       'Residency Road', 'Shivajinagar', 'Infantry Road',

       'St. Marks Road', 'Cunningham Road', 'Race Course Road', 'Domlur',

       'Koramangala 8th Block', 'Frazer Town', 'Ejipura', 'Vasanth Nagar',

       'Jeevan Bhima Nagar', 'Old Madras Road', 'Commercial Street',

       'Koramangala 6th Block', 'Majestic', 'Langford Town',

       'Koramangala 7th Block', 'Brookefield', 'Whitefield',

       'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield',

       'Koramangala 2nd Block', 'Koramangala 3rd Block',

       'Koramangala 4th Block', 'Koramangala', 'Bommanahalli',

       'Hosur Road', 'Seshadripuram', 'Electronic City', 'Banaswadi',

       'North Bangalore', 'RT Nagar', 'Kammanahalli', 'Hennur',

       'HBR Layout', 'Kalyan Nagar', 'Thippasandra', 'CV Raman Nagar',

       'Kaggadasapura', 'Kanakapura Road', 'Nagawara', 'Rammurthy Nagar',

       'Sankey Road', 'Central Bangalore', 'Malleshwaram',

       'Sadashiv Nagar', 'Basaveshwara Nagar', 'Rajajinagar',

       'New BEL Road', 'West Bangalore', 'Yeshwantpur', 'Sanjay Nagar',

       'Sahakara Nagar', 'Jalahalli', 'Yelahanka', 'Magadi Road',

       'KR Puram']

location.sort()

print("Search Restaurants according to their name")

@interact

def show_articles_more_than(Restaurant_Name=''):

    return zomato[zomato['name'].str.contains(Restaurant_Name)]
@interact 

def show_Restaurants_according_to_search(Location=location,

                                         Restaurant_Type=['Buffet', 

                                             'Cafes',

                                             'Delivery',

                                             'Desserts',

                                             'Dine-out',

                                             'Drinks & nightlife',

                                             'Pubs and bars'],

                            Min_Rating=(0,5,0.1),

                            Max_Cost_For_Two_People=(100,5000,50)):

    print("")

    return zomato[ (zomato['rate'] > Min_Rating) 

                &(zomato['listed_in(type)'] == Restaurant_Type) 

                &(zomato['location'] == Location) 

                & (zomato['approx_cost(for two people)'] < Max_Cost_For_Two_People)]

print('number of restaurents with online delivery')

(zomato.online_order == 'Yes').sum()
print('Number of restaurents which does not deliver online')

(zomato.online_order == 'No').sum()
zomato.name.count()
sns.countplot(x=zomato['online_order'])

plt.title('Restuarents delivering online or not')
sns.countplot(x=zomato['online_order'],hue=zomato['listed_in(type)'],)

fig = plt.gcf()  # here gcf means 'GET THE CURRENT FIGURE'

fig.set_size_inches(10,10)

plt.title('Type of restaurents delivering online or not')

print("Number of restaurents with table booking facility")

(zomato.book_table == 'Yes').sum()
print('Number of restaurents without table facility')

(zomato.book_table == 'No').sum()
sns.countplot(x=zomato['book_table'])

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.title('Restaurents providing table booking facility')
sns.countplot(x=zomato['book_table'],hue=zomato['listed_in(type)'])

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.title("Type of restaurents providing table booking facility")

plt.show()
print('Restaurents on there unique ratings')

zomato.rate.unique()
print("Number of restaurents rating between 1.5 and 2")

((zomato.rate>=1.5) & (zomato.rate<2)).sum()
print('number of restaurents rating between 2 and 2.5')

((zomato.rate>=2)&(zomato.rate<2.5)).sum()
print('number of restaurents rating between 2.5 and 3')

((zomato.rate>=2.5) & (zomato.rate<3)).sum()
print('number of restaurents rating between 3 and 3.5')

((zomato.rate>=3)&(zomato.rate<3.5)).sum()
print('number of restaurents rating between 3.5 and 4')

((zomato.rate>=3.5)&(zomato.rate<4)).sum()
print('number of restaurents rating between 4 and 4.5')

((zomato.rate>=4)&(zomato.rate<4.5)).sum()
print('number of restaurents rating between 4.5 and 5')

((zomato.rate>=4.5)&(zomato.rate<=5)).sum()
slices=[((zomato.rate>=1.5) & (zomato.rate<2)).sum(),

        ((zomato.rate>=2) & (zomato.rate<2.5)).sum(),

        ((zomato.rate>=2.5) & (zomato.rate<3)).sum(),

        ((zomato.rate>=3.0) & (zomato.rate<3.5)).sum(),

        ((zomato.rate>=3.5) & (zomato.rate<4)).sum(),

        ((zomato.rate>=4) & (zomato.rate<4.5)).sum(),

        ((zomato.rate>=4.5) & (zomato.rate<5)).sum()

       ]

labels=['1.5-2','2-2.5','2.5-3','3-3.5','3.5-4','4-4.5','4.5-5']

colors = ['Red','blue','Green','black','orange','pink']

plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2)

fig = plt.gcf()

plt.title("Percentage of Restaurants according to their ratings", bbox={'facecolor':'2', 'pad':5})



fig.set_size_inches(10,10)

plt.show()
plt.figure(figsize=(20,10))

Aa=sns.countplot(x='rate',hue='book_table',data=zomato)

plt.title('Rating of restaurents VS book_table')

plt.show()
plt.figure(figsize=(20,10))

Aa=sns.countplot(x='rate',hue='online_order',data=zomato)

plt.title('Rating of restaurents VS book_table')

plt.show()
print("All unique locations of restaurents in bangalore")

zomato.location.unique()
print("count of restaurents at unique locations")

locationCount=zomato['location'].value_counts().sort_values(ascending=True)

locationCount
# Now lets check the location where there is maximum number of restaurents.

print('Maximum number of restaurents is at:')

count_max=max(locationCount)

for x,y in locationCount.items():

    if(y==count_max):

        print(x)
# now, lets find the location where there is minimum number of restaurents.

print('Minimum number of restaurents at :')

count_min=min(locationCount)

for x,y in locationCount.items():

    if(y==count_min):

        print(x)
fig=plt.figure(figsize=(20,40))

locationCount.plot(kind="barh",fontsize=20)

plt.ylabel("Location names",fontsize=50,color="red",fontweight='bold')

plt.title("LOCATION VS RESTAURANT COUNT GRAPH",fontsize=40,color="BLACK")

plt.show()
print('all different dinning type of restaurents')

zomato['listed_in(type)'].unique()
print('Count of all different dinning type restaurents')

restaurantTypeCount=zomato['listed_in(type)'].value_counts().sort_values(ascending=True)

restaurantTypeCount
slices=[restaurantTypeCount[0],

        restaurantTypeCount[1],

        restaurantTypeCount[2],

        restaurantTypeCount[3],

        restaurantTypeCount[4],

        restaurantTypeCount[5],

        restaurantTypeCount[6]]

labels=['Pubs and bars','Buffet','Drinks & nightlife','Cafes','Desserts','Dine-out','Delivery ']

colors = ['Blue','green','pink','yellow','red','brown','orange']

plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2)

fig = plt.gcf()

plt.title("Percentage of Restaurants according to their Type", bbox={'facecolor':'2', 'pad':5})



fig.set_size_inches(12,12)

plt.show()
NorthIndianFoodRestaurants = zomato[zomato['cuisines'].str.contains('North Indian', case=False, regex=True,na=False)]

NorthIndianFoodRestaurants.head()
SouthIndianFoodRestaurants = zomato[zomato['cuisines'].str.contains('South Indian', case=False, regex=True,na=False)]

SouthIndianFoodRestaurants.head()

ChineseFoodRestaurants = zomato[zomato['cuisines'].str.contains('Chinese|Momos', case=False, regex=True,na=False)]

ChineseFoodRestaurants.head()

ItalianFoodRestaurants = zomato[zomato['cuisines'].str.contains('Italian|Pizza', case=False, regex=True,na=False)]

ItalianFoodRestaurants.head()

MexicanFoodRestaurants = zomato[zomato['cuisines'].str.contains('Mexican', case=False, regex=True,na=False)]

MexicanFoodRestaurants.head()

AmericanFoodRestaurants = zomato[zomato['cuisines'].str.contains('american|Burger', case=False, regex=True,na=False)]

AmericanFoodRestaurants.head()

branches = zomato.groupby(['name']).size().to_frame('count').reset_index().sort_values(['count'],ascending=False)

ax = sns.barplot(x='name', y='count', data=branches[:12])

plt.xlabel('')

plt.ylabel('Branches')

plt.title('Food chains and their counts')



fig = plt.gcf()

fig.set_size_inches(25,15)
# Reading required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import operator



%matplotlib inline



import os

print(os.listdir("."))
# Reading the dataset

df = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')
# look at the dataset



df.head()
print('Data has {} rows and {} columns'.format(df.shape[0],df.shape[1]))
# Basic information regarding dataframe



df.info()
df.drop(['url', 'address', 'phone', 'menu_item'], axis=1, inplace=True)
df.head()
def plot_location_graph(data, title):

    '''

    Function to plot barplot between locations and restaurants

    based on provided filtered data

    

    Input : 

     - data : frequency data for locations

     - title : Title for plot 

    

    '''



    loc_count = data

    plt.figure(figsize=(20,10))

    sns.barplot(loc_count.index, loc_count.values, alpha=0.8, color = 'skyblue')

    plt.title(title, fontsize=25)

    plt.ylabel('Number of Restaurants', fontsize=20)

    plt.xlabel('Locations', fontsize=20)

    plt.xticks(

        rotation=45, 

        horizontalalignment='right',

        fontweight='light',

        fontsize='x-large'  

    )

    plt.show()

    
# Filtering top 15 locations with maximum number of restaurants in it



plot_location_graph(df['location'].value_counts()[:15,], 'Top 15 location with most number of Restaurants')
print('There are total {} unique Restaurants in Bangalore'.format(len(df['name'].unique())))
# Filtering locations with most number of unique restaurants



plot_location_graph(df.groupby('location')['name'].nunique().sort_values(ascending=False)[:15,], 'Top 15 location with Restaurants Diveristy/Unique Restaurants')
# Filtering locations based on number of votes given by customers



plot_location_graph(df.groupby('location')['votes'].sum().sort_values(ascending=False)[:15,], 'Top 15 Popular locations for Restaurants')
def clean_data(df):

    

    df = df[df['rate'] != 'NEW']

    df = df[df['rate'] != '-']

    df_rate = df.dropna(subset=['location', 'rate', 'rest_type', 'cuisines', 'approx_cost(for two people)'])

    

    # dropping dish_liked column

    df_rate = df_rate.dropna(axis=1)

    

    binary_encode_dict = { 'Yes' : 0, 'No' : 1}

    df_rate.replace({'online_order' : binary_encode_dict, 'book_table' : binary_encode_dict}, inplace=True)

    

    df_rate['rate'] = df_rate['rate'].apply(lambda x: float(x[:-2].strip()))

    

    df_rate = pd.get_dummies(df_rate, columns=['listed_in(type)'], prefix = 'Listed')

    df_rate = pd.get_dummies(df_rate, columns=['listed_in(city)'], prefix = 'City')

    

    df_rate['approx_cost(for two people)'] = df_rate['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))

    

    for i,row in df_rate.iterrows():

        rest_types = [x.strip() for x in row['rest_type'].split(',')]

        for rest_type in rest_types:

            df_rate.loc[i,rest_type] = int(1)

    

    df_rate.fillna(0, inplace=True)

    df_rate.drop(['name', 'location', 'rest_type', 'cuisines', 'reviews_list'],axis=1, inplace=True)

    

    return df_rate

    

    
df_rate = clean_data(df)
plt.figure(figsize=(20,10))

sns.distplot(df_rate['rate'])

plt.title('Rate Distribution', fontsize=25)

plt.xlabel('Rate', fontsize=20)

plt.xticks(



        fontweight='light',

        fontsize='x-large'  

    )

plt.show()
print('First Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 0.25)))

print('Second Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 0.50)))

print('Third Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 0.75)))

print('Forth Quantile of rate distribution is {} '.format(np.quantile(df_rate['rate'], 1)))

print('Average Rating is {} '.format(df_rate['rate'].mean()))
corr = df_rate.corr()

corr_clean = corr[['rate']]
plt.figure(figsize=(20,10))

sns.distplot(corr_clean)

plt.title('Rate Correlation', fontsize=25)

plt.xlabel('Correlation', fontsize=20)

plt.xticks(



        fontweight='light',

        fontsize='x-large'  

    )

plt.show()
corr_clean[corr_clean['rate']>0.3]
plt.figure(figsize=(20,10))

sns.scatterplot(x='rate',y='votes',data=df_rate)

plt.show()
plt.figure(figsize=(20,10))

sns.scatterplot(x='rate',y='approx_cost(for two people)',data=df_rate)

plt.show()
def dish_liked_counter(df):

    

    dish_liked_dict = {}

    dishes = df['dish_liked'].dropna()



    for dish in dishes:

        dish_list = [x.strip() for x in dish.split(',')]

        for dish_item in dish_list:

            if dish_item in dish_liked_dict.keys():

                dish_liked_dict[dish_item] +=1

            else:

                dish_liked_dict[dish_item] = 1

    return dish_liked_dict

def plot_top_dishes(dish_liked_dict):

    sorted_dish = sorted(dish_liked_dict.items(), key=operator.itemgetter(1), reverse=True)

    x = [x[0] for x in sorted_dish[:20]]

    y = [y[1] for y in sorted_dish[:20]]

    

    plt.figure(figsize=(20,10))

    sns.barplot(x, y, alpha=0.8, color = 'skyblue')

    plt.title('Top 20 most liked dishes', fontsize=25)

    plt.ylabel('Number of Restaurants', fontsize=20)

    plt.xlabel('Locations', fontsize=20)

    plt.xticks(

        rotation=45, 

        horizontalalignment='right',

        fontweight='light',

        fontsize='x-large'  

    )

    plt.show()
dish_liked_dict = dish_liked_counter(df)

plot_top_dishes(dish_liked_dict)
def online_order_pie(df):

    '''

    Function to plot online order pie chart

    

    Input :

     - df : 

    

    '''



    online_order = df['online_order'].value_counts()

    plt.pie(online_order.values, labels=online_order.index, autopct='%1.1f%%', explode=(0, 0.1) ,shadow=True)

    plt.title('Is online order available ?')

    plt.axis('equal')

    plt.show()

    
online_order_pie(df)
def plot_distribution_overlay(df, attribute):

    '''

    Funtion to plot distribution graph of one plot on top of another

    

    Input:

     - df : Dataframe containing restuarants details

     - attribute : attribute with which online ordering needs to be tested

     

     Output:

     - Provide overlay distribution plot

    

    '''

    

    sns.distplot(df_rate[df_rate['online_order']==0][attribute].values, hist = False, kde = True,

                 kde_kws = {'shade': True, 'linewidth': 3}, label = 'Online')

    sns.distplot(df_rate[df_rate['online_order']==1][attribute].values, hist = False, kde = True,

                 kde_kws = {'shade': True, 'linewidth': 3}, label = 'Offline')

    

    plt.title('online_order vs. {} '.format(attribute), fontsize=25)

    plt.xlabel(attribute, fontsize=20)

    plt.show()

    

    

    
plot_distribution_overlay(df_rate, 'rate')
plot_distribution_overlay(df_rate, 'votes')
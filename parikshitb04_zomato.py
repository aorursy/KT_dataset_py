# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

from itertools import chain

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')

df.drop(['url','address','phone','reviews_list'], inplace=True, axis=1)

df.head()

df_individual_rest= df.drop_duplicates('name')

df1= df_individual_rest.groupby('location')['location']

number_rest= df1.count()

# number_rest
no = number_rest

plt.figure(figsize=(20,10))

plt.bar(x=no.index, height=no)

plt.xlabel('Location')

plt.ylabel('Number of restaraunts')

plt.title('Location wise number of restaraunts')

plt.xticks(rotation='vertical')

plt.plot()

    
df_order_only = df[df['listed_in(type)'] == 'Delivery']

df_group = df_order_only.groupby('name')['listed_in(type)']

counts = df_group.count()

top_20= counts.sort_values(ascending=False)[:30] 

names =  list(top_20.index)

values = list(top_20.array)

#     return values

plt.figure(figsize=(20,10))

plt.bar(names,values)

plt.xticks(rotation='vertical')

plt.xlabel('Name of Restaurant')

plt.ylabel('Number of delivery orders via Zomato')

plt.title('Most popular ordered from restaurants on Zomato')

plt.show()

df_dine_only = df[df['listed_in(type)'] != 'Delivery']

df_group = df_dine_only.groupby('name')['listed_in(type)']

counts = df_group.count()

top_20= counts.sort_values(ascending=False)[:30] 

names =  list(top_20.index)

values = list(top_20.array)

#     return values

plt.figure(figsize=(20,10))

plt.bar(names,values)

plt.xticks(rotation='vertical')

plt.xlabel('Name of Restaurant')

plt.ylabel('Number of customers via Zomato')

plt.title('Most popular dine in restaraunts')

plt.show()

df_unique = df.drop_duplicates('name')



df_order = df_unique.dropna(subset=['online_order'])

online_order = df_order['online_order'].value_counts()

print(online_order)



df_book = df_unique.dropna(subset=['book_table'])

book_table = df_book['book_table'].value_counts()



plt.figure(figsize=(20,20))



plt.subplot(1,2,1)

plt.pie(online_order,labels=online_order.index,autopct='%1.1f%%')

plt.title('Availability of ordering online from restaraunts')



plt.subplot(1,2,2)

plt.pie(book_table, labels=book_table.index,autopct='%1.1f%%')

plt.title('Availability of booking tables in restaraunts via Zomato')



# df[(df['online_order'] == 'No') & (df['book_table']=='No')]
df_individual_rest.head()

cuisines = []

df_cuisines = df_individual_rest['cuisines'].astype(str)

df_cuisines

for row in df_cuisines:

    split = row.split(',')

    for c in split:

        cuisines.append(c)



def clean_list(cuisines):

    clean= []

    for value in cuisines:

        if value[0] == ' ':

            value = str(value[1:])

        clean.append(value)

    return clean

        

def count_dic(cuisines):

    clean = clean_list(cuisines)

    dic = {}

    for key in clean:

        if key in dic:

            dic[key] += 1

        else:

            dic[key] = 1

    return dic





cuisine_dict= count_dic(cuisines)





plt.figure(figsize=(30,15))

plt.bar(cuisine_dict.keys(), cuisine_dict.values(), align='center')

plt.xticks(rotation='vertical')

plt.show()







df_top_loc = df[(df['location']== 'BTM' )| (df['location'] == 'HSR')| (df['location'] == 'Whitefield') | (df['location']=='Electronic City') | (df['location'] == 'Marathahalli')]

df_top_loc = df_top_loc.dropna(subset=['rate'])

df_top_loc = df_top_loc[(df_top_loc['rate'] != 'NEW') &(df_top_loc['rate']!= '-') ]

temp_df = df_top_loc['rate'].str.split('/', expand=True)

df_top_loc['ratings'] = temp_df[0]

pd.to_numeric(df_top_loc['ratings'])

df_top_loc['ratings'] = df_top_loc['ratings'].astype(float)





df_top_loc.boxplot(column=['ratings'], by='location', figsize=(20,20))

plt.xlabel('Locations', fontsize=20)

plt.ylabel('Ratings',fontsize=20)

# plt.boxplot(x= df_top_loc['ratings'], labels=df_top_loc['location'])
df_top_dish = df[(df['location']== 'BTM' )| (df['location'] == 'HSR')| (df['location'] == 'Whitefield') | (df['location']=='Electronic City') | (df['location'] == 'Marathahalli')]

df_top_dish= df_top_dish.dropna(subset=['dish_liked'])



def cleaning(label):

    df_top = df_top_dish[df_top_dish['location'] == label]

    df_top = df_top['dish_liked']

    top_dishes = []

    for value in df_top:

        dishes = value.split(',')

        top_dishes.append(dishes)

    dishes_flatten = list(chain.from_iterable(top_dishes))

    clean= []

    for value in dishes_flatten:

        if value[0] == ' ':

            value = str(value[1:])

        clean.append(value)

    fav_dishes = Counter(clean).most_common(20)

    fav_dishes = dict(fav_dishes)

    

    plt.figure(figsize=(20,5))

    plt.bar(fav_dishes.keys(), fav_dishes.values())

    plt.xticks(rotation='vertical')

    plt.xlabel('Dishes')

    plt.title(label)



    plt.show()



cleaning('HSR')

cleaning('Whitefield')

cleaning('BTM')

# df_top_dish = df[(df['location']== 'BTM' )| (df['location'] == 'HSR')| (df['location'] == 'Whitefield') | (df['location']=='Electronic City') | (df['location'] == 'Marathahalli')]

df_new = df[df['rate'] == 'NEW']

df_new_group = df_new.groupby(by='location')['location']

new = df_new_group.count()



plt.figure(figsize=(20,20))

plt.pie(new,labels=new.index, labeldistance=1,center=(0,0),rotatelabels=True)

plt.title('New restaurants in different areas', fontsize=20,loc='center')
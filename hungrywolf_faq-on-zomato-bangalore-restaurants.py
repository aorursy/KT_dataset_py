#FAQ

# 1. Does number of restaurants depends on area?

# 2. Relation between area and type of restaurants

# 3. How much minimum it will cost for two people in a particular area?

# 4. Approx spending in a particular area for two people

# 5. Does online order facility affect number of votes?

# 6. Does online order facility affect average rating of a restaurants?

# 7. Does table booking facility affect average rating of a restaurants?

# 8. Does restaurant rating depend on location?

# 9. Does number of votes depend on restaurant type?

# 10. Does particular type of restaurants get more rating than other type?
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')

df.head()
df.columns
df.isna().sum()
df['address'].iloc[0] # To check address format
df['listed_in(city)'].unique()
df['listed_in(city)'].nunique()
df['location'].unique()
df['location'].nunique()
df.shape
# Renaming of column
df.rename(columns={'approx_cost(for two people)': 'approx_cost_2', 'listed_in(type)': 'type', 'rate':'rating'}, inplace=True)

df.columns
df = df[['name', 'online_order', 'book_table', 'rating', 'votes', 'location', 'type', 'approx_cost_2']]

df.head()
# Data Cleaning
df.duplicated().unique()
df.drop_duplicates(inplace=True, keep='first')
df.shape
# Drop nan from location and approx cost

# Drop restaurants with zero cost 

df = df[(~df['location'].isna())& (~df['approx_cost_2'].isna())]

df.reset_index(inplace=True, drop=True) # drop index column
df.shape
df.isna().sum()
# 1. Total number of restuarant in a particular area

count = df['location'].value_counts()

ax = count.plot(kind='bar', figsize=(25, 12))

ax.set_xlabel(' Area', fontsize=20)

ax.set_ylabel('Restuarant count', fontsize=20)

ax.set_title('Total number of restuarants in a particular area', fontsize=30)

for i, v in enumerate(count):

    plt.text(i, v+20, str(v), rotation=90, verticalalignment='bottom', fontsize=8)

plt.show()
# 2. Relation between area and type of restaurant count

temp = df[['location', 'type']].groupby(['location', 'type']).size().reset_index()

ax = temp.set_index(['location', 'type']).unstack(level=1).plot(kind='bar', stacked=True, figsize=(25, 10))

ax.set_title('Relation between area and type of restaurant count', fontsize=20)

ax.set_xlabel('Area', fontsize=20)

ax.set_ylabel('Restaurant type', fontsize=20)

ax.legend(temp['type'].unique())

plt.show()
# approx cost data is not uniform, in some rows it is for 2 person and for other one person

# To make it uniform 
for i in df.index:

        if df['approx_cost_2'].iloc[i][1] == ',':

            if df['approx_cost_2'].iloc[i][0] =='2':

                df.loc[df.index==i, ['approx_cost_2']] = 2*int(df['approx_cost_2'].iloc[i][2:])

            else:

                df.loc[df.index==i, ['approx_cost_2']] = df['approx_cost_2'].iloc[i][2:]
df['approx_cost_2'] = df['approx_cost_2'].astype(int)
df['votes'] = df['votes'].astype(int)
# Drop restaurants with zero cost 

df.where(df['approx_cost_2']>0, inplace=True)

df = df[~df['approx_cost_2'].isna()]
# 3. Cost for cheapest restaurants in a particular area

temp = df[['name', 'location', 'approx_cost_2']].loc[df.groupby('location')['approx_cost_2'].idxmin()].reset_index()

temp['location'] = temp['location'].astype('category')

plt.scatter(x=range(0, temp.shape[0]), y =temp['approx_cost_2'])

plt.xticks(range(temp.shape[0]), temp['location'], rotation=90)

plt.rcParams['figure.figsize'] = [60,20]

plt.xlabel('Location', fontsize=20)

plt.ylabel('Cost', fontsize=20)

plt.title('Cost for cheapest restaurants in a particular area', fontsize=40)

plt.show()
# 4. cost range per area

temp = df.groupby('location')['approx_cost_2'].agg([('Cmin','min'), ('Cmax','max')])

ax = temp.plot(figsize=(25,8), grid=True)

ax.set_xticks(range(len(temp)))

ax.set_xticklabels(temp.index, rotation=90)

ax.set_xlabel('Location', fontsize=20)

ax.set_ylabel('Cost', fontsize=20)

ax.set_title('Title:- Cost range', fontsize=20)

plt.show()
df = df[df['votes']>0] # calculation only for positive votes
df.isna().sum()
df['rating'].unique()
# replace nan with 0/5

df['rating'].fillna('0/5', inplace=True)
df['rating']=df['rating'].apply(lambda x: x[:-2]) # strip '/5' from rating
df['rating'] = df['rating'].astype(float)
# Number of restaurants with online ordering facility

df.groupby('online_order').size()
# 5. Relation between online order facility and avg vote

temp = df.groupby('online_order')['votes'].mean()

ax = temp.plot(kind='bar', figsize=(10,5))

ax.set_title('Relation between online order facility and vote count', fontsize=20)

ax.set_ylabel('Average vote count', fontsize=15)

plt.show()
# 6. Relation between online order facility and ave rating

temp = df.groupby('online_order')['rating'].mean()

ax = temp.plot(kind='bar', figsize=(10,5))

ax.set_title('Relation between online order facility and Average rating', fontsize=20)

ax.set_ylabel('Avg rating', fontsize=15)

ax.set_xlabel('Is an online order facility available?', fontsize=15)

plt.show()
# 7. Relation between table booking and rating

temp = df.groupby('book_table')['rating'].mean()

ax = temp.plot(kind='bar', figsize=(10,5))

ax.set_title('Relation between table booking facility and Average rating', fontsize=20)

ax.set_ylabel('Avg rating', fontsize=15)

ax.set_xlabel('Is table booking facility available?', fontsize=15)

plt.show()
# 8. Avg restaurant rating per area

temp= df.groupby('location')['rating'].mean().reset_index()

ax = temp.plot()

ax.set_xticks(temp.index)

ax.set_xticklabels(temp['location'], rotation=90)

ax.set_xlabel('Location', fontsize=20)

ax.set_ylabel('Rating', fontsize=20)

ax.set_title('Avg restaurant rating per area', fontsize=20)

plt.show()
# Total count of a particular type of restaurant

df.groupby('type').size()
# 9. Does particular type of restaurants get more votes than other type?

temp = df.groupby('type')['votes'].mean().reset_index()

ax = temp.plot(kind='bar')

ax.set_xticks(temp.index)

ax.set_xticklabels(temp['type'], fontsize=10)

ax.set_ylabel('Avg votes', fontsize=15)

ax.set_xlabel('Type of restaurants', fontsize=15)

ax.set_title('Avg votes for a particular type of restaurant', fontsize=20)

plt.show()
# 10. Does particular type of restaurants get more rating than other type?

temp = df.groupby('type')['rating'].mean().reset_index()

ax = temp.plot(kind='bar')

ax.set_xticks(temp.index)

ax.set_xticklabels(temp['type'], fontsize=10)

ax.set_ylabel('Avg rating', fontsize=15)

ax.set_xlabel('Type of restaurants', fontsize=15)

ax.set_title('Avg rating for a particular type of restaurant', fontsize=20)

plt.show()
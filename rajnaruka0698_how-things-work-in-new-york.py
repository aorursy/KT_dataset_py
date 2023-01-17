import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.shape
df.columns
df.duplicated().sum()
df.info()
df.head()
df.isnull().sum()
df[df['host_name'].isnull()]
df[df['name'].isnull()]
df['host_name'].value_counts().head(10)
most_hotel_owners = df['host_name'].value_counts().head(10)



name_list = list(df['host_name'].value_counts().head(10).index)



temp_dict = {}
for name, group in df.groupby(by = 'host_name') :

    if name in name_list:

        if name not in temp_dict:

            temp_dict[name] = group['neighbourhood_group']

        else :

            temp_dict[name].append(group['neighbourhood_group'])



del name_list
dealing_area = pd.DataFrame(temp_dict).mode().T.join(pd.DataFrame(most_hotel_owners))



dealing_area.columns=['most_frequent_area','total_hotels_owned']



del temp_dict,most_hotel_owners



dealing_area.sort_values(by='total_hotels_owned',inplace=True)



dealing_area
df['neighbourhood_group'].value_counts()
df['neighbourhood_group'].value_counts().plot(kind='bar')
df['room_type'].value_counts()
df['room_type'].value_counts().plot(kind='bar')
df.sort_values(by='price',ascending=False).head(10)
df[df['minimum_nights']>=30].groupby(by='host_name')['id'].count().sort_values(ascending=False).head(10)
df['price_per_night'] = df['price']/df['minimum_nights']

df.drop('price',axis=1,inplace=True)
df.sort_values(by='price_per_night',ascending=False).head(10)
plt.figure(figsize=(15,4))

for name,group in df.groupby(by='neighbourhood_group'):

    sns.distplot(group['price_per_night'])

plt.title('Distribution neighbourhood_group-price')
plt.figure(figsize=(15,4))

for name,group in df.groupby(by='room_type'):

    sns.distplot(group['price_per_night'])

plt.title('Distribution room_type-price')
df.groupby(by='room_type')['price_per_night'].median()
pd.pivot_table(data=df, index='neighbourhood_group', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar')
pd.pivot_table(data=df, index='neighbourhood_group', columns='room_type', values='price_per_night' ,aggfunc='median')
manhattan_data = df[df['neighbourhood_group']=='Manhattan']

brooklyn_data = df[df['neighbourhood_group']=='Brooklyn']

queens_data = df[df['neighbourhood_group']=='Queens']
fig,ax = plt.subplots(1,figsize=(15,4))

pd.pivot_table(data=manhattan_data, index='neighbourhood', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar',ax=ax)
temp_table = pd.pivot_table(data=manhattan_data, index='neighbourhood', columns='room_type', values='price_per_night' ,aggfunc='median').sort_values(by='Private room',ascending = False)

temp_table
color = ['y','g','b']

flag=0

for i in temp_table.columns :

    sns.distplot(temp_table[i].dropna(),color=color[flag])

    flag+=-1

del color,flag
fig,ax = plt.subplots(1,figsize=(15,4))

pd.pivot_table(data=brooklyn_data, index='neighbourhood', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar',ax=ax)
temp_table = pd.pivot_table(data=brooklyn_data, index='neighbourhood', columns='room_type', values='price_per_night' ,aggfunc='median').sort_values(by='Private room',ascending = False)

temp_table
color = ['y','g','b']

flag=0

for i in temp_table.columns :

    sns.distplot(temp_table[i].dropna(),color=color[flag])

    flag+=-1

del color,flag
fig,ax = plt.subplots(1,figsize=(15,4))

pd.pivot_table(data=queens_data, index='neighbourhood', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar',ax=ax)
temp_table = pd.pivot_table(data=queens_data, index='neighbourhood', columns='room_type', values='price_per_night' ,aggfunc='median').sort_values(by='Private room',ascending = False)

temp_table
color = ['y','g','b']

flag=0

for i in temp_table.columns :

    sns.distplot(temp_table[i].dropna(),color=color[flag])

    flag+=-1

del color,flag
sns.scatterplot(data = df, x = 'longitude',y = 'latitude',hue = 'neighbourhood_group')
sns.scatterplot(data = df, x = 'longitude',y = 'latitude',hue = 'room_type')
df1=df.copy()
df1.head()
df1.describe(include='object')
df1.drop(columns=['name','host_name','last_review'],inplace=True)
df1.describe()
df1.drop('reviews_per_month',axis=1,inplace=True)
sns.heatmap(df1.corr(),cmap='Greens')
df1 = pd.get_dummies(df1)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
from sklearn.feature_selection import RFE



selector = RFE(rfr,10)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
df1.shape
x = df1.drop('price_per_night',axis=1)

y = df1['price_per_night']
x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)



rfr.fit(x_train,y_train)

pred = rfr.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,pred)
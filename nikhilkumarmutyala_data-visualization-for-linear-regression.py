#import

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#read the file 'NYC_2019.csv' from the file

df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#obtain information about the dataframe

df.info()
df
#find no of columns and no of rows

df.shape
#obtaining the description of the dataframe

df.describe()
#finding out if there are any null or empty values

df.isnull().sum()
#knowing how many neighbourhood groups are there and count of them

df.neighbourhood_group.value_counts()
fig,ax=plt.subplots(figsize=(10,8))

sub_df = df[df.price < 1000]

plot_2=sns.violinplot(data=sub_df, x='neighbourhood_group', y='price')

plot_2.set_title('Density and distribution of prices for each neighberhood_group')
sub_df = df[df.price < 1000]

plt.figure(figsize = (12, 6))

sns.boxplot(x = 'room_type', y = 'price',  data = sub_df)

#regression
top10_freq_neighbourhood=df.neighbourhood.value_counts().head(10)

print(top10_freq_neighbourhood)
top10_freq_neighbourhood_data=df[df['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',

'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]

top10_freq_neighbourhood_data
t=sns.catplot(x="neighbourhood", y="price", col="room_type", data=top10_freq_neighbourhood_data)

t.set_xticklabels(rotation=45)
df.fillna('0',inplace=True)

df
fig,ax=plt.subplots(figsize=(10,8))

sns.distplot(np.log1p(df['number_of_reviews']))
fig,ax=plt.subplots(figsize=(10,8))

sns.countplot(df['neighbourhood_group'])
df['Cat'] = df['price'].apply(lambda x: 'costly' if x > 3000

                                                    else ('medium' if x >= 1000 and x < 3000

                                                    else ('reasonable' if x >= 500 and x < 1000

                                                     else ('cheap' if x >= 100 and x <500

                                                          else'very cheap'))))
plt.figure(figsize=(10,8))



sns.scatterplot(df.latitude,df.longitude, hue='Cat', data=df)
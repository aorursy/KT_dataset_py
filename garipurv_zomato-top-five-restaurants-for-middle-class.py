import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns
df = pd.read_csv("../input/zomato.csv")
print(df.columns)

print(df.shape)
df.head()
df.isna().sum()
#check null values and how much they are

null_data = pd.DataFrame({'Column':[i.capitalize() for i in df.columns], 'Count':df.isnull().sum().values, 'Percentage':((df.isnull().sum().values*100)/len(df)).round(2)})
null_data
df.rate = df.rate.str.replace('/5', '')

df = df.rename(columns = {'approx_cost(for two people)':'approx_cost'})
print("There are {} resturants across {} locations in Bangalore".format(len(df.name.unique()), len(df.location.unique())))
location_count  = df['location'].value_counts()
plt.figure(figsize=(18, 12))

sns.barplot(location_count.index[:25], location_count[:25].values)

plt.xticks(rotation=90, fontsize=12)

plt.title("Resturanat in Particular Area", fontsize=15)

plt.ylabel("Count of Resturant", fontsize=15)

plt.xlabel("Area", fontsize=15)

plt.show()
df = df.rename(columns={'listed_in(type)':'Resturant_Type', 'listed_in(city)':'Locality'})
df.columns
count_rest = df.Resturant_Type.value_counts()

#print count_rest.values

#print count_rest.index

plt.figure(figsize=(18, 12))

plt.pie(count_rest, labels=count_rest.index, startangle=90, autopct='%.1f%%')

plt.title("Types of Resturant in City", fontsize=15)

plt.show()
df.approx_cost = pd.to_numeric(df.approx_cost, errors='coerce')
df.dtypes
data_budget = df[(df.approx_cost>=300)&(df.approx_cost<=1000)&(df.rate>='4.0')&(df.votes>=100)]

ag = data_budget.groupby(['approx_cost', 'name', 'location', 'cuisines'], as_index=True).agg({'votes':sum})

g = ag['votes'].groupby(level=0, group_keys=False)

g.nlargest(5)
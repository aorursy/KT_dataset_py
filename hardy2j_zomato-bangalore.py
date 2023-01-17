import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
csv='/Users/y2z/Desktop/zomato.csv'
df=pd.read_csv(csv)
df.head()
df.columns

df.describe()
df.info()
df=df.dropna()
df.shape
df.head()
df_area=df[['name','location','rest_type','rate','votes','cuisines','approx_cost(for two people)']]
df_area.head()
df_area_lr=df_area.groupby('location')['rest_type'].value_counts()
plt.figure(figsize=(8,8))

df_b=df_area[df_area['location']=='Banashankari']['rest_type'].value_counts(ascending=True).plot('barh')

plt.xlabel('No.of Resturants')

plt.ylabel('Type of Resturant')

plt.title('Resurants in Banashakari')

plt.show()
plt.figure(figsize=(8,8))

df_jp=df_area[df_area['location']=='JP Nagar']['rest_type'].value_counts(ascending=True).plot('barh')

plt.xlabel('No.of Resturants')

plt.ylabel('Type of Resturant')

plt.title('Resurants in JP Nagar')

plt.show()
df_rate_b=df[df['location']=='Banashankari'][['name','rate']][:10].sort_values(by='rate',ascending=True)

plt.barh(y=df_rate_b.name,width=df_rate_b.rate)

plt.xticks(rotation=90)

plt.xlabel("Name of the Resturant")

plt.ylabel("Ratings")

plt.title("Ratings of Resturants in Banashankari")

plt.show()
df_rate_b
df_rate_jp=df[df['location']=='JP Nagar'][['name','rate']][:10].sort_values(by='rate',ascending=True)

plt.bar(x=df_rate_jp.name,height=df_rate_jp.rate)

plt.xticks(rotation=90)

plt.xlabel("Name of the Resturant")

plt.ylabel("Ratings")

plt.title("Ratings of Resturants in JP Nagar")

plt.show()
df_app2_b=df[df['location']=='Banashankari'][['name','approx_cost(for two people)','rate']][:10].sort_values(by='rate',ascending=True)

plt.barh(y='name',width='approx_cost(for two people)',data=df_app2_b)

plt.xticks(rotation=45)

plt.xlabel('Appox rate for 2 people')

plt.ylabel('Name of the resturant')

plt.title("Appox rate for 2 people in Banashankari")

plt.show()
df_app2_jp=df[df['location']=='JP Nagar'][['name','approx_cost(for two people)','rate']][:10].sort_values(by='rate',ascending=True)

plt.barh(y='name',width='approx_cost(for two people)',data=df_app2_jp)

plt.xticks(rotation=45)

plt.xlabel('Appox rate for 2 people')

plt.ylabel('Name of the resturant')

plt.title("Appox rate for 2 people in JP Nagar")

plt.show()
df_online_jp=df[df['location']=='JP Nagar']['online_order'].value_counts()

size=[709,307]

labels=['YES','NO']

plt.pie(size,labels=labels,explode=[0.1,0.1],colors=['g','b'],autopct='%1.1f%%')

plt.title("Online Orders(JP NAGAR)")

plt.show()
df_online_b=df[df['location']=='Banashankari']['online_order'].value_counts()

size=[285,102]

labels=['YES','NO']

plt.pie(size,labels=labels,explode=[0.1,0.1],colors=['g','b'],autopct='%1.1f%%')

plt.title("Online Orders(BANASHANKARI)")

plt.show()
df_rt_cd=df[df['rest_type']=='Casual Dining']['location'].value_counts().sort_values(ascending=False)[:10].reset_index()

# df_rt_cd.rename(index={'index':'location','location':'count'},inplace=True)

sns.barplot('index','location',data=df_rt_cd)

plt.xticks(rotation=90)

plt.xlabel("Location")

plt.ylabel("No.of Casual Dinings")

plt.title("No.of Casual Dining in different location")

plt.show()
df_cuisines_b=df[df['location']=='Banashankari']['cuisines'].value_counts()[:10].reset_index()

plt.bar(height='cuisines',x='index',data=df_cuisines_b)

plt.xlabel("Type of Cuisines")

plt.ylabel("No.of Resturants")

plt.title("Cuisines in Banashankari")

plt.xticks(rotation=90)

plt.show()
df_cuisines_b
df_cuisines_jp=df[df['location']=='JP Nagar']['cuisines'].value_counts()[:10].reset_index()

plt.bar(height='cuisines',x='index',data=df_cuisines_jp)

plt.xlabel("Type of Cuisines")

plt.ylabel("No.of Resturants")

plt.title("Cuisines in JP Nagar")

plt.xticks(rotation=90)

plt.show()

df.columns
df_il=df.groupby(['location','name','rest_type','cuisines','approx_cost(for two people)'])[['dish_liked']].count().reset_index().sort_values(by='dish_liked',ascending=False)[:10]
df_il
sns.barplot(y='dish_liked',x='approx_cost(for two people)',data=df_il)

plt.title("Dishes Liked VS Approx Cost for two")

plt.show()
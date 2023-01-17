#importing the required libraries

import pandas as pd

import numpy as np

import datetime



#viz Libraries

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight') # setting style for the plots



#warnings

import warnings

warnings.filterwarnings("ignore")



#word cloud

from wordcloud import WordCloud, ImageColorGenerator
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv',index_col='id')
df.shape # shape of data
df.head() # first 5 rows of data - 5 by default
df.info()
df.describe()
df.isna().sum()
#dropping the rows with price = 0 or availability_365 = 0

indexNames = df[ (df['price'] == 0) | (df['availability_365'] == 0) ].index

df.drop(indexNames , inplace=True)
#dropping the duplicates

df = df.drop_duplicates()
#dropping the columns which may not add extra value to the analysis

df = df.drop(['host_id','latitude','longitude'],axis=1)
df.dtypes
#converting "last_review" to "date_time" data type

df['last_review'] = pd.to_datetime(df['last_review'])



#converting categorical variables into "categorical" data type

cat_var = ['neighbourhood_group','neighbourhood','room_type']

df[cat_var] = df[cat_var].astype('category')



df.info()
#popular neighborhood groups

ax = sns.countplot(x="neighbourhood_group", data=df)

#df['neighbourhood_group'].value_counts().plot(kind="bar")

plt.title('Popular neighborhood groups')

plt.xlabel('Neighborhood Group')

plt.ylabel('Count')

plt.show()
ax = sns.countplot(y="neighbourhood", hue="neighbourhood_group", data=df,

              order=df['neighbourhood'].value_counts().iloc[:5].index)

plt.title('Popular Neighborhoods')

plt.ylabel('Neighborhood')

plt.xlabel('Count')

plt.show()
ax = sns.countplot(x="room_type", data=df)

plt.title('Room Type distribution')

plt.xlabel('Room Type')

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(10,10))

ax = sns.countplot(x="room_type", data=df,hue="neighbourhood_group")
df['price'].describe()
sns.boxplot(x='neighbourhood_group',y='price',data = df)

plt.title("Price distribution among the neighborhood groups")

plt.show()
df3 = df[df['price'] <= 200]

df3.price.plot(kind='hist')

plt.xlabel("Price")

plt.title("Price distribution for chosen listings(Price <= 200)")

plt.show()
sns.boxplot(x='neighbourhood_group',y='price',data = df3)

plt.show()
sns.boxplot(x='room_type',y='price',data = df3)

plt.show()
df['number_of_reviews'].plot(kind='hist')

plt.xlabel("Price")

plt.show()
plt.figure(figsize=(9, 6))

plt.subplot(1,2,1)

df3.groupby(['room_type']).count()['number_of_reviews'].plot(kind='bar',alpha = 0.6,color = 'orange')

plt.title('Room Type Vs Number of Reviews',fontsize=15)



plt.subplot(1,2,2)

df3.groupby(['neighbourhood_group']).count()['number_of_reviews'].plot(kind='bar',color='green',alpha=0.5)

plt.title('Neighborhood Vs Number of Reviews',fontsize=15)

plt.tight_layout()

plt.show()
plt.figure(figsize=(10, 6))

plt.subplot(1,2,1)

df3.groupby(['room_type']).count()['minimum_nights'].plot(kind='bar',alpha = 0.6,color = 'orange')

plt.title('Room Type Vs Minimum Nights',fontsize=15)



plt.subplot(1,2,2)

df3.groupby(['neighbourhood_group']).count()['minimum_nights'].plot(kind='bar',alpha = 0.6,color = 'green')

plt.title('Neighborhood Group Vs Minimum Nights',fontsize=15)

plt.tight_layout()

plt.show()
text = " ".join(str(each) for each in df.name)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=100, background_color="white").generate(text)

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
columns =['price','minimum_nights','number_of_reviews','reviews_per_month','availability_365']

#sns.heatmap(df[columns])

corr = df[columns].corr()

corr.style.background_gradient(cmap='coolwarm')
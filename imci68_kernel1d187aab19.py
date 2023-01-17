# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



plt.style.use('ggplot')

warnings.filterwarnings(action = "ignore")

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the Csv File

df=pd.read_csv('../input/zomato.csv')

df.head()
df.info()
#Check for null values

100*(df.isnull() | df.isna()).sum() / len(df)
# Drop unnecessary columns

df.drop(['url','phone','location','dish_liked'],axis=1,inplace=True)
# Unique values in rate column

df.rate.unique()
#Remove null values from Rate Column

df.rate = df.rate.replace(('NEW','-'),np.nan)

df.rate = df.rate.astype('str')

df.rate = df.rate.apply(lambda x:x.replace('/5','').strip())

df.rate = df.rate.astype('float')

df.dropna(subset = ['rate'], inplace = True)
df['price']=df['approx_cost(for two people)']

df.drop(['approx_cost(for two people)'], axis=1)



df.price.unique()

df.head()

df.drop(['approx_cost(for two people)'], axis=1, inplace = True)

# Conversion of price column

df.price = df.price.astype('str')

df.price = df.price.apply(lambda x:x.replace(',','').strip())

df.price = df.price.astype('float')

df.dropna(subset = ['price'], inplace = True)



df.shape
plt.rc('font', size=15)

df.rate.hist(bins=[0,3,3.5,4,4.5,5])



plt.title('Rating Distribution\n')

plt.xlabel('Rating')

plt.ylabel('Count');


sns.set_context('poster')

plt.figure(figsize=(20,15))

hotel = df['name'].value_counts()[:10]

rating = df.rate[:10]

sns.barplot(x = hotel, y = hotel.index, palette='deep')

plt.title("Most Rated Hotel")

plt.xlabel("# of times Rated")

plt.ylabel("Hotel")

plt.show()
sns.set_context('notebook')

plt.figure(figsize=(30,20))

plt.xticks(fontsize = 20)

ax = df.groupby('rest_type')['name'].count().plot.bar()

ax.set_title("Restaurant type count")

ax.set_xlabel("Restaurant Name")

ax.set_ylabel("# of restaurant")

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))

most_text = df.sort_values('votes', ascending = False).head(100).set_index('name')

plt.figure(figsize=(25,15))

sns.set_context('poster')

ax = sns.barplot(most_text['votes'], most_text.index, palette='magma')

for i in ax.patches:

    ax.text(i.get_width()+2, i.get_y()+0.8,str(round(i.get_width())), fontsize=16,color='black')

plt.show()

sns.set_context('talk')

most_books = df.groupby('listed_in(city)')['name'].count().reset_index().sort_values('name', ascending=False).head(10).set_index('listed_in(city)')

plt.figure(figsize=(15,10))

ax = sns.barplot(most_books['name'],most_books.index,palette = 'deep')

ax.set_title("Most Restaurant in City")

ax.set_xlabel("# of Restaurant")

for i in ax.patches:

    ax.text(i.get_width()+.3, i.get_y()+0.5, str(round(i.get_width())), fontsize = 10, color = 'k')



    
cheap_rest = df[(df['rate']>=3) & ((40 <= df['price']) & (df['price'] < 200))].sort_values('price', ascending = True).head(30).set_index('name')



plt.figure(figsize = (15,10))

ax= sns.barplot(x = cheap_rest['price'], y = cheap_rest.index, palette = 'rocket')

ax.set_title("Cheapest Restaurant")

ax.set_xlabel("Price for 2 people")

ax.set_ylabel("Restaurant name")



for i in ax.patches:

    ax.text(i.get_width()+.3, i.get_y()+0.5, str(round(i.get_width())), fontsize = 10, color = 'k')



## Loading the required libraries:



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import seaborn as sns





# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

kaggle=1



if kaggle==1:

    data=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

else:

    data=pd.read_csv("../data/AB_NYC_2019.csv")
## Examining the first 5 rows to understand the type of data:

data.head()
data.shape
data.columns
## Check if any of the columns has null value:

data.isna().sum()
## Number of unique listings:

print(f'There are {data.id.nunique()} unique listings in the neighbourhood')
## Check the location of the listings where maximum listings:

data.neighbourhood_group.value_counts()
# Check the area where the number of listings are higher:

neighbourhood_group=data.neighbourhood_group.unique()

for n in neighbourhood_group:

    print(f'Top 5 neighbourhood in neighbourhood group {n}')

    print(data.loc[data['neighbourhood_group']==n]['neighbourhood'].value_counts()[:5])

    print()
## Room Type % listing:

(data['room_type'].value_counts()/data.shape[0])*100
## Check the distribution of price:

plt.figure(figsize=(8,8))

sns.distplot(data['price'],bins=50,kde=True)

plt.title("Distribution of price(in USD)")
data['price'].describe()
plt.figure(figsize=(8,8))

sns.distplot(data[(data['price']>0) & (data['price']<1000)]['price'],bins=50,kde=True)

plt.title("Distribution of price(in USD)")
data.loc[data.price==10000]
## Check the average property value for each neighbourhood group:

plt.figure(figsize=(8,8))

sns.boxplot(x=data['neighbourhood_group'],y=data['price'],palette=sns.color_palette('Set2'))

plt.title("Boxplot of price for each neighbourhood group",fontsize=15)

plt.xlabel("Neighbourhood group",fontsize=12)

plt.ylabel("Price",fontsize=12)

plt.show()
data.groupby('neighbourhood_group')['price'].agg(['median','mean']).sort_values('median',ascending=False)
## Minimum nights and price:

plt.figure(figsize=(8,8))

sns.scatterplot(x='price',y='minimum_nights',data=data[(data.price>0) & (data.minimum_nights>0)])

plt.title("Price Vs Number of Nights",fontsize=15)

plt.xlabel("Price",fontsize=12)

plt.ylabel("Number of minimum nights",fontsize=12)

plt.show()
data.groupby('neighbourhood_group').agg({'price':'median','minimum_nights':'median'}).sort_values("price",ascending=False)
## Room Type and Price:

data.groupby('room_type')['price'].median()
### Price and reviews:

plt.figure(figsize=(8,8))

sns.scatterplot(x='price',y='number_of_reviews',data=data[data.price<1000])

plt.title("Relation between price and number of reviews(For properties less than 1000 USD)",fontsize=15)

plt.xlabel("Price",fontsize=12)

plt.ylabel("Number of review",fontsize=12)

plt.show()
print(f'There are {data.host_id.nunique()} unique hosts in the dataset')
## Host listing count:

plt.figure(figsize=(8,8))

sns.distplot(data.calculated_host_listings_count,bins=20,kde=False)

plt.title("Distribution of Number of properties listed by host",fontsize=15)

plt.xlabel("Number of properties by host",fontsize=12)

plt.show()
(data[data.calculated_host_listings_count<50]['host_id'].nunique()/data.shape[0])*100
data.calculated_host_listings_count.describe()
plt.figure(figsize=(8,8))

sns.boxplot(x='room_type',y='calculated_host_listings_count',data=data)

plt.title("Boxplot of properties listed by each host with room type",fontsize=15)

plt.xlabel("Room Type",fontsize=10)

plt.ylabel("Number of properties listed by each host",fontsize=10)

plt.show()
variety=data[data.calculated_host_listings_count>1].groupby('host_id')['room_type'].nunique().reset_index().sort_values('room_type',ascending=False)

print(f'Number of hosts with all three room types listed in Airbnb {len(variety[variety.room_type==3].host_id)}')
variety_data=data[data.host_id.isin(variety[variety.room_type==3].host_id)]

variety_data.groupby('host_id')['neighbourhood_group'].nunique().sort_values(ascending=False)
set(variety_data[variety_data.host_id==213781715]['neighbourhood_group'])
max_host=data[data.calculated_host_listings_count==327]
print(f'Name of host:{list(max_host.host_name.unique())}')

print(f'Neighborhood groups listed:{list(max_host.neighbourhood_group.unique())}')

print(f'Neighbourhoods listed:{list(max_host.neighbourhood.unique())}')

print(f'Room type listed:{list(max_host.room_type.unique())}')

print(f'Maximum price listed:{max(max_host.price)} USD Located in neighbourhood {max_host[max_host.price==max(max_host.price)].neighbourhood.unique()}')

print(f'Minimum price listed:{min(max_host.price)} USD Located in neighbourhood {max_host[max_host.price==min(max_host.price)].neighbourhood.unique()}')
## Top 5 Host with maximum median price and median nights for those holding more than 1 property:

data[data.calculated_host_listings_count>1].groupby('host_id').agg({'price':'median','minimum_nights':'median'}).sort_values('price',ascending=False)[:5]
## Create date columns from last review date:

data['last_review']=pd.to_datetime(data['last_review'])

data['year']=data['last_review'].dt.year

data['month']=data['last_review'].dt.month

data['day']=data['last_review'].dt.day

data['day_name']=data['last_review'].dt.day_name()
data.head()
multi_host=data[data.calculated_host_listings_count>1]
multi_host['review_min']=multi_host.groupby('host_id')['number_of_reviews'].min()

multi_host['review_max']=multi_host.groupby('host_id')['number_of_reviews'].max()

multi_host['review_median']=multi_host.groupby('host_id')['number_of_reviews'].median()

multi_host['review_diff']=multi_host.groupby('host_id')['number_of_reviews'].max()-multi_host.groupby('host_id')['number_of_reviews'].min()
multi_host.describe()
def _gen_histogram(df,column):

    plt.figure(figsize=(8,8))

    sns.distplot(df[column].dropna(),bins=10)

    plt.xlabel(r"{}".format(column))

    plt.ylabel("Density")

    plt.title(r"Distribution of {}".format(column))
columns=['review_min','review_max','review_median','review_diff']



for c in columns:

    _gen_histogram(multi_host,c)
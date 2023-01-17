# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/womens-shoes-prices/Datafiniti_Womens_Shoes_Jun19.csv")
df.info()
#selecting some relevant features

features=['brand','categories','colors','prices.amountMax']

df_shoes=df[features]
df_shoes.info()

df_shoes.describe()
df_shoes.head()
#brand list

brand=df_shoes.brand.unique()

print(brand)
#there are 505 unique brand names in the dataset

brand.shape
#calculating the mean price of t=all the brand 

avg_price=[]

for i in brand:

    a=df_shoes['brand']==i

    rows=df_shoes[a]

    avg_price.append(rows['prices.amountMax'].mean())

    

    
x=np.arange(505)

avg_price=np.array(avg_price)

import matplotlib.pyplot as plt

print(x.shape)

print(avg_price.shape)
#plotting avg_price

plt.scatter(x,avg_price)
#creating a dataframe

avg_price_df=pd.DataFrame({'brand':brand,'avg_price':avg_price})
#sorting values

avg_price_df.sort_values('avg_price',ascending=False,inplace=True)
avg_price_df.head(10)
#for max_price for a certain brand

max_price=[]

for i in brand:

    a=df_shoes['brand']==i

    rows=df_shoes[a]

    max_price.append(rows['prices.amountMax'].max())
#creating dataframe

max_price_df=pd.DataFrame({'brand':brand,'max_price':max_price})
max_price_df.sort_values('max_price',ascending=False,inplace=True)
max_price_df.head(10)
a=df_shoes['brand']=='TOTES'

df_shoes[a]
no_of_items=[]

for i in brand:

    no_of_items.append((df_shoes['brand']==i).sum())

    
df_items=pd.DataFrame({'Brand':brand,'total_items':no_of_items})

df_items.sort_values('total_items',ascending=False,inplace=True)

df_items.head(10)
import seaborn as sns

#plotting distplot for Brinley Co.

a=df_shoes['brand']=='Brinley Co.'

price=df_shoes['prices.amountMax'][a]



sns.distplot(price,hist=False)

plt.title(" price distplot for Brinley Co. ")

plt.xlabel("price")

plt.ylabel("price frequency")
#plotting distplot for Propet

a=df_shoes['brand']=='Propet'

price=df_shoes['prices.amountMax'][a]

sns.distplot(price,hist=False)

plt.title(" price distplot for Propet")

plt.xlabel("price")

plt.ylabel("price frequency")
#plotting distplot for SAS

a=df_shoes['brand']=='SAS'

price=df_shoes['prices.amountMax'][a]

sns.distplot(price,hist=False)

plt.title(" price distplot for SAS")

plt.xlabel("price")

plt.ylabel("price frequency")


#plotting distplot for Soda

a=df_shoes['brand']=='Soda'

price=df_shoes['prices.amountMax'][a]

sns.distplot(price,hist=False)

plt.title(" price distplot for Soda")

plt.xlabel("price")

plt.ylabel("price frequency")
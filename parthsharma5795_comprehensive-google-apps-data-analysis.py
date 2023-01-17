# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.



import os

from matplotlib import rcParams



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/googleplaystore.csv')

data2 = pd.read_csv('../input/googleplaystore_user_reviews.csv')



data.head()
print("Shape of the dataframe is",data.shape)

print("The number of nulls in each column are \n", data.isna().sum())
print("Percentage null or na values in df")

((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
data.head()

# data.Installs.unique()

# data.Size.unique()
df=data

df.Installs = df.Installs.replace("Free", np.nan)

df.dropna(how ='any', inplace = True)

df.Installs = df.Installs.astype(str)

df.Installs = df.Installs.apply(lambda x: x.replace(',',''))

df.Installs = df.Installs.apply(lambda x: x.replace('+',''))

df.Installs = df.Installs.apply(lambda x: int(x))

df.head()
df['Size'].replace('Varies with device', np.nan, inplace = True ) 

df.Size = (df.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \

df.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)

            .fillna(1)

            .replace(['k','M'], [10**3, 10**6]).astype(int))

df.Size = df.Size.apply(lambda x: x/(10**6))

df.rename(columns={'Size': 'Size(in MB)'}, inplace=True)

# df.Size.unique()

# df.head()
df.Category = df.Category.apply(lambda x: x.replace('_',' '))

df.Price = df.Price.apply(lambda x: x.replace('$',''))

df.rename(columns={'Price': 'Price(in $)'}, inplace=True)
rcParams['figure.figsize'] = 15,7

category_plot = sns.countplot(x="Category",data=df, palette = "spring")

category_plot.set_xticklabels(category_plot.get_xticklabels(), rotation=90, ha="right")

category_plot 

plt.title('TOTAL apps in each category',size = 20)
# X['Content Rating'].value_counts()

rcParams['figure.figsize'] = 15,7

content_plot = sns.countplot(x="Content Rating",data=df, palette = "pink")

content_plot.set_xticklabels(content_plot.get_xticklabels(), rotation=90, ha="right")

content_plot 

plt.title('Content Rating distribution',size = 20)
# # X['Type'].value_counts()

rcParams['figure.figsize'] = 10,7

type_plot = sns.countplot(x="Type", data=df, palette = "twilight")

type_plot.set_xticklabels(type_plot.get_xticklabels(), rotation=90, ha="right")

type_plot 

plt.title('Number of Free Apps vs Paid Apps',size = 20)

rcParams['figure.figsize'] = 10,7

type_size = sns.boxplot(x="Type",y="Size(in MB)", data=df, palette = "rainbow")

type_size.set_xticklabels(type_size.get_xticklabels(), rotation=90, ha="right")

type_size 

plt.title('Size Range for free and paid Apps',size = 20)
rcParams['figure.figsize'] = 15,10

content_price = sns.stripplot(y="Content Rating",x="Price(in $)", data=df, palette = "Set1")

content_price.set_xticklabels(content_price.get_xticklabels(), rotation=90, ha="right")

content_price 

plt.title('Content Rating vs Price',size = 20)
df['Price(in $)'] = df['Price(in $)'].apply(lambda x: float(x))

rcParams['figure.figsize'] = 15,7

priced_apps=df[df['Price(in $)'] != 0.0]

price_plot = sns.countplot(priced_apps['Price(in $)'], palette = "inferno")

price_plot.set_xticklabels(price_plot.get_xticklabels(), rotation=90, ha="right")

price_plot 

plt.title('Number of apps for every price',size = 20)
# X['Installs'].value_counts()

rcParams['figure.figsize'] = 15,7

install_plot = sns.countplot(y="Installs",data=df, palette = "PuBu")

install_plot.set_xticklabels(install_plot.get_xticklabels(), rotation=90, ha="right")

install_plot 

plt.title('Installs count distribution',size = 20)
rcParams['figure.figsize'] = 15,7

ratings_plot = sns.countplot(x="Rating",data=df, palette = "inferno")

ratings_plot.set_xticklabels(ratings_plot.get_xticklabels(), rotation=90, ha="right")

ratings_plot 

plt.title('Rating Distribution',size = 20)
content_ratings = sns.violinplot(x="Content Rating",y="Rating",data=df, kind="box", height = 10 ,palette = "Set3")

content_ratings.set_xlabel(xlabel = 'Content Rating', fontsize = 9)

content_ratings.set_ylabel(ylabel = 'Rating', fontsize = 9)

content_ratings.set_title(label = 'Rating vs Content Rating', fontsize = 20)

plt.show()

# X.Installs.mean()

# X.Installs.median()

df= df.drop_duplicates(subset='App',keep='first')

df['Installs'] = df['Installs'].apply(lambda x: int(x))

df[['App','Installs','Category','Content Rating','Price(in $)']].head()

newdf=df[['App','Installs','Category','Content Rating','Price(in $)','Reviews','Genres']].groupby(['Installs'], sort = True)

newdf=newdf.filter(lambda x: x['Installs'].mean() >= 1000000000)

newdf=newdf.sort_values(by=['Installs'])

newdf

rcParams['figure.figsize'] = 15,7

free_categories = sns.countplot(x="Category",data=newdf, palette = "bone")

free_categories.set_xticklabels(free_categories.get_xticklabels(), rotation=90, ha="right")

free_categories 

plt.title('Top Categories for Free Apps',size = 20)
rcParams['figure.figsize'] = 15,7

free_genres = sns.countplot(y="Genres",data=newdf, palette = "spring")

free_genres.set_xticklabels(free_genres.get_xticklabels(), rotation=90, ha="right")

free_genres 

plt.title('Top Genres for Free Apps',size = 20)
new=df[['App','Category','Content Rating','Price(in $)','Reviews']].groupby(['Price(in $)'], sort = True)

new=new.filter(lambda x: x['Price(in $)'].mean() != 0)

new=new.sort_values(by=['Price(in $)'])

new

newdf2=df[['App','Installs','Genres']].groupby(['Installs'], sort = True)

newdf2=newdf2.filter(lambda x: x['Installs'].mean() >= 1000000)

newdf2=newdf2.sort_values(by=['Installs'])

newdf2



s1 = pd.merge(new, newdf2, how='inner', on=['App'])

s1

rcParams['figure.figsize'] = 15,7

paid_categories = sns.countplot(x="Category",data=s1, palette = "bone")

paid_categories.set_xticklabels(paid_categories.get_xticklabels(), rotation=90, ha="right")

paid_categories 

plt.title('Top Categories for PAID APPS',size = 20)
rcParams['figure.figsize'] = 15,7

paid_genres = sns.countplot(y="Genres",data=s1, palette = "spring")

paid_genres.set_xticklabels(paid_genres.get_xticklabels(), rotation=90, ha="right")

paid_genres 

plt.title('Top Genres for PAID APPS',size = 20)
df['new'] = pd.to_datetime(df['Last Updated'])

df.drop(labels = ['Last Updated'], axis = 1, inplace = True)

df.rename(columns={'new': 'Last Updated'}, inplace=True)
freq= pd.Series()

freq=df['Last Updated'].value_counts()

newfreq=freq[freq>50]

newfreq.plot()

plt.xlabel("Dates")

plt.ylabel("Number of updates")

plt.title("Time series plot of Last Updates")
df['Rating'] = df['Rating'].apply(lambda x: float(x))

df['Reviews'] = df['Reviews'].apply(lambda x: int(x))



newdf_rate=df[['App','Rating','Category','Content Rating']].groupby(['Rating'], sort = True)

newdf_rate=newdf_rate.filter(lambda x: x['Rating'].mean() >= 4.5)

newdf_rate=newdf_rate.sort_values(by=['Rating'])



newdf_reviews=df[['App','Reviews']].groupby(['Reviews'], sort = True)

newdf_reviews=newdf_reviews.filter(lambda x: x['Reviews'].mean() >= 255435)

newdf_reviews=newdf_reviews.sort_values(by=['Reviews'])



newdf_installs=df[['App','Installs']].groupby(['Installs'], sort = True)

newdf_installs=newdf_installs.filter(lambda x: x['Installs'].mean() >= 10000000)

newdf_installs=newdf_installs.sort_values(by=['Installs'])



s1 = pd.merge(newdf_reviews, newdf_rate, how='inner', on=['App'])

s2 = pd.merge(s1, newdf_installs, how='inner', on=['App'])

s2
rcParams['figure.figsize'] = 15,7

likeable_apps = sns.countplot(y="Category",data=s2, palette = "Set3")

likeable_apps.set_xticklabels(likeable_apps.get_xticklabels(), rotation=90, ha="right")

likeable_apps 

plt.title('CATEGORIES OF MOST LIKEABLE APPS ON THE ANDROID APP STORE',size = 20)
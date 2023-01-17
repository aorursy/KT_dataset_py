# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")
df_train.info()
df_train.head()

print(df_train.shape, df_test.shape)
list(df_train.columns)
df_train.isna().sum()
df_train.isnull().any().any()
df_train["revenue"].min()
df_train["revenue"].max()
for j, k in enumerate(df_train['belongs_to_collection'][:5]):

    print(j, k)
df_train1=df_train.drop(['belongs_to_collection'], axis=1)

df_test=df_test.drop(['belongs_to_collection'],axis=1)
for j, k in enumerate(df_train['genres'][:5]):

    print(j, k)
for j, k in enumerate(df_train['production_countries'][:10]):

    print(j, k)
sns.jointplot(x="budget", y="revenue", data=df_train, height=8, ratio=5, color="b")

plt.show()
sns.jointplot(x="popularity", y="revenue", data=df_train, height=8, ratio=5, color="b")

plt.show()
sns.jointplot(x="runtime", y="revenue", data=df_train, height=8, ratio=5, color="b")

plt.show()
sns.distplot(df_train.revenue)
df_train.revenue.describe()
sns.distplot(df_train.popularity)



df_train.popularity.describe()
df_train['logRevenue'] = np.log1p(df_train['revenue'])

sns.distplot(df_train['logRevenue'] )
df_train[['release_month','release_day','release_year']]=df_train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

# Some rows have 4 digits of year instead of 2, that's why I am applying (train['release_year'] < 100) this condition

df_train.loc[ (df_train['release_year'] <= 19) & (df_train['release_year'] < 100), "release_year"] += 2000

df_train.loc[ (df_train['release_year'] > 19)  & (df_train['release_year'] < 100), "release_year"] += 1900



releaseDate = pd.to_datetime(df_train['release_date']) 

df_train['release_dayofweek'] = releaseDate.dt.dayofweek

df_train['release_quarter'] = releaseDate.dt.quarter
plt.figure(figsize=(25,15))

sns.countplot(df_train['release_year'].sort_values())

plt.title("Movie Release count by Year",fontsize=25)

loc, labels = plt.xticks()

plt.xticks(fontsize=15,rotation=90)

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(df_train['release_month'].sort_values())

plt.title("Release Month Count",fontsize=10)

loc, labels = plt.xticks()

loc, labels = loc, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.xticks(loc, labels,fontsize=10)

plt.show()
plt.figure(figsize=(18,11))

sns.countplot(df_train['release_day'].sort_values())

plt.title("Release Day Count",fontsize=20)

plt.xticks(fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(df_train['release_dayofweek'].sort_values())

plt.title("Total movies released on Day Of Week",fontsize=20)

loc, labels = plt.xticks()

loc, labels = loc, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

plt.xticks(loc, labels,fontsize=20)

plt.show()
plt.figure(figsize=(20,12))

sns.countplot(df_train['release_quarter'].sort_values())

plt.title("Total movies released in a quarter",fontsize=20)

plt.show()
df_train['meanRevenueByYear'] = df_train.groupby("release_year")["revenue"].aggregate('mean')

df_train['meanRevenueByYear'].plot(figsize=(15,10),color="g")

plt.xticks(np.arange(1920,2018,4))

plt.xlabel("Release Year")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue By Year",fontsize=20)

plt.show()
df_train['meanRevenueByMonth'] = df_train.groupby("release_month")["revenue"].aggregate('mean')

df_train['meanRevenueByMonth'].plot(figsize=(15,10),color="g")

plt.xlabel("Release Month")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue Release Month",fontsize=20)

plt.show()
df_train['meanRevenueByDayOfWeek'] = df_train.groupby("release_dayofweek")["revenue"].aggregate('mean')

df_train['meanRevenueByDayOfWeek'].plot(figsize=(15,10),color="g")

plt.xlabel("Day of Week")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue by Day of Week",fontsize=20)

plt.show()
df_train['meanRevenueByQuarter'] = df_train.groupby("release_quarter")["revenue"].aggregate('mean')

df_train['meanRevenueByQuarter'].plot(figsize=(15,10),color="g")

plt.xticks(np.arange(1,5,1))

plt.xlabel("Quarter")

plt.ylabel("Revenue")

plt.title("Movie Mean Revenue by Quarter",fontsize=20)

plt.show()
df_train['meanruntimeByYear'] = df_train.groupby("release_year")["runtime"].aggregate('mean')

df_train['meanruntimeByYear'].plot(figsize=(15,10),color="g")

plt.xticks(np.arange(1920,2018,4))

plt.xlabel("Release Year")

plt.ylabel("Runtime")

plt.title("Movie Mean Runtime by Year",fontsize=20)

plt.show()
df_train['meanPopularityByYear'] = df_train.groupby("release_year")["popularity"].aggregate('mean')

df_train['meanPopularityByYear'].plot(figsize=(15,10),color="g")

plt.xticks(np.arange(1920,2018,4))

plt.xlabel("Release Year")

plt.ylabel("Popularity")

plt.title("Movie Mean Popularity by Year",fontsize=20)

plt.show()
df_train['meanBudgetByYear'] = df_train.groupby("release_year")["budget"].aggregate('mean')

df_train['meanBudgetByYear'].plot(figsize=(15,10),color="g")

plt.xticks(np.arange(1920,2018,4))

plt.xlabel("Release Year")

plt.ylabel("Budget")

plt.title("Movie Mean Budget by Year",fontsize=20)

plt.show()
plt.figure(figsize=(20,15))

sns.countplot(df_train['original_language'].sort_values())

plt.title("Native Language Count",fontsize=20)

plt.show()
df_train['status'].value_counts()
df_train.loc[df_train['status'] == "Rumored"][['status','revenue']]
df_train.loc[df_train['status']=='Released'][['status','revenue']].isnull().any().any()
df_test['status'].value_counts()
df_train['has_homepage'] = 1

df_train.loc[pd.isnull(df_train['homepage']) ,"has_homepage"] = 0

plt.figure(figsize=(15,8))

sns.countplot(df_train['has_homepage'].sort_values())

plt.title("Having Homepage?",fontsize=20)

plt.show()

sns.catplot(x="has_homepage", y="revenue", data=df_train)

plt.title('Revenue of the movies with and without image homepage');
df_train.head()
print("Missing status in Train data set", df_train['status'].isna().sum())

print("Missing runtime in Train data set", df_test['runtime'].isna().sum())

print("Missing production_companies in Train data set", df_test['production_companies'].isna().sum())
df_train[['release_month','release_day','release_year']]=df_train['release_date'].str.split('/',expand=True).replace(np.nan, 0).astype(int)

df_train['release_year'] = df_train['release_year']

df_train.loc[ (df_train['release_year'] <= 19) & (df_train['release_year'] < 100), "release_year"] += 2000

df_train.loc[ (df_train['release_year'] > 19)  & (df_train['release_year'] < 100), "release_year"] += 1900
releaseDate = pd.to_datetime(df_train['release_date']) 

df_train['release_dayofweek'] = releaseDate.dt.dayofweek 

df_train['release_quarter'] = releaseDate.dt.quarter     

    

df_train['originalBudget'] = df_train['budget']

df_train['inflationBudget'] = df_train['budget'] + df_train['budget']*1.8/100*(2018-df_train['release_year']) #Inflation simple formula

df_train['budget'] = np.log1p(df_train['budget']) 
plt.figure(figsize=(15,11)) #figure size



#It's another way to plot our data. using a variable that contains the plot parameters

g1 = sns.boxenplot(x='original_language', y='revenue', 

                   data=df_train[(df_train['original_language'].isin((df_train['original_language'].value_counts()[:10].index.values)))])

g1.set_title("Revenue by original language of the movies", fontsize=20) # title and fontsize

g1.set_xticklabels(g1.get_xticklabels(),rotation=45) # It's the way to rotate the xticks when we use variable to our graphs

g1.set_xlabel('Native language', fontsize=18) # Xlabel

g1.set_ylabel('Revenue per', fontsize=18) #Ylabel



plt.show()
(sns.FacetGrid(df_train[(df_train['release_year']\

                        .isin(df_train['release_year']\

                              .value_counts()[:5].index.values))],

               hue='release_year', height=5, aspect=2)

  .map(sns.kdeplot, 'budget', shade=True)

 .add_legend()

)

plt.title("Budget revenue for a years")

plt.show()
features_list = list(df_train.columns)

features_list =  [i for i in features_list if i != 'id' and i != 'revenue']
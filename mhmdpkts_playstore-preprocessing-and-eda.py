#import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
#load data

data = pd.read_csv('../input/googleplaystore.csv')
#show data first 5 record

data.head()
#show data last 5 record

data.tail()
#get information about data

data.info()
data = data.drop(["App","Current Ver","Android Ver"],1)
#detect null cols and null rate

nulls = [i for i in data.isna().any().index if data.isna().any()[i]==True]

rates = []

counts = []

for i in nulls:    

    rates.append((data[i].isna().sum()/data.shape[0])*100)

    counts.append(data[i].isna().sum())

null_df = pd.DataFrame.from_dict({"Col":nulls,"Count":counts,"Null_Rates":rates})

null_df
#delete Type,Content Rating, Current Ver, Android Ver null values row

df_train = data.copy()

for i in ['Type','Content Rating']:

    df_train = df_train.drop(df_train.loc[df_train[i].isnull()].index,0)

df_train.info()
df_train.Rating.describe()
#fill rating null values with mean quartiles

x = sum(df_train.Rating.describe()[4:8])/4

df_train.Rating = df_train.Rating.fillna(x)

print("Dataset contains ",df_train.isna().any().sum()," Nan values.")
df_train = df_train[df_train["Rating"]<=5]
#get unique values in Catagory feature 

df_train.Category.unique()
# convert to categorical Categority by using one hot tecnique 

df_dummy = df_train.copy()

df_dummy.Category = pd.Categorical(df_dummy.Category)



x = df_dummy[['Category']]

del df_dummy['Category']



dummies = pd.get_dummies(x, prefix = 'Category')

df_dummy = pd.concat([df_dummy,dummies], axis=1)

df_dummy.head()
#Genres unique val

df_dummy["Genres"].unique()
plt.figure(figsize=(25,6))

sns.barplot(x=df_dummy.Genres.value_counts().index,y=df_dummy.Genres.value_counts())

plt.xticks(rotation=80)

plt.title("Genres and their counts")

plt.show()
np.sort(df_dummy.Genres.value_counts())
lists = []

for i in df_dummy.Genres.value_counts().index:

    if df_dummy.Genres.value_counts()[i]<20:

        lists.append(i)



print(len(lists)," genres contains too few (<20) sample")

df_dummy.Genres = ['Other' if i in lists else i for i in df_dummy.Genres] 
df_dummy.Genres = pd.Categorical(df_dummy['Genres'])

x = df_dummy[["Genres"]]

del df_dummy['Genres']

dummies = pd.get_dummies(x, prefix = 'Genres')

df_dummy = pd.concat([df_dummy,dummies], axis=1)
df_dummy.shape
#get unique values in Contant Rating feature 

df_dummy['Content Rating'].value_counts(dropna=False)
#object(string) values transform to ordinal in Content Rating Feature without nan

df = df_dummy.copy()

df['Content Rating'] = df['Content Rating'].map({'Unrated':0.0,

                                                 'Everyone':1.0,

                                                 'Everyone 10+':2.0,

                                                 'Teen':3.0,

                                                 'Adults only 18+':4.0,

                                                 'Mature 17+':5.0})

df['Content Rating'] = df['Content Rating'].astype(float)

df.head()
#change type to float

df2 = df.copy()

df2['Reviews'] = df2['Reviews'].astype(float)
df2["Size"].value_counts()
#clean 'M','k', fill 'Varies with device' with median and transform to float 

lists = []

for i in df2["Size"]:

    if 'M' in i:

        i = float(i.replace('M',''))

        i = i*1000000

        lists.append(i)

    elif 'k' in i:

        i = float(i.replace('k',''))

        i = i*1000

        lists.append(i)

    else:

        lists.append("Unknown")

    

k = pd.Series(lists)

median = k[k!="Unknown"].median()

k = [median if i=="Unknown" else i for i in k]

df2["Size"] = k



del k,median,lists
#clean 'M'and transform to float 

print("old: ",df['Size'][10]," new: ",df2['Size'][10])
#clean '$' and transform to float 

df2['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in df2['Price'] ] 
print("old: ",df['Price'][9054]," new: ",df2['Price'][9054])
df2.Installs.unique()
df2["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in df2["Installs"] ]
print("old: ",df['Installs'][0]," new: ",df2['Installs'][0])
df2["Type"].unique()
df2.Type = df2.Type.map({'Free':0,"Paid":1})
df2["Last Updated"][:3]
from datetime import datetime

df3 = df2.copy()

df3["Last Updated"] = [datetime.strptime(i, '%B %d, %Y') for i in df3["Last Updated"]]
df3 = df3.set_index("Last Updated")

df4 = df3.sort_index()

df4.head()
df4.isna().any().sum()
data = df4.copy()

data.shape
data.info()
#additional libraries

from scipy.stats import norm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



style = sns.color_palette("ch:2.5,-.2,dark=.3")
#histogram

plt.figure(figsize=(10,5))

sns.distplot(data['Rating'],color='g');

plt.title("Rating Distrubition")

plt.show()
#skewness and kurtosis

print("Skewness: %f" % data['Rating'].skew())

print("Kurtosis: %f" % data['Rating'].kurt())
#histogram

plt.figure(figsize=(10,5))

sns.countplot(data['Type'],color='red',palette=style);

plt.title("Type Distrubition")

plt.show()
#histogram

plt.figure(figsize=(8,6))

sns.barplot(x=data['Installs'],y=data.Reviews,color='b',palette=sns.color_palette("ch:2.5,-.2,dark=.3"));

plt.title("Installs Distrubition")

plt.xticks(rotation=80)

plt.show()
#boxplot plot installs/rates

ax = plt.figure(figsize=(10,5))

sns.set()

sns.boxplot(x="Installs", y="Rating", data=data)

plt.title("Installs/Rating")

plt.xticks(rotation=80)

plt.show()
chart_data = data.loc[:,"Category_ART_AND_DESIGN":"Category_WEATHER"]

chart_data["Rating"] = data["Rating"]

for i in range(0, len(chart_data.columns), 5):

    sns.pairplot(data=chart_data,

                x_vars=chart_data.columns[i:i+5],

                y_vars=['Rating'])
import math

#del chart_data["Rating"]

l = len(chart_data.columns.values)

r = math.ceil(l/5)



chart_data["Type"] = data["Type"]

j=1

plt.subplots(figsize=(15,10),tight_layout=True)

for i in chart_data.columns.values:

    if i=="Type":

        continue

    d = chart_data[chart_data[i]==1]

    plt.subplot(r, 5, j)

    plt.hist(d["Type"])

    plt.title(i)

    j +=1

    

plt.show()
chart_data = data[data["Price"]>0]

chart_data = chart_data.sort_values(by=['Price'],ascending=False)

chart_data = chart_data.head(100)

#chart_data

dic = {}

cols = chart_data.loc[:,"Category_ART_AND_DESIGN":"Category_WEATHER"].columns.values

for i in cols:

    dic[i]=0

    

for i in range(100):

    x = chart_data.iloc[[i]]

    x = x.loc[:,"Category_ART_AND_DESIGN":"Category_WEATHER"]

    for j in x.columns.values:

        if (x[j][0] == 1):

            dic[j]= dic[j] + 1



plt.figure(figsize=(12,5))

plt.bar(dic.keys(), dic.values(), color='g')

plt.xticks(rotation=85)

plt.title("Categories of the 100 most expensive applications")

plt.show()

    
fig,ax = plt.subplots(figsize=(8,7))

ax = sns.heatmap(data[["Reviews","Price","Rating","Installs","Size"]].corr(), annot=True,linewidths=.5,fmt='.1f')

plt.show()
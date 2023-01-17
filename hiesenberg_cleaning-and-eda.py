import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import plotly.graph_objs as go

import plotly.graph_objs as go

import plotly.offline as pyoff

%matplotlib inline

sns.set()
df=pd.read_csv('../input/agricultural-raw-material-prices-19902020/agricultural_raw_material.csv')

df.head()
#Checking Null Values of each column

df.isnull().sum()
#Filling na values

df.fillna(method='ffill',inplace=True)
#Creating Year Column

df['Year']=df['Month']

for i in range(len(df)):

    df['Year'][i]=df['Month'][i][-2:]
#Improving data of year column

for i in range(117):

    df['Year'][i]='19'+df['Year'][i]



for i in range(117,361):

    df['Year'][i]='20'+df['Year'][i]
#Extracting the month data

for i in range(len(df)):

    df['Month'][i]=df['Month'][i][:-3]
df.fillna(0,inplace=True)
df.dtypes
#Converting into numeric by removing special character

cols=['Fine wool Price','Coarse wool Price','Copra Price']

for col in cols:

    for i in range(len(df[col])):

        x=re.sub(',','',df[col][i])

        x=pd.to_numeric(x)

        df[col][i]=x
#Converting column to float

for col in cols:

    df[col]=df[col].astype(float)
names=df.columns

for name in names:

    if '%' in name:

        for i in range(len(df[name])):

            df[name][i]=df[name][i][:-1]
df.replace('',0,inplace=True)
names=df.columns

for name in names:

    if '%' in name:

        df[name]=df[name].astype(float)
df.dtypes
df.head()
#Year wise total price of each crop

cols=[]

for col in df.columns:

    if(('%' not in col)):

        cols.append(col)

year_wise_price=pd.DataFrame(df.groupby("Year")[cols].sum())

year_wise_price
#Plot to depict how price of each crop changes each year

year_wise_price.plot(figsize=(20,5))
#Plot to depict total money spent on each year

plt.figure(figsize=(15,4))

x=year_wise_price.sum(axis=1)

x.plot(kind='bar')
#Crop price sum month wise

month_wise_price=pd.DataFrame(df.groupby("Month")[cols].sum())

month_wise_price
#Plot of crops price month wise

month_wise_price.plot(figsize=(20,5))
#Money spent in each month 

plt.figure(figsize=(15,4))

x=month_wise_price.sum(axis=1)

x.plot(kind='bar')
#Most expensive crops of each year

pd.DataFrame(year_wise_price.idxmax(axis=1),columns=['Crop'])
cols=cols[1:-1]

cols
#Change of price of each crop every year

plt.figure(figsize=(20,5))

for col in cols:

    plt.figure(figsize=(20,5))

    sns.barplot(df['Year'],df[col])

    plt.ylabel(str(col))

    plt.show()
#Correlating matrix

plt.figure(figsize=(20,15))

sns.heatmap(df.corr(),annot=True)
#Finding columns with high correaltion

corr_={}

for col in cols:

    #print(col)

    X=df.drop(col,axis=1)

    Y=df[col]

    x=X.corrwith(Y)

    x=x[x>0.5]

    corr_.setdefault(col, [])

    for n in x.index:

        corr_[col].append(n)
corr_
#Removing empty list

def new_s(s):

    return {a:(b if not isinstance(b, dict) else new_s(b)) for a, b in s.items() if b}



corr_=new_s(corr_)

corr_
cols=list(corr_.keys())

cols
#ploting columns against its highly correlated columns

dict_col={}

for i in range(len(corr_)):

    dict_col[i]=df.loc[:,corr_[cols[i]]].columns



for i in range(len(dict_col)):

    plt.figure(figsize=(20,5))

    for col in dict_col[i]:

        sns.lineplot(df[cols[i]],y=df[col],label=str(col))



    plt.show()  

cols2=[]

for col in df.columns:

    if(('%' in col)):

        cols2.append(col)

cols2
cols2

df.dtypes
#Ploting change of percent each year of every crop

for col in cols2:

    plt.figure(figsize=(25,5))

    sns.lineplot(df['Year'],df[col])

    plt.show()
#import libraries 



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

sns.set_style('white')
#load data 

data=pd.read_csv('../input/top-women-chess-players/top_women_chess_players_aug_2020.csv')

data.head(10)
#Data information 

info=data.info()

shape=data.shape



print(info)

print(shape)
#Separate out columns -> str and numerical 



str_cols=data.select_dtypes('object').columns

num_cols=data.select_dtypes(['int64','float64']).columns

num_cols
null_cols=data.isnull().sum()

nan_cols = [i for i in data.columns if data[i].isnull().any()]



nn_cols=[]

sn_cols=[]



for nancol in nan_cols:

    if(nancol in num_cols):

        nn_cols.append(nancol)

    else:

        sn_cols.append(nancol)
plt.figure(figsize=(10,8))

sns.heatmap(data.isnull(),annot=False)
count_miss=len(data)-data[nn_cols[0]].isnull().count()

yob=data[nn_cols[0]]

sns.distplot(data[nn_cols[0]],hist=True,color='r') 
null_count_rapid=data[nn_cols[1]].isnull().sum()



sns.distplot(data[nn_cols[1]],hist=True,color='y')



data[nn_cols[1]].mean()
data[sn_cols[0]].value_counts()
df=pd.DataFrame(data.groupby('Federation')[sn_cols[0]].count().sort_values(ascending=False).head(10),

                )

indx=df.index

df.columns=['Counts']



plt.figure(figsize=(20,8))

b=sns.barplot(x=indx,y=df['Counts'])

b.set_xlabel("Countries ",fontsize=20)

b.set_ylabel("Title Title Counts" ,fontsize=20)


data['Title'].value_counts()





df=data[data['Title']=='GM']

df1=pd.DataFrame(df.groupby('Federation')['Title'].count().sort_values(ascending=False))

df1.columns=['GMs']



indx=df1.index





plt.figure(figsize=(20,8))

b=sns.barplot(x=indx,y=df1['GMs'])

b.set_xlabel("Countries ",fontsize=20)

b.set_ylabel("Total GM Counts" ,fontsize=20)



df3=df1['GMs'].head(5)



plt.figure(figsize=(20,8))

plt.pie(df3,labels=indx[:5],autopct='%4.2f%%',

        shadow=True, startangle=90)
df1=data[data['Standard_Rating']>2500]

df2=pd.DataFrame(df1.groupby('Federation')['Standard_Rating'].count().sort_values(ascending=False))

df2.columns=['MaxRating']





indx=df2.index



plt.figure(figsize=(20,8))

b=sns.barplot(x=indx,y=df2['MaxRating'])

b.set_xlabel("Countries ",fontsize=20)

b.set_ylabel("Max Rating Counts" ,fontsize=20)

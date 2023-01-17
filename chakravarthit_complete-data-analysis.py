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
import matplotlib.pyplot as plt

import pandas_profiling

from pandas.plotting import scatter_matrix
file ='/kaggle/input/youtube-new/USvideos.csv'

df = pd.read_csv(file)

df.head()
df.info()
df.isnull().sum()
#Since description is least usefull, so will drop it

df.dropna(axis=1,how='any',inplace = True)
df.head(3)
#Check for null or empty value's percenge

Total = df.isnull().sum().sort_values(ascending=False)

percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([Total,percent],axis=1,keys=['Total','percent'])

missing_data
df.head()

#Chenage the date columns to pandas date data type

df['publish_time'] = pd.to_datetime(df['publish_time'] ).dt.strftime('%y-%m-%d')

df['trending_date'] = df['trending_date'].str.replace( '.','-')

df['trending_date']=pd.to_datetime(df['trending_date'], format='%y-%d-%m')



#drop unwanted columns from DataFrame

del df['tags']

del df['thumbnail_link']

del df['video_id']

del df['category_id']
df.info()

df.head()
#Data Analysis starts from here

#Which Title have most likes

MaxLikes = df['likes'].max()

mask = df['likes'] == MaxLikes

df[mask]
#Which Title have most dislikes

MinLikes = df['dislikes'].min()

mask = df['dislikes'] == MinLikes

df[mask]
df[mask].count()
#What is the row of max likes in video had error or removed.

df.head()

mask1 = df[df['video_error_or_removed']]

Max_df = mask1['likes'].max()



mask1 [mask1['likes'] == Max_df]
#What is the row of MIN likes in video had error or removed.

mask1 = df[df['video_error_or_removed']]

Min_df = mask1['likes'].min()

mask1 [mask1['likes'] == Min_df]

df.head()
df['publish_time'] = pd.to_datetime(df['publish_time'] ).dt.strftime('%d-%m-%Y')

df['publish_year'] =df['publish_time'].str.split('-',expand=True)[2]

df['publish_time'] = pd.to_datetime(df['publish_time'] )

#df['publish_year'] = pd.to_datetime(df['publish_year'] )

df.info()

df.head(3)
#Which publish year has most views and its rows.

df.groupby('publish_year')['views'].sum().sort_values(ascending = False)



#Which year had most likes 



df.groupby('publish_year')['likes'].sum().sort_values(ascending = False)
#Which year had most dislikes 

df.groupby ('publish_year')['dislikes'].sum().sort_values(ascending = False)
#Which Title has most likes and views in 2017

mass3 = df['publish_year'] == '2017'

temp = df[mass3]



Max2017L = temp['likes'].max()

Max2017V = temp['views'].max()

mass1 = temp['likes'] == Max2017L

mass2 = temp['views'] == Max2017V



temp[ mass1 | mass2 ]



#Which Title has most likes and views in 2018

mass3 = df['publish_year'] == '2018'

temp = df[mass3]



Max2017L = temp['likes'].max()

Max2017V = temp['views'].max()

mass1 = temp['likes'] == Max2017L

mass2 = temp['views'] == Max2017V



temp[ mass1 | mass2 ]
#Visualizing the inference 

import seaborn as sns

from matplotlib import pyplot
#Visualize correlations

pyplot.scatter(df['views'], df['likes'])
#Calculate Covariance

#cov(X, Y) = (sum (x - mean(X)) * (y - mean(Y)) ) * 1/(n-1)

vmean = df['views'].mean()

lmean = df['likes'].mean()



vsum=0

#calculate sum (x - mean(X)

for val in df['views']:

    dif = val - vmean

    vsum = vsum + dif

    

lsum=0

#calculate sum (x - mean(Y)

for val in df['likes']:

    dif = val - lmean

    lsum = lsum + dif

total = lsum * vsum



le = len(df) -1



Covar = total /le



print(Covar)



df=[]

col_list = ['title', 'views', 'likes']

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.csv'):

            df1=pd.DataFrame(pd.read_csv(os.path.join(dirname, filename),header=0,usecols=col_list))

            df1['country']=filename[:2]

            

            df.append(df1)

train=pd.concat(df,axis=0,ignore_index=True)
df = []

col_list = ['title', 'views','likes']

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.csv'):

            df1 = pd.DataFrame(pd.read_csv (os.path.join (dirname, filename),header = 0 , usecols = col_list))

            df1['country']=filename[:2]

            df.append(df1)

            

train = pd.concat (df,axis=0,ignore_index=True)



            
train.shape

train.isna().sum()
train.describe()
train.corr()
train.head(10)
train.cov()


sns.heatmap(train.corr(),annot=True)
scatter_matrix(train)

plt.show()
sns.regplot(x='views',y='likes',data=train)

plt.title('Correlation between views and likes')
pd.DataFrame(train.groupby ('country').sum())

country = pd.DataFrame(train.groupby ('country').sum())
sns.barplot(x=country.index, y=country['views'])
sns.barplot(x=country.index, y=country['likes'])
titlewise=pd.DataFrame(train.groupby(by=['title']).sum())

titlewise

titlewise.sort_values(by=['views','likes'],ascending=False,inplace=True)



titlewise
titlewise[titlewise['views']==titlewise['views'].max()]
titlewise[titlewise['likes']==titlewise['likes'].max()]
sns.barplot(x=titlewise.index[:10],y=titlewise.likes[:10])

plt.xticks(rotation=90)
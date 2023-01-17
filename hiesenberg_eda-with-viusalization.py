import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected = True)

import plotly.graph_objs as go

import re

import plotly.express as px

pd.options.display.float_format = "{:.2f}".format

%matplotlib inline

sns.set()
df1=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df2=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
df1.head()
df2.head()
df1.shape,df2.shape
df1.isnull().sum()
df1.dropna(inplace=True)

df2.dropna(inplace=True)

df2.drop(['Translated_Review','Sentiment','Sentiment_Subjectivity'],axis=1,inplace=True)
aggregation_functions = {'Sentiment_Polarity': 'mean'}

df_new = df2.groupby(df2['App']).agg(aggregation_functions)
df_new['App'] = df_new.index

df_new=df_new.reset_index(drop=True)

df_new=df_new.loc[:,['App','Sentiment_Polarity']]
df=pd.merge(df1,df_new,on='App')
df.drop(['Last Updated','Current Ver','Android Ver','Content Rating'],axis=1,inplace=True)
df.head()
df.dtypes
df['Rating']=df['Rating'].astype(float)

df['Reviews']=df['Reviews'].astype(int)
df=df[~(df['Size']=='Varies with device')]

df['Size'] = df['Size'].map(lambda x: re.sub(r'k', '', x))

df['Size'] = df['Size'].map(lambda x: re.sub(r'M', '', x))

df['Size']=pd.to_numeric(df['Size'])
df['Installs'] = df['Installs'].map(lambda x: re.sub(r'\W+', '', x))

df['Installs']=pd.to_numeric(df['Installs'])
df['Price']=df['Price'].map(lambda x: re.sub(r'\W+', '', x))

df['Price']=pd.to_numeric(df['Price'])
df.drop_duplicates(subset='App',inplace=True)
df.head()
#App with best rating

df[df['Reviews']>500].sort_values('Rating',ascending=False).reset_index()[['App','Rating']][:10].drop_duplicates()
#App with worst rating

df[df['Reviews']>500].sort_values('Rating').reset_index()[['App','Rating']][:10].drop_duplicates()
#App with most Review

df.sort_values('Reviews',ascending=False).reset_index()[['App','Reviews']][:10].drop_duplicates()
#App with most Installs

df.sort_values('Installs',ascending=False).reset_index()[['App','Installs']][:10].drop_duplicates()
#App with best Sentiment Score

df[df['Reviews']>500].sort_values('Sentiment_Polarity',ascending=False).reset_index()[['App','Sentiment_Polarity']][:10].drop_duplicates()
#App with good rating and downloads

df.loc[(df['Rating']>4)&(df['Installs']>100000000)]
#App with poor rating and have high downloads

df.loc[(df['Rating']<3)&(df['Installs']>100000)]
#Distribution of Category in Android Market

cat=[]

for i in range(len(df['Category'].value_counts().index)):

    cat.append(df['Category'].value_counts().index[i])



size=[]

for i in range(len(df['Category'].value_counts())):

    size.append(df['Category'].value_counts()[i])

    

trace = go.Pie(labels = cat, values = size)

data = [trace]

fig = go.Figure(data = data)

iplot(fig)
#Average Rating of each category

aggregation_functions = {'Rating': 'mean'}

x=df.groupby(df['Category']).agg(aggregation_functions)

data = [go.Bar(

   x = x.index,

   y = x['Rating']

)]

fig = go.Figure(data=data)

iplot(fig)
#Distribution of rating of each category

plt.figure(figsize=(15,5))

ax = sns.violinplot(x="Category", y="Rating", data=df);



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
#Chnange of rating with respect to number of reviews

plt.figure(figsize=(15,5))

sns.lineplot(x="Rating", y="Reviews", data=df,hue='Type');
#Distribution of rating of paid and free app

sns.catplot(x="Type", y="Rating", data=df);
#How size affect the rating of app

plt.figure(figsize=(15,5))

fig = px.scatter(df, x="Rating", y="Size")

fig.show()
#How size affect the number of installation

plt.figure(figsize=(15,5))

sns.lineplot(x="Installs", y="Size", data=df,hue='Type');
#Paid or Free app category wise

plt.figure(figsize=(15,8))

ax = plt.scatter(x="Price", y="Category", data=df);



#ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
#Paid App Category

paid_apps=df[df['Type']=='Paid']['Category']

plt.figure(figsize=(8,5))

sns.countplot(paid_apps)
#Number of installs of paid and free app

aggregation_functions = {'Installs': 'sum'}

x1=df.groupby(df['Type']).agg(aggregation_functions)

sns.barplot(x=x1.index,y=x1['Installs'])

x1
#Correlation Matrix

sns.heatmap(df.corr(),annot=True)
#Distribution of Genre in Android Market

gen=[]

for i in range(len(df['Genres'].value_counts().index)):

    gen.append(df['Genres'].value_counts().index[i])



size=[]

for i in range(len(df['Genres'].value_counts())):

    size.append(df['Genres'].value_counts()[i])

    

trace = go.Pie(labels = gen, values = size)

data = [trace]

fig = go.Figure(data = data)

iplot(fig)
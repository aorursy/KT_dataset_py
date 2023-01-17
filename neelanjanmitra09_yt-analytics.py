#import the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#read the dataset
df=pd.read_csv(r'../input/av-guided-hackathon/train.csv',low_memory=False)
#look at the first 5 entries of the dataset
df.head(5)
#We can see that there is a 'publish_date' column. We will use the datetime library to extract features from this
import datetime as dt
#take a look at the total number of columns in the dataset
df.columns
#a little more information
df.info()
df.describe
df.shape
#check for missing values
df.isna().sum()
#plotting the density function of the target variable
df['likes'].plot(kind='density',title='Density function of target var(likes)',figsize=(10,8))
#creating a copy of the dataset
df_temp=df.copy()
df_temp.head(2)
#performing the log transform
df_temp['likes']= np.log1p(df_temp['likes'])
#Visualising the new distribution
df_temp['likes'].plot(kind='density',title='transformed Density function of target var(likes)',figsize=(10,8))
#Check for unique values in the features
df_temp.nunique()
#numerical features
num_feats= [feature for feature in df_temp.columns if df_temp[feature].dtypes != 'O']
num_feats
for feature in df_temp[num_feats].drop(columns=['video_id','likes']):
    df[feature].plot(kind='density')
    plt.xlabel(feature)
    plt.show()
#Applying log transfrom to all the skewed data. we have dropped video_id and likes 
for feature in df_temp[num_feats].drop(columns=['video_id','likes']):
    df_temp[feature]=np.log1p(df_temp[feature])
#Visualisation
for feature in df_temp[num_feats].drop(columns=['video_id','likes']):
    df_temp[feature].plot(kind='density')
    plt.xlabel(feature)
    plt.show()
#categorical variables
cat_feats=[feature for feature in df_temp.columns if df_temp[feature].dtypes=='O']
cat_feats
df_temp.head(5)
#Lets visualise the correlation between various features using a heatmap
sns.heatmap(df_temp[num_feats].corr(),annot=True)
#Country_code feature distribution
plt.figure(figsize=(10,10))
df_temp['country_code'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()
#finding the top 10 countries
top10=df_temp['channel_title'].value_counts().head(10)
#visualisation
top10.plot(kind='bar')
top10.columns=['channel_title','num_videos']
country_wise_channels = df_temp.groupby(['country_code', 'channel_title']).size().reset_index()
country_wise_channels.columns = ['country_code', 'channel_title', 'num_videos']
country_wise_channels = country_wise_channels.sort_values(by = 'num_videos', ascending=False)
fig, axes = plt.subplots(4, 1, figsize=(10, 20))

for i, c in enumerate(df_temp['country_code'].unique()):
  country = country_wise_channels[country_wise_channels['country_code'] == c][:10]
  _ = sns.barplot(x = 'num_videos', y = 'channel_title', data = country, ax = axes[i])
  _ = axes[i].set_title(f'Country Code {c}')

plt.tight_layout()
sns.catplot(x='country_code',y='views',data=df_temp)
#likes per country
df_temp.groupby('country_code')['likes'].count().plot(kind='bar')
#AVERAGE likes per country
df_temp.groupby('country_code')['likes'].mean().sort_values().plot(kind='bar')
#handling of datetime variables
import datetime as dt
#conversion
df_temp['publish_date'] = pd.to_datetime(df_temp['publish_date'], format='%Y-%m-%d')
df_temp['publish_date']
#lets see what is the tenure through which our data is spread out
df_temp['publish_date'].min(),df_temp['publish_date'].max()
#lets try and see year wise videos
df_temp.groupby('publish_date').size().plot()
#above data is unreadable. lets cosider data points post 2015
new_date=df_temp[df_temp['publish_date']>'2017-11']
new_date.sort_values(by='publish_date').groupby('publish_date').size().plot(figsize=(18, 6))
#mean likes per month
new_date.sort_values(by='publish_date').groupby('publish_date')['likes'].mean().plot(figsize=(18, 6))
#videos by country
new_date.sort_values(by='publish_date').groupby('publish_date')['country_code'].count().plot(figsize=(18, 6))
#avg/mean likes per country
new_date.sort_values(by='publish_date').groupby(['publish_date','country_code'])['likes'].mean().plot(subplots=True,figsize=(18, 6))
#the above is not a good way of representation. Use pivot table
new_date.pivot_table(index='publish_date',columns='country_code',values='likes').plot(subplots=True,figsize=(20,20))
cat_feats
text_feats=['tags','title','description']
from wordcloud import WordCloud, STOPWORDS
wc = WordCloud(stopwords = set(list(STOPWORDS) + ['|']), random_state = 42)
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = [ax for axes_row in axes for ax in axes_row]
for i, c in enumerate(text_feats):
  op = wc.generate(str(df_temp[c]))
  _ = axes[i].imshow(op)
  _ = axes[i].set_title(c.upper(), fontsize=24)
  _ = axes[i].axis('off')

_ = fig.delaxes(axes[3])

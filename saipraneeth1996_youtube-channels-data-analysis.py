#Importing neccessary libraries

import numpy as np

import pandas as pd



#Importing neccessary plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
#Reading the data set using pandas

data = pd.read_csv('../input/Data.csv')

data.shape
data.head()
type(data)
print('Descriptive Statastics of our Data:')

data.describe().T
print('Showing Meta Data :')

data.info()
data.columns.tolist()
#renaming columns

data.columns = ['rank','grade','channel_name','video_uploads','subscribers','video_views']

data.head()
#Checking for missing values

pd.isnull(data).sum()
data[data['video_uploads']=='--']
data[data['subscribers']=='-- '].subscribers.count()
df = data.replace(['--','-- '],[np.nan,np.nan])
#changing data type to float 

df['video_uploads'] = df['video_uploads'].astype('float64')

df['subscribers'] = df['subscribers'].astype('float64')
df.tail()
#since rank is in 'object' datatype we are converting it to float for data analysis

df['rank'] = np.arange(1,df.shape[0]+1) 

df['rank'] = df['rank'].astype('int64')
df.dtypes
df.head()
df.describe().T
#Finding the top 10 youtube channels as ordered by ‘number of subcribers’.



df.sort_values(by='subscribers',ascending=False)[['channel_name','subscribers']].head(10)
#Generating a table which only contains data for channels which have less than 5000 video uploads. 

#Sorting this table by Rank.



vu_5000 = df[df.video_uploads<5000]



print('no. of channels which have less than 5000 video uploads : ',vu_5000.shape[0])

vu_5000.sort_values(by='rank',ascending=True)[['channel_name','video_uploads']].head(10)
#Is there a relation ship between the number of video uploads and the number of subscribers?

sns.set() #setting seaborn plotting style

sns.scatterplot(x='video_uploads',y='subscribers',data = df)

print('correlation value :',df.video_uploads.corr(df.subscribers))
#Plotting the distribution of video views for the top 500 channels.



top_500 = df.sort_values(by='rank',ascending=True)['video_views'].head(500) #taking top 500 channels by rank

sns.distplot(top_500)

plt.title('distribution of video views for the top 500 channels')
df.head()
#Plotting a gradewise distribution of all the youtube channels. Which grade has the highest number of channels?



print(df.grade.value_counts())

sns.countplot(df.grade)
#Creating correlation matrix for our data.

print('the correlation matrix of our data is :')

df.corr()
print('Correlation Heat map of the data :')

plt.figure(figsize=(8,6))

sns.heatmap(df.corr(),annot=True,fmt='.2f',vmin=-1,vmax=1)

plt.show()
sns.pairplot(df,diag_kind = 'kde')
#Creating a new column in the dataset which contains the number of subcribers per video upload. 

#and finding the top 5 channels based on this calculated column.



df['subs_per_upload'] = df['subscribers']/df['video_uploads']

print('top 5 channels based on subscribers per video upload  are: ')

df.sort_values(by='subs_per_upload',ascending=False)[['channel_name','subs_per_upload']].head()
#For A++ grade channels, plotting the distributions for the following columns:

#a. Video Uploads

#b. Subscribers

#c. Video views



get_channel  =  df[df.grade == 'A++ ']

sns.distplot(get_channel['video_views'])
sns.distplot(get_channel['video_uploads'])
sns.distplot(get_channel['subscribers'])
cols  = df.columns.tolist()

cols
# Are there any outliers in the data?? Lets check it out using boxplots.



def continous_data(i):

    if df[i].dtype!='object':

        sns.boxplot(df[i])

        plt.title("Boxplot---"+str(i))

        plt.show()

        plt.title("histogram---"+str(i))

        df[i].plot.kde()

        plt.show()

        sns.set()

        plt.clf()



for k in cols:

    continous_data(i=k)        
#i. The advertiser wants to advertise in channels which regularly release fresh videos.What are his top 5 choices?



print('Top 5 Channels which regularly release fresh videos')

df.sort_values(by=['video_uploads'],ascending=False)[['channel_name']].head(5)
#ii.The advertiser has some constraints with regard to the cost of advertising. 

#The cost depends on the number of average views per video. 

#Suggest the best 10 channels with average views between 1000000 to 500000.



df['avg_views'] = df['video_views']/df['video_uploads']

avg_btwn = df[(df.avg_views>500000) & (df.avg_views<1000000)]



print('Top 10 Channels with average views between 1000000 to 500000 in decreasing order of Cost')

avg_btwn.sort_values(by=['avg_views'],ascending=False)[['channel_name','avg_views']].head(10)
print('Top 10 Channels with average views between 1000000 to 500000 with Increasing order of Cost')

avg_btwn.sort_values(by=['avg_views'],ascending=True)[['channel_name','avg_views']].head(10)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/twitchdata/twitchdata-update.csv') #importing our data
df.head() #checking the head of our data
df.isnull().sum() #checking for null values in ou data
df.dtypes #checking the type of each column
df.describe().T
plt.style.use('dark_background') #checking the stream times of top 50 streamers

plt.figure(figsize = (20,7))

df['Stream time(minutes)'].head(50).plot.bar(color = 'orangered')

plt.title('Comparing the different stream times (in minutes)')

plt.xlabel('Streamers')

plt.ylabel('Count')

plt.show()
plt.style.use('dark_background') #checking the followers gained by our Top 50 Streamers

plt.figure(figsize = (20,7))

df['Followers gained'].head(50).plot.bar(color = 'orangered')

plt.title('Comparing the followers gained by our Top 50 Streamers')

plt.xlabel('Streamers')

plt.ylabel('Count')

plt.show() 
plt.style.use('dark_background') #checking the views gained by our Top 50 Streamers

plt.figure(figsize = (20,7))

df['Views gained'].head(50).plot.bar(color = 'orangered')

plt.title('Comparing the views gained by our Top 50 Streamers')

plt.xlabel('Streamers')

plt.ylabel('Count')

plt.show()  
plt.style.use('dark_background') #checking the Average nmber of viewers of our Top 50 Streamers

plt.figure(figsize = (20,7))

df['Average viewers'].head(50).plot.bar(color = 'orangered')

plt.title('Comparing the average viewers of our Top 50 Streamers')

plt.xlabel('Streamers')

plt.ylabel('Count')

plt.show()  
plt.style.use('dark_background') #checking the streamers that stream in a perticular language

plt.figure(figsize = (20,7))

df['Language'].value_counts().head(20).plot.bar(color = 'orangered')

plt.title('Languages that Streamers stream in')

plt.xlabel('Languages')

plt.ylabel('Count')

plt.show()
df.dtypes
sns.countplot(x='Partnered',data = df) #checking how many are twitch partnered
df[df['Partnered'] == True][['Channel', 'Watch time(Minutes)', 'Stream time(minutes)', 'Followers']].head(10) #checking the top 10 streamers that are twitch partnered
sns.countplot(x='Mature',data = df) #checking how many streams are tagged as mature
df[df['Mature'] == True][['Channel', 'Watch time(Minutes)', 'Stream time(minutes)', 'Followers']].head(10) #checking the top 10 streamers with mature streams
plt.figure(figsize=(12,8))

sns.heatmap(df[['Channel', 'Watch time(Minutes)', 'Stream time(minutes)', 'Followers','Peak viewers','Average viewers','Followers gained','Views gained','Partnered','Mature','Language']].corr(), annot = True) #overall correlation between the various columns present in our data

plt.title('Overall relation between columns of the Dataset', fontsize = 20)

plt.show()
def streamer(x): #method to check stats of an individual streamer

    return df.loc[df['Channel']==x]
def lang(x): #method to check the details about a streamer that streams in a particular language

        return df[df['Language'] == x][['Channel','Followers','Partnered','Mature']].head(10)
streamer('Anomaly')
lang('Spanish')
plt.figure(figsize=(12,8)) #comparing streaming time v/s followers gained

sns.lineplot(df['Stream time(minutes)'], df['Followers gained'], palette = "Set1")

plt.title('Streaming time v/s Followers gained', fontsize = 20)

plt.show()
plt.figure(figsize=(12,8)) #comparing streaming time v/s average viewers

sns.lineplot(df['Stream time(minutes)'], df['Average viewers'], palette = "Set1")

plt.title('Streaming time v/s Average Viewers', fontsize = 20)

plt.show()
df.head()
def streamtime(x): #method to check the streamer that had the most streaming time in our data

    return df.loc[df['Stream time(minutes)']==x]

def watchtime(x): #method to check the streamer that had the most watch time in our data

    return df.loc[df['Watch time(Minutes)']==x]

def avgviewers(x): #method to check the streamer that had the most number of average viewers

    return df.loc[df['Average viewers']==x]

def follow(x): #method to check the streamer that had the most followers in our data

    return df.loc[df['Followers']==x] 

def followgained(x): #method to check the streamer that had the most followers gained in our data

    return df.loc[df['Followers gained']==x] 

def viewgained(x): #method to check the streamer that had the most views gained in our data

    return df.loc[df['Views gained']==x] 
streamtime(df['Stream time(minutes)'].max())
watchtime(df['Watch time(Minutes)'].max())
avgviewers(df['Average viewers'].max())
follow(df['Followers'].max())
followgained(df['Followers gained'].max())
viewgained(df['Views gained'].max())
plt.figure(figsize=(20,8)) #comparing streamers on basis of their number of followers

top = ('Tfue', 'summit1g','NICKMERCS')

df2 = df.loc[df['Channel'].isin(top)  & df['Followers'] ]



ax = sns.barplot(x=df2['Channel'], y=df2['Followers'], palette="Set1");

ax.set_title(label='Channel comparison on basis of no.of Followers', fontsize=20);
from wordcloud import WordCloud

plt.subplots(figsize=(12,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Language))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from wordcloud import WordCloud

plt.subplots(figsize=(12,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.Channel))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
df1 = df.copy()
df1.head()
df1.dtypes
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
X = df1[['Watch time(Minutes)','Stream time(minutes)','Peak viewers','Average viewers','Followers','Views gained']]

y = df1['Followers gained']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
linear  = LinearRegression()

linear.fit(X_train,y_train)

pred = linear.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error

print('r2 score: '+str(r2_score(y_test, pred)))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, pred))))
df1.head()
user_input = [[6196161750,215250,222720,27716,3246298,93036735]]        #['Watch time(Minutes)','Stream time(minutes)','Peak viewers','Average viewers','Followers','Views gained']

user_pred = linear.predict(user_input)

print("Follower Gained by the streamer are:-",user_pred)
#Visualising the results

plt.figure(figsize=(12,8))

sns.regplot(pred,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'},line_kws={'color':'red','alpha':0.5})

plt.xlabel('Followers Gained')

plt.ylabel('Features')

plt.title("Linear Prediction of Followers Gained by a Streamer")

plt.show()
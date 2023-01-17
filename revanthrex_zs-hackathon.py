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
# install packages to use state of art pretrained Sentiment model

!pip  install vaderSentiment

!pip install textblob
import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec 

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import matplotlib_venn as venn

from gensim.summarization.summarizer import summarize

import datetime 

import statistics as stat

import vaderSentiment

sample_submission = pd.read_csv("../input/hacky-zs/dataset/sample_submission.csv")

test = pd.read_csv("../input/hacky-zs/dataset/test_file.csv")

train = pd.read_csv("../input/hacky-zs/dataset/train_file.csv")
train.shape
train.info()
train.head(10)
df= train

print("source_len",len(df['Source'].unique()))

print("source_varients",df['Source'].unique())

print("topic_varients",df['Topic'].unique())
def topic_obama(data):

    if data == 'obama':

        return 1

    else:

        return 0

    

def topic_economy(data):

    if data == 'economy':

        return 1

    else:

        return 0

def topic_microsoft(data):

    if data == 'microsoft':

        return 1

    else:

        return 0

def topic_palestine(data):

    if data == 'palestine':

        return 1

    else:

        return 0



df['topic_obama'] = df['Topic'].apply(topic_obama)

df['topic_economy'] = df['Topic'].apply(topic_economy)

df['topic_microsoft'] = df['Topic'].apply(topic_microsoft)

df['topic_palestine'] = df['Topic'].apply(topic_palestine)
dt = test

#test dataset

def topic_obama(data):

    if data == 'obama':

        return 1

    else:

        return 0

    

def topic_economy(data):

    if data == 'economy':

        return 1

    else:

        return 0

def topic_microsoft(data):

    if data == 'microsoft':

        return 1

    else:

        return 0

def topic_palestine(data):

    if data == 'palestine':

        return 1

    else:

        return 0



dt['topic_obama'] = dt['Topic'].apply(topic_obama)

dt['topic_economy'] = dt['Topic'].apply(topic_economy)

dt['topic_microsoft'] = dt['Topic'].apply(topic_microsoft)

dt['topic_palestine'] = dt['Topic'].apply(topic_palestine)
df
#check for rows which are white spaces

print(df[df['Source']==' '])



#converts all empty values into Nan

df
#count no of nan

df.isnull().sum()
df = df.dropna()

df.shape
source = df['Source'].unique()

for i in  source:

    print(i )
#no of unique sources

print('No of Unique sources=',len(df['Source'].unique()))



#no of unique sources which are mentioned only once 

val_count = df['Source'].value_counts()

count = 0  

for i in val_count: 

    if i ==1:

        count += 1

print("No of sources which aren't repeated=",count)
df['Month']=pd.DatetimeIndex(df['PublishDate']).month

df['Year']= pd.DatetimeIndex(df['PublishDate']).year

df['Day'] =pd.to_datetime(df['PublishDate']).dt.weekday_name

df['Time'] =pd.to_datetime(df['PublishDate']).dt.time

df['Date_day'] = pd.to_datetime(df['PublishDate']).dt.day

# dropping the publish date column after extracting the data out

df.drop(columns=['PublishDate'])
dt['Month']=pd.DatetimeIndex(dt['PublishDate']).month

dt['Year']= pd.DatetimeIndex(dt['PublishDate']).year

dt['Day'] =pd.to_datetime(dt['PublishDate']).dt.weekday_name

dt['Time'] =pd.to_datetime(dt['PublishDate']).dt.time

dt['Date_day'] = pd.to_datetime(dt['PublishDate']).dt.day

# dropping the publish date column after extracting the data out

dt.drop(columns=['PublishDate'])
#frequency of news articles for each day  

df['Day'].value_counts()
#frequency of news articles for each month 

df['Month'].value_counts()
#frequency of news articles for each Year 

df['Year'].value_counts()
#frequency of news articles for each time interval

print(df['Time'].value_counts())

#no of unique sources which are mentioned only once 

val_count = df['Time'].value_counts()

count = 0  

for i in val_count: 

    if i ==1:

        count += 1

print(count) 



# as the number of time interval with one data point is more we will keep those values
df['Date_day'].value_counts()
year_list = (2002,2008,2012)

month_list = [9,8,7,6,4]

df.drop(df[(df['Year']== 2002) |

                                   (df['Year']== 2008)|

                                   (df['Year']== 2012) |

                                   (df['Month']== 4)|

                                   (df['Month']== 6)|

                                   (df['Month']== 7)|

                                   (df['Month']== 8)|

                                   (df['Month']== 9)

                                  ].index, inplace=True)

#data frame shape after removal of all the unimportant rows 

df.shape
dt.shape
#gather sentiment data for each day 

day = df['Day'].unique()

sentiment_title=[]

sentiment_headline= []

for i in day:

    df1=df[df['Day'].str.contains(i)]

    sentiment_title.append(df1['SentimentTitle'].mean())

    sentiment_headline.append(df1['SentimentHeadline'].mean())

    

%matplotlib inline



pos = list(range(len(sentiment_title)))

width = 0.25



fig, ax = plt.subplots(figsize=(10,5))

    

plt.bar(pos, sentiment_title, width, alpha=0.5, color='#EE3224')

plt.bar([p+width for p in pos], sentiment_headline, width, alpha=0.5, color='#F78F1E')

    

ax.set_ylabel('sentiment_score')

ax.set_title('Sentiment vs Day')

ax.set_xticks([p +  width for p in pos])

ax.set_xticklabels(day)

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([min(sentiment_title+sentiment_headline)+0.001, max(sentiment_title+ sentiment_headline)+0.001])

plt.legend(['Sentiment_Title', 'Sentiment_Headline'], loc='upper left')

plt.grid()

plt.show()
#gather sentiment data for each day 

day = df['Date_day'].unique()

sentiment_title=[]

sentiment_headline= []

for i in day:

    df1=df[df['Date_day']==i]

    sentiment_title.append(df1['SentimentTitle'].mean())

    sentiment_headline.append(df1['SentimentHeadline'].mean())

    

%matplotlib inline



pos = list(range(len(sentiment_title)))

width = 0.25



fig, ax = plt.subplots(figsize=(10,5))

    

plt.bar(pos, sentiment_title, width, alpha=0.5, color='#EE3224')

plt.bar([p+width for p in pos], sentiment_headline, width, alpha=0.5, color='#F78F1E')

    

ax.set_ylabel('sentiment_score')

ax.set_title('Sentiment vs Date_Day')

ax.set_xticks([p +  width for p in pos])

ax.set_xticklabels(day)

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([min(sentiment_title+sentiment_headline)+0.001, max(sentiment_title+ sentiment_headline)+0.001])

plt.legend(['Sentiment_Title', 'Sentiment_Headline'], loc='upper left')

plt.grid()

plt.show()
#gather sentiment data for each month

month = df['Month'].unique()

sentiment_title=[]

sentiment_headline= []

for i in month:

    df1=df[df['Month'] == i]

    sentiment_title.append(df1['SentimentTitle'].mean())

    sentiment_headline.append(df1['SentimentHeadline'].mean())

    

%matplotlib inline



pos = list(range(len(sentiment_title)))

width = 0.25



fig, ax = plt.subplots(figsize=(10,5))

    

plt.bar(pos, sentiment_title, width, alpha=0.5, color='#EE3224')

plt.bar([p + width for p in pos], sentiment_headline, width, alpha=0.5, color='#F78F1E')

    

ax.set_ylabel('sentiment_score')

ax.set_title('Sentiment vs Month')

ax.set_xticks([p + width for p in pos])

ax.set_xticklabels(day)

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([min(sentiment_title+sentiment_headline)+0.001, max(sentiment_title+ sentiment_headline)+0.001])

plt.legend(['Sentiment_Title', 'Sentiment_Headline'], loc='upper left')

plt.grid()

plt.show()
#gather sentiment data for respective year 

month = df['Year'].unique()

sentiment_title=[]

sentiment_headline= []

for i in month:

    df1=df[df['Year'] == i]

    sentiment_title.append(df1['SentimentTitle'].mean())

    sentiment_headline.append(df1['SentimentHeadline'].mean())

    

%matplotlib inline



pos = list(range(len(sentiment_title)))

width = 0.25



fig, ax = plt.subplots(figsize=(10,5))

    

plt.bar(pos, sentiment_title, width, alpha=0.5, color='#EE3224')

plt.bar([p + width for p in pos], sentiment_headline, width, alpha=0.5, color='#F78F1E')

    

ax.set_ylabel('sentiment_score')

ax.set_title('Sentiment vs Year')

ax.set_xticks([p + width for p in pos])

ax.set_xticklabels(day)

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([min(sentiment_title+sentiment_headline)+0.001, max(sentiment_title+ sentiment_headline)+0.001])

plt.legend(['Sentiment_Title', 'Sentiment_Headline'], loc='upper left')

plt.grid()

plt.show()
#analysing the correlation between SentimentTitle and SentimentHeadline

matplotlib.style.use('ggplot')



plt.scatter(df['SentimentTitle'], df['SentimentHeadline'])

plt.show()
# Analysing the correlation between given variables

#pd.plotting.scatter_matrix(df, figsize=(15, 20))

#plt.show()
#Analysing the sentiment of title and headline

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def sentiment_analyse(text):

    return analyzer.polarity_scores(text)['compound']



df['TitlePolarity']= df['Title'].apply(sentiment_analyse)

df['HeadlinePolarity']=df['Headline'].apply(sentiment_analyse)

dt['TitlePolarity']= dt['Title'].apply(sentiment_analyse)

dt['HeadlinePolarity']=dt['Headline'].apply(sentiment_analyse)
from textblob import TextBlob

def sentiment_analyse_tb(text):

    testimonial = TextBlob(text)

    return testimonial.sentiment.polarity



df['TitlePolarity_tb']= df['Title'].apply(sentiment_analyse_tb)

df['HeadlinePolarity_tb']=df['Headline'].apply(sentiment_analyse_tb)

dt['TitlePolarity_tb']= dt['Title'].apply(sentiment_analyse_tb)

dt['HeadlinePolarity_tb']=dt['Headline'].apply(sentiment_analyse_tb)
#comparing between two methods textblob and vader

plt.scatter(df['TitlePolarity'], df['TitlePolarity_tb'])

plt.show()
plt.scatter(df['HeadlinePolarity'], df['HeadlinePolarity_tb'])

plt.show()
plt.scatter(df['TitlePolarity'], df['SentimentTitle'])

plt.show()
plt.scatter(df['HeadlinePolarity'], df['SentimentHeadline'])

plt.show()
plt.scatter(df['TitlePolarity_tb'], df['SentimentTitle'])

plt.show()
plt.scatter(df['HeadlinePolarity_tb'], df['SentimentHeadline'])

plt.show()
plt.scatter(((df['TitlePolarity_tb']+df['TitlePolarity'])/2), df['SentimentTitle'])

plt.show()
plt.scatter(((df['HeadlinePolarity_tb']+df['HeadlinePolarity'])/2), df['SentimentHeadline'])

plt.show()
#df1['SentimentTitle'] = ((dt['TitlePolarity']+dt['TitlePolarity_tb'])/2)

#df1['SentimentHeadline'] = ((df1['HeadlinePolarity']+df1['HeadlinePolarity_tb'])/2)

#df1=df1[['IDLink','SentimentTitle','SentimentHeadline']]

#df1.to_csv("submission1.csv",index=False)
#train

df['Hours']= df['Time'].apply(lambda x: x.hour)

df['Minutes']= df['Time'].apply(lambda x: x.minute)



#test

dt['Hours']= dt['Time'].apply(lambda x: x.hour)

dt['Minutes']= dt['Time'].apply(lambda x: x.minute)
onehotvar = pd.get_dummies(df['Day'])

df= pd.concat([df,onehotvar],axis =1)



#test data set 

onehotvar = pd.get_dummies(dt['Day'])

dt= pd.concat([dt,onehotvar],axis =1)
df.columns
df_title = df[['TitlePolarity','TitlePolarity_tb','topic_obama', 'topic_economy', 'topic_microsoft',

       'topic_palestine','Year','Month', 'Date_day','Friday', 'Monday', 'Saturday','Sunday', 'Thursday',

        'Tuesday', 'Wednesday', 'Hours', 'Minutes','Facebook', 'GooglePlus', 'LinkedIn', 'SentimentTitle',

       ]]

dt_title = dt[['TitlePolarity','TitlePolarity_tb','topic_obama', 'topic_economy', 'topic_microsoft',

       'topic_palestine','Year','Month', 'Date_day','Friday', 'Monday', 'Saturday',

       'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'Hours', 'Minutes','Facebook', 'GooglePlus', 'LinkedIn',

       ]]



df_headline = df[['HeadlinePolarity','HeadlinePolarity_tb','topic_obama', 'topic_economy', 'topic_microsoft',

       'topic_palestine','Year','Month', 'Date_day', 'Friday', 'Monday', 'Saturday',

       'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'Hours', 'Minutes','Facebook', 'GooglePlus', 'LinkedIn',

       'SentimentHeadline']]

dt_headline = dt[['HeadlinePolarity','HeadlinePolarity_tb','topic_obama', 'topic_economy', 'topic_microsoft',

       'topic_palestine','Year','Month', 'Date_day', 'Friday', 'Monday', 'Saturday',

       'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'Hours', 'Minutes','Facebook', 'GooglePlus', 'LinkedIn'

                 ]]
df_title['titlepolarity'] = (df['TitlePolarity']+df['TitlePolarity_tb'])/2

dt_title['titlepolarity'] = (dt['TitlePolarity']+dt['TitlePolarity_tb'])/2



df_headline['headlinepolarity'] = (df['HeadlinePolarity']+df['HeadlinePolarity_tb'])/2

dt_headline['headlinepolarity'] = (dt['HeadlinePolarity']+dt['HeadlinePolarity_tb'])/2
df_title =df_title.drop(['TitlePolarity', 'TitlePolarity_tb'], axis=1)

dt_title =dt_title.drop(['TitlePolarity', 'TitlePolarity_tb'], axis=1)



df_headline =df_headline.drop(['HeadlinePolarity','HeadlinePolarity_tb'], axis=1)

dt_headline =dt_headline.drop(['HeadlinePolarity','HeadlinePolarity_tb'], axis=1)
df_title= pd.concat([df_title['titlepolarity'],df_title.iloc[:,0:20]],axis=1)

df_headline= pd.concat([df_headline['headlinepolarity'],df_headline.iloc[:,0:20]],axis=1)



#test

dt_title= pd.concat([dt_title['titlepolarity'],dt_title.iloc[:,0:19]],axis=1)

dt_headline= pd.concat([dt_headline['headlinepolarity'],dt_headline.iloc[:,0:19]],axis=1)
import numpy as np

from sklearn.model_selection import train_test_split

# for title 

X_train, X_test, y_train, y_test = train_test_split(df_title.iloc[:,0:20], df_title.iloc[:,20], test_size=0.33, random_state=42)



# for Headline

Xh_train, Xh_test, yh_train, yh_test = train_test_split(df_headline.iloc[:,0:20], df_headline.iloc[:,20], test_size=0.33, random_state=42)
print(X_train.shape,X_test.shape)

print(Xh_train.shape,Xh_test.shape)
# for title

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42,max_depth = 15,verbose =1)

rf.fit(X_train, y_train)



#for headline

rfh = RandomForestRegressor(n_estimators = 100, random_state = 42, max_depth = 15,verbose =1)

rfh.fit(Xh_train, yh_train)
#prediction

predictions = rf.predict(X_test)

predictions_h = rfh.predict(Xh_test)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, predictions)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(yh_test, predictions_h)
# pred test

dt['SentimentTitle'] = rf.predict(dt_title)

dt['SentimentHeadline'] = rfh.predict(dt_headline)



#submission file 

sub = dt[['IDLink','SentimentTitle','SentimentHeadline']]

sub.to_csv("submission3.csv",index = False)
feature_list = list(X_train.columns)

# Get numerical feature importances 

importances = list(rf.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# Import matplotlib for plotting and use magic command for Jupyter Notebooks

import matplotlib.pyplot as plt

%matplotlib inline

# Set the style

plt.style.use('fivethirtyeight')

# list of x locations for plotting

x_values = list(range(len(importances)))

# Make a bar chart

plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42, max_depth=15,n_jobs=-1,verbose=1)

# Extract the two most important features



train_important = X_train.drop(['topic_microsoft','topic_palestine','Thursday','Sunday','Saturday','Monday','Friday','Year'],axis =1)

test_important = X_test.drop(['topic_microsoft','topic_palestine','Thursday','Sunday','Saturday','Monday','Friday','Year'],axis=1)

# Train the random forest

rf_most_important.fit(train_important, y_train)

# Make predictions and determine the error

predictions = rf_most_important.predict(test_important)
#error

mean_absolute_error(y_test, predictions)
# Import tools needed for visualization

from sklearn.tree import export_graphviz

import pydot

# Pull out one tree from the forest

tree = rf.estimators_[5]

# Import tools needed for visualization

from sklearn.tree import export_graphviz

import pydot

# Pull out one tree from the forest

tree = rf.estimators_[5]

print("debug1")

# Export the image to a dot file

export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph

print("debug2")

(graph, ) = pydot.graph_from_dot_file('tree.dot')

print("debug3")

# Write graph to a png file

graph.write_png('tree.png')


feature_list = list(Xh_train.columns)

# Get numerical feature importances 

importances = list(rfh.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# Import matplotlib for plotting and use magic command for Jupyter Notebooks

import matplotlib.pyplot as plt

%matplotlib inline

# Set the style

plt.style.use('fivethirtyeight')

# list of x locations for plotting

x_values = list(range(len(importances)))

# Make a bar chart

plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
rfh_most_important = RandomForestRegressor(n_estimators= 100, random_state=42,max_depth=15,n_jobs=-1,verbose=1)

# Extract the two most important features



train_important = Xh_train.drop(['topic_obama','topic_economy','topic_microsoft','topic_palestine','Tuesday','Thursday','Sunday','Saturday','Monday','Friday','Year'],axis =1)

test_important = Xh_test.drop(['topic_obama','topic_economy','topic_microsoft','topic_palestine','Tuesday','Thursday','Sunday','Saturday','Monday','Friday','Year'],axis=1)

# Train the random forest

rfh_most_important.fit(train_important, yh_train)

# Make predictions and determine the error

predictions_h = rfh_most_important.predict(test_important)

#Error

mean_absolute_error(yh_test, predictions_h)
dt_title = dt_title.drop(['topic_microsoft','topic_palestine','Thursday','Sunday','Saturday','Monday','Friday','Year'],axis =1)



dt_headline = dt_headline.drop(['topic_obama','topic_economy','topic_microsoft','topic_palestine','Tuesday','Thursday','Sunday','Saturday','Monday','Friday','Year'],axis =1)

dt['SentimentTitle'] = rf_most_important.predict(dt_title)

dt['SentimentHeadline'] = rfh_most_important.predict(dt_headline)
sub = dt[['IDLink','SentimentTitle','SentimentHeadline']]

sub.to_csv("submission4_rf_optimised.csv",index = False)
est_range = [100,200,500,1000,1500]

from sklearn.model_selection import validation_curve

train_scoreNum, test_scoreNum = validation_curve(

                                RandomForestRegressor(),

                                X = X_train, y = y_train, 

                                param_name = 'n_estimators',

                                param_range = est_range,

                                scoring="neg_mean_absolute_error", cv = 3)
import pandas as pd

nf = pd.read_csv("../input/hacky-zs1/News_Final.csv")
nf = nf.iloc[55946:,:]
nf = nf[nf['Headline'].notna()]
nf = nf.iloc[:,6:8]
#nf.reset_index(inplace = True) 

nf= nf.iloc[:,1:3]
df= pd.concat([test['IDLink'],nf],axis =1)
df.to_csv("submission1.csv",index= False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import plotly.express as ex

import plotly.graph_objs as go

from wordcloud import WordCloud,STOPWORDS

stopwords = list(STOPWORDS)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
t_data = pd.read_csv('/kaggle/input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')

t_data.head(3)
def remove_stop_words(sir):

    splited = sir.split(' ')

    splited = [word for word in splited if word not in stopwords]

    return ' '.join(splited)



t_data.Review = t_data.Review.apply(remove_stop_words)
sid = SentimentIntensityAnalyzer()



def get_char_count(sir):

    return len(sir)

def get_word_count(sir):

    return len(sir.split(' '))

def get_average_word_length(sir):

    aux = 0

    for word in sir.split(' '):

        aux += len(word)

    return aux/len(sir.split(' '))

def get_pos_sentiment(sir):

    sent = sid.polarity_scores(sir)

    return sent['pos']

def get_neg_sentiment(sir):

    sent = sid.polarity_scores(sir)

    return sent['neg']

def get_neu_sentiment(sir):

    sent = sid.polarity_scores(sir)

    return sent['neu']
t_data['Char_Count'] =  t_data.Review.apply(get_char_count)

t_data['Word_Count'] =  t_data.Review.apply(get_word_count)

t_data['Average_Word_Length'] =  t_data.Review.apply(get_average_word_length)

t_data['Positive_Sentiment'] =   t_data.Review.apply(get_pos_sentiment)

t_data['Negative_Sentiment'] = t_data.Review.apply(get_neg_sentiment)

t_data['Neutral_Sentiment'] =t_data.Review.apply(get_neu_sentiment)
word_list = ''

for word in t_data.Review:

    splited = word.lower()

    word_list +=splited

    

wordcloud = WordCloud(width=800,height=800,background_color='white',stopwords=stopwords,min_font_size=5).generate(word_list)

plt.figure(figsize = (25, 15), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
ex.box(t_data,x='Rating',y='Positive_Sentiment',notched=True,title='Rating Positive Sentiment Distributions')
ex.box(t_data,x='Rating',y='Negative_Sentiment',notched=True,title='Rating Positive Sentiment Distributions')
sns.pairplot(t_data)
sns.jointplot(x=t_data['Average_Word_Length'],y=t_data['Positive_Sentiment'],height=15,kind='kde',levels=20)
print('Average_Word_Length Skew: ',t_data['Average_Word_Length'].skew(),"  Average_Word_Length Kurtosis",t_data['Average_Word_Length'].kurt())
print('Average_Word_Length Mean: ',t_data['Average_Word_Length'].mean(),"  Average_Word_Length Median",t_data['Average_Word_Length'].median(),' Average_Word_Length Mode : ',t_data['Average_Word_Length'].mode()[0])
data_info = t_data.describe()

data_info.loc['skew'] = t_data.skew()

data_info.loc['kurt'] = t_data.kurt()

data_info
tout_l = t_data.copy()

tout_l['OLL'] = 'Normal'

tout_l.loc[tout_l[tout_l['Word_Count']>1000].index,'OLL']= 'Outlier'

tout_l.loc[tout_l[tout_l['Neutral_Sentiment']<0.25].index,'OLL']= 'Outlier'

tout_l.loc[tout_l[tout_l['Neutral_Sentiment']>0.98].index,'OLL']= 'Outlier'



ex.scatter_3d(tout_l,x='Rating',y='Neutral_Sentiment',z='Word_Count',color='OLL')
t_data = t_data[t_data['Neutral_Sentiment']>0.25]

t_data = t_data[t_data['Neutral_Sentiment']<0.98]

t_data = t_data[t_data['Word_Count']<1000]
cors = t_data.corr('pearson')

plt.figure(figsize=(20,13))

sns.heatmap(cors,annot=True,cmap='mako')
t_data.head(3)
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score 
train_x,test_x,train_y,test_y = train_test_split(t_data[['Positive_Sentiment','Negative_Sentiment','Average_Word_Length']],t_data['Rating'])



LR_Pipe = Pipeline(steps=[('model',LinearRegression())])

LR_Pipe.fit(train_x,train_y)

LR_predictions= LR_Pipe.predict(test_x)

LR_predictions = np.round(LR_predictions)

cfm = confusion_matrix(LR_predictions,test_y)



plt.figure(figsize=(20,13))

sns.heatmap(cfm,annot=True,cmap='mako',fmt='d')
print('accuracy: ',accuracy_score (LR_predictions,test_y))
DT_Pipe = Pipeline(steps=[('model',DecisionTreeRegressor())])

DT_Pipe.fit(train_x,train_y)

predictions= DT_Pipe.predict(test_x)

predictions = np.round(predictions*0.1 + LR_predictions*0.9)

cfm = confusion_matrix(predictions,test_y)



plt.figure(figsize=(20,13))

sns.heatmap(cfm,annot=True,cmap='mako',fmt='d')
print('accuracy: ',accuracy_score (predictions,test_y))
LR_Pipe.fit(t_data[['Positive_Sentiment','Negative_Sentiment','Average_Word_Length']],t_data['Rating'])

DT_Pipe.fit(t_data[['Positive_Sentiment','Negative_Sentiment','Average_Word_Length']],t_data['Rating'])

LR_predictions= LR_Pipe.predict(t_data[['Positive_Sentiment','Negative_Sentiment','Average_Word_Length']])

RF_predictions= DT_Pipe.predict(t_data[['Positive_Sentiment','Negative_Sentiment','Average_Word_Length']])

predictions = np.round(LR_predictions*0.9 + RF_predictions*0.1)



cfm = confusion_matrix(predictions,t_data['Rating'])



plt.figure(figsize=(20,13))

sns.heatmap(cfm,annot=True,cmap='mako',fmt='d')
print('accuracy: ',accuracy_score (predictions,t_data['Rating']))
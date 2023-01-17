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
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

tweet_test= pd.read_csv('../input/nlp-getting-started/test.csv')

all_tweets=pd.concat([tweet,tweet_test])

tweet.shape, tweet_test.shape,all_tweets.shape
tweet.head(3), tweet_test.head(3),all_tweets.head(3)
tweet.isnull().sum()
tweet.info(), tweet.shape
tweet.id.duplicated().any()
count_keywords=tweet[['keyword','text']].groupby('keyword').count()

count_keywords_df=pd.DataFrame(count_keywords).reset_index()

count_keywords_df.columns=["keyword","tweet_number"]



count_keywords_df.sort_values(by="tweet_number", ascending=False)
import altair as alt

chart = alt.Chart(count_keywords_df.head(50)).mark_bar(color="").encode(

    x='tweet_number',

    y=alt.Y('keyword',sort='-x')# y axis sorted by x

   

)# all text characteristics

text = chart.mark_text(

    align='left',

    baseline='middle',

    dx=3,

    color='black'

).encode(

    text='tweet_number'# the variable that is showing in the chart

).properties(

    title='Top number of tweets by keyword'

)



chart + text # both figures are showed overlapping

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorizer.fit(all_tweets.text)

train_vectorizer_features=vectorizer.transform(tweet.text)

test_vectorizer_features=vectorizer.transform(tweet_test.text)
target=tweet["target"]

target.head()
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model = log_model.fit(X=train_vectorizer_features, y=target)

y_pred = log_model.predict(test_vectorizer_features)
# predicting values tranform to dataframe

y_pred_df=pd.DataFrame(y_pred)
y_pred_df.head()
tweet_test.columns
from sklearn.metrics import f1_score

#f1_score(target, y_pred)

sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pred})

sub.to_csv('/kaggle/working/submission.csv',index=False)

sub.to_csv('submission.csv',index=False)
sub.head()
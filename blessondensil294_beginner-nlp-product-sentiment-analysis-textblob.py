# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os #import the os Packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install -U textblob

!python -m textblob.download_corpora
from textblob import TextBlob
#Negative sentiment score

a = TextBlob("I am the worst programmer ever")

a.sentiment
#Positive Sentiment score

a = TextBlob("I am the best programmer ever")

a.sentiment
#Neutral Sentiment Score

a = TextBlob("I am programmer")

a.sentiment
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load the Dataset to the DataFrame

df_Tweets = pd.read_csv('../input/twitter-product-sentiment-analysis/Twitter Product Sentiment Analysis.csv')
#Display the head of the DataFrame

df_Tweets.head()
print(df_Tweets['tweet'][2])

Tweet = TextBlob(df_Tweets['tweet'][2])

Tweet.sentiment
print(df_Tweets['tweet'][4])

Tweet = TextBlob(df_Tweets['tweet'][4])

Tweet.sentiment
print(df_Tweets['tweet'][5])

Tweet = TextBlob(df_Tweets['tweet'][5])

Tweet.sentiment
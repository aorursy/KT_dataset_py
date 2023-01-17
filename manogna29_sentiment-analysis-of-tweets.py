# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

tweets=pd.read_csv("../input/demonetization-tweets.csv",encoding = "ISO-8859-1")

tweets.head()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import *



from nltk import tokenize



sid = SentimentIntensityAnalyzer()



tweets['sentiment_compound_polarity']=tweets.text.apply(lambda x:sid.polarity_scores(x)['compound'])

tweets['sentiment_neutral']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neu'])

tweets['sentiment_negative']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neg'])

tweets['sentiment_pos']=tweets.text.apply(lambda x:sid.polarity_scores(x)['pos'])

tweets['sentiment_type']=''

tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'

tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'

tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'

tweets.head()
tweets.sentiment_type.value_counts().plot(kind='bar',title="sentiment analysis")
from ggplot import *

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (6.0, 6.0)



tweets['hour'] = pd.DatetimeIndex(tweets['created']).hour

tweets['date'] = pd.DatetimeIndex(tweets['created']).date

tweets['minute'] = pd.DatetimeIndex(tweets['created']).minute

df=(tweets.groupby('hour',as_index=False).sentiment_compound_polarity.mean())

print(ggplot(aes(x='hour',y='sentiment_compound_polarity'),data=df)+geom_line())
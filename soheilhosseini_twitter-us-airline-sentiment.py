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
tweets = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
tweets.head()
tweets.shape
tweets.info()
import seaborn as sns

import matplotlib.pyplot as plt
#Number of sentiments in each category



tweets.airline_sentiment.value_counts()
plt.figure(figsize=(3,5))

sns.countplot(tweets['airline_sentiment'], order =tweets.airline_sentiment.value_counts().index,palette= 'plasma')

plt.show()
g = sns.FacetGrid(tweets, col="airline", col_wrap=3, height=5, aspect =0.7)

g = g.map(sns.countplot, "airline_sentiment",order =tweets.airline_sentiment.value_counts().index, palette='plasma')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from lightgbm import LGBMClassifier

from sklearn.pipeline import make_pipeline

from warnings import filterwarnings

from sklearn.preprocessing import Normalizer

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import cross_validate



from sklearn.pipeline import Pipeline

twitter_sentiment = Pipeline([('CVec', (CountVectorizer(stop_words='english'))),

                     ('Tfidf', TfidfTransformer()),

                      ('norm', Normalizer()),

                     ('lgb', LGBMClassifier(n_jobs=-1))])

%%time

cv_pred = cross_validate(twitter_sentiment, 

                             tweets['text'], 

                             tweets['airline_sentiment'], 

                             cv=5,

                             scoring=('roc_auc_ovr'))
sorted(cv_pred.keys())
cv_pred['test_score']
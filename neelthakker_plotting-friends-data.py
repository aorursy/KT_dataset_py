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



%matplotlib inline
friends=pd.read_csv('/kaggle/input/friends-series-dataset/friends_episodes_v2.csv')
friends.head()
friends.describe()
friends.columns
friends.info()

from nltk.sentiment.vader import SentimentIntensityAnalyzer



s=SentimentIntensityAnalyzer()
friends['neg'] = friends['Summary'].apply(lambda x:s.polarity_scores(x)['neg'])

friends['neu'] = friends['Summary'].apply(lambda x:s.polarity_scores(x)['neu'])

friends['pos'] = friends['Summary'].apply(lambda x:s.polarity_scores(x)['pos'])

friends['compound'] = friends['Summary'].apply(lambda x:s.polarity_scores(x)['compound'])
friends.head()
friends.loc[friends['compound'] > 0.2, 'sent']= 'negative'

friends.loc[(friends['compound'] <= 0.2) & (friends['compound'] >= -0.2), 'sent'] = 'neutral'  

friends.loc[friends['compound'] < -0.2, 'sent']= 'positive'
friends
sns.countplot(x='Stars', data=friends, hue='sent')
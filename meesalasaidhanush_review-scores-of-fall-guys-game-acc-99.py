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
df=pd.read_csv(r'/kaggle/input/fall-guys-metacritic-reviews/fall_guys_metacritic_reviews.csv')
df.head()
df.shape
df.isnull().sum()
df['review_type'].unique()
df.drop(columns=['review_type','published_date','published_date'],inplace=True)
df.head()
df['score'].unique()
df.shape
df.isnull().any()
## we have to predict the score based on the text give so we dont need other columns
df.drop(columns=['username','votes','profile_url','platform'],inplace=True)
df.head()
df.describe()
df['score'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df['score'].value_counts().plot(kind='bar',color='r')
train_qs = pd.Series(df['review_text'].tolist()).astype(str)
from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
import nltk
from nltk.corpus import stopwords
import re
df.head(56)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score,classification_report
import matplotlib.pyplot as plt
%matplotlib inline
stopset=set(stopwords.words('english'))
vector=TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii',stop_words=stopset)
df.dropna(inplace=True)
vector.fit(df)
y=df.score
x=vector.fit_transform(df.review_text)
x.shape
from sklearn.tree import DecisionTreeClassifier
tr= DecisionTreeClassifier()
tr.fit(x,y)
pr=tr.predict(x)
pr
print(classification_report(y,pr))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,pr))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/7817_1.csv')

df.head(2)
from IPython.display import HTML

cat_hist = df.groupby('categories',as_index=False).count()

HTML(pd.DataFrame(cat_hist['categories']).to_html())
import nltk 

from nltk import word_tokenize

from nltk.corpus import stopwords

import re

import string

import matplotlib.pyplot as plt

from collections import Counter

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

def removePunctuation(x):

    x = x.lower()

    x = re.sub(r'[^\x00-\x7f]',r' ',x)

    return re.sub("["+string.punctuation+"]", " ", x)



stops = set(stopwords.words("english"))

def removeStopwords(x):

    filtered_words = [word for word in x.split() if word not in stops]

    return " ".join(filtered_words)

def removeAmzString(x):

    return re.sub(r'[0-9]+ people found this helpful\. Was this review helpful to you Yes No', "", x)
from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



reviews = [sent if type(sent)==str else "" for sent in df['reviews.title'].values]

reviews = [removeAmzString(sent) for sent in reviews]

#reviews = [removeStopwords(sent) for sent in reviews]

reviews = [removePunctuation(sent) for sent in reviews]



stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=200,

                      max_font_size=40,random_state=42).generate(str(reviews))

plt.figure(figsize=(15,20))

ax1 = plt.subplot2grid((4, 2), (0, 0))

ax2 = plt.subplot2grid((4, 2), (1, 0))

ax3 = plt.subplot2grid((4, 2), (0, 1), rowspan=2)

ax4 = plt.subplot2grid((4, 2), (2, 0), colspan=2,rowspan=2)



rat_hist = df.groupby('reviews.rating',as_index=False).count()

sns.barplot(x=rat_hist['reviews.rating'].values,y=rat_hist['id'].values,ax=ax1)



cat_hist = cat_hist.sort_values(by='id')

sns.barplot(x=cat_hist['categories'].index,y=cat_hist['id'].values,ax=ax3)



hf_hist = df.groupby('reviews.numHelpful',as_index=False).count()[0:30]

sns.barplot(x=hf_hist['reviews.numHelpful'].values.astype(int),y=hf_hist['id'].values,ax=ax2)



ax1.set_title("Reviews Ratings",fontsize=16)

ax3.set_title("Categories",fontsize=16)

ax2.set_title("Helpful Feedback",fontsize=16)

ax4.set_title("Words Cloud",fontsize=16)

ax4.imshow(wordcloud)

ax4.axis('off')

plt.show()
df['reviews.title'].head(5)
import nltk 

from nltk import word_tokenize

from nltk.corpus import stopwords

import re

import string

import matplotlib.pyplot as plt

from collections import Counter

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

def removePunctuation(x):

    x = x.lower()

    x = re.sub(r'[^\x00-\x7f]',r' ',x)

    return re.sub("["+string.punctuation+"]", " ", x)



stops = set(stopwords.words("english"))

def removeStopwords(x):

    filtered_words = [word for word in x.split() if word not in stops]

    return " ".join(filtered_words)

def removeAmzString(x):

    return re.sub(r'[0-9]+ people found this helpful\. Was this review helpful to you Yes No', "", x)
df['reviews.rating']=df['reviews.rating'].fillna(0)

target = df['reviews.rating']
cv = CountVectorizer(stop_words=None,token_pattern=r'\b\w\w+-?\w+\b',ngram_range=(1, 3))

fcv = cv.fit_transform(reviews)

print(fcv[1])

print(reviews[1])

len(df[df['reviews.rating'].isnull()==True])
rev_array = fcv.toarray()

X_train, X_test, y_train, y_test= train_test_split(rev_array, target, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100,criterion='entropy')

y_pred = clf.fit(X_train, y_train).predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
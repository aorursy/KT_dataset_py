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
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')

true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake.head()
true['label'] = 1

fake['label'] = 0

#fake.head()
df = pd.concat([true,fake]) #Merging the 2 datasets

df.head()
plt.figure(figsize=(10,10))

sns.heatmap(df.isnull(),cbar=False,cmap='YlGnBu')

plt.ioff()

f,ax=plt.subplots(1,2,figsize=(18,8))

df['label'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Type of News')

ax[0].set_ylabel('Count')

sns.countplot('label',data=df,ax=ax[1],order=df['label'].value_counts().index)

ax[1].set_title('Type of News')

ax[1].set_xlabel('Label')

plt.show()

from wordcloud import WordCloud

wrds1 = fake["title"].str.split("(").str[0].value_counts().keys()



wc1 = WordCloud(scale=5,max_words=1000,colormap="rainbow",background_color="black").generate(" ".join(wrds1))

plt.figure(figsize=(13,14))

plt.imshow(wc1,interpolation="bilinear")

plt.axis("off")

plt.title("Most Used Word In Fake News Title",color='b')

plt.show()
#X = df.drop('label',axis=1)

#X.head()
df.shape
label = df['label']

#y
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
df = df.dropna()

df.shape
messages = df.copy()
messages.reset_index(inplace=True)
messages['title'][0]
import re

import string

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

def message_cleaning(message):

    Test_punc_removed = [char for char in message if char not in string.punctuation ]

    Test_punc_removed_join=''.join(Test_punc_removed)

    Test_punc_removed_join_clean=[ word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]

    return Test_punc_removed_join_clean
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(analyzer=message_cleaning)

df_countvectorizer=vectorizer.fit_transform(df['title'])
from sklearn.naive_bayes import MultinomialNB

NB_classifier=MultinomialNB()

NB_classifier.fit(df_countvectorizer,label)
testing_sample=['fake is name of vegetable']

testing_sample_countvectorizer=vectorizer.transform(testing_sample)

test_predict=NB_classifier.predict(testing_sample_countvectorizer)

test_predict
X=df_countvectorizer

X.shape
label = df['label']

label.shape
X.shape 
label.shape
#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test,y_test=train_test_split(X,label,test_size=0.2)
#from sklearn.naive_bayes import MultinomialNB

#NB_classifier=MultinomialNB()

#NB_classifier.fit(X_train,y_train)
#from sklearn.metrics import classification_report,confusion_matrix

#y_predict_train=NB_classifier.predict(X_train)

#y_predict_train
#cm=confusion_matrix(y_train,y_predict_train)

#sns.heatmap(cm,annot=True)
"""y_predict_test=NB_classifier.predict(X_test)

y_predict_test

cm=confusion_matrix(y_test,y_predict_test)

sns.heatmap(cm,annot=True)"""
#print(classification_report(y_test,y_predict_test))
df_clean=df['title'].apply(message_cleaning)
df_clean
import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(0,len(messages)):

    review = re.sub('[^A-Zaz-]',' ',messages['title'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.word('english')]

    review = ' '.join(review)

    corpus.append(review)
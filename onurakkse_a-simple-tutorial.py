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
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import re
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.info()
train.shape
train.describe().T

train.head()
train[train["target"] == 1]["text"].values[1]
train[train["target"] == 0]["text"].values[0]
df = pd.concat([train,test])
# lower case

train['text'] = train['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))



# Punctional

train['text'] = train['text'].str.replace('[^\w\s]','')



# Removing URL

def remove_URL(text):    

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



train['text']=train['text'].apply(lambda x : remove_URL(x))



# Removing HTML Tags

def remove_html(text):    

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



train['text']= train['text'].apply(lambda x : remove_html(x))



# Removing Emojis

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)

train['text']= train['text'].apply(lambda x: remove_emoji(x))













delete = pd.Series(' '.join(train['text']).split()).value_counts()[-1000:]

train['text'] = train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))
train.head()
# StopWords





import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

sw = stopwords.words("english")

train["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

test.head()
x = train["text"]

y = train["target"]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

y_train = encoder.fit_transform(y_train)

y_test = encoder.fit_transform(y_test)
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(X_train)

test_vectors = count_vectorizer.transform(X_test)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf_model = rf.fit(train_vectors,y_train)
from sklearn import model_selection

accuracy = model_selection.cross_val_score(rf_model, 

                                           train_vectors, 

                                           y_train, 

                                           cv = 10).mean()



print("Count Vectors Doğruluk Oranı:", accuracy)
# Naive Bayes

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb_model = nb.fit(train_vectors, y_train)
accuracy = model_selection.cross_val_score(nb_model, 

                                           train_vectors, 

                                           y_train, 

                                           cv = 10).mean()



print("Count Vectors Doğruluk Oranı:", accuracy)
test = count_vectorizer.transform(test["text"])

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = nb_model.predict(test)

sample_submission.to_csv("submission.csv", index=False)

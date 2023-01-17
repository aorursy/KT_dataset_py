import numpy as np 

import pandas as pd 

import os

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import re
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

train_data.head()

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



train_data['text'] = train_data['text'].apply(lambda x : remove_URL(x))
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



train_data['text'] = train_data['text'].apply(lambda x : remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

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



train_data['text'] = train_data['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



#train_data['text'] = train_data['text'].apply(lambda x : remove_punct(x))
x = train_data["text"]

y = train_data["target"]



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



vect = CountVectorizer(stop_words = 'english')



x_train_cv = vect.fit_transform(X_train)

x_test_cv = vect.transform(X_test)





clf = MultinomialNB()

clf.fit(x_train_cv, y_train)



pred = clf.predict(x_test_cv)



accuracy_score(y_test,pred)
y_test = test_data["text"]

y_test_cv = vect.transform(y_test)

preds = clf.predict(y_test_cv)

submission["target"] = preds

submission.to_csv("submission.csv",index=False)
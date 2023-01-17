import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import missingno as msno

import matplotlib.pyplot as plt

import string

import re
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.info()
msno.matrix(train_df, figsize=(12,5))
plt.bar(train_df[train_df['target']==1]['target'].unique(), train_df[train_df['target']==1]['target'].value_counts(), label='Real')

plt.bar(train_df[train_df['target']==0]['target'].unique(), train_df[train_df['target']==0]['target'].value_counts(), label='Not')

plt.title("count of Real or Not Disater Tweets")

plt.ylabel("count")

plt.xticks(train_df['target'].unique(),('1','0'))

plt.legend()
train_df
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def clean_tweets(tweet):

    """Removes links and non-ASCII characters"""

    

    tweet = ''.join([x for x in tweet if x in string.printable])

    

    # Removing URLs

    tweet = re.sub(r"http\S+", "", tweet)

    

    return tweet



train_df['text']=train_df['text'].apply(clean_tweets)

test_df['text']=test_df['text'].apply(clean_tweets)
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

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



train_df['text']=train_df['text'].apply(remove_emoji)

test_df['text']=test_df['text'].apply(remove_emoji)
# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

def remove_punctuations(text):

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    

    for p in punctuations:

        text = text.replace(p, f' {p} ')



    text = text.replace('...', ' ... ')

    

    if '...' not in text:

        text = text.replace('..', ' ... ')

    

    return text



train_df['text']=train_df['text'].apply(remove_punctuations)

test_df['text']=test_df['text'].apply(remove_punctuations)
def remove_numbers(text):

    

    for number in string.digits:

        text = text.replace(number, '')



    return text



train_df['text']=train_df['text'].apply(remove_numbers)

test_df['text']=test_df['text'].apply(remove_numbers)
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False, header=True)
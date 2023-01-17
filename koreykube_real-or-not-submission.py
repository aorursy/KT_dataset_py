import numpy as np

import pandas as pd

import spacy

from array import *

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
nlp = spacy.load('en_core_web_lg')
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test_cleaned = []

train_cleaned = []



for i in range(len(train_df["text"].values)):

    tweet_text = train_df["text"].values[i]

    tweet_text = tweet_text.lower()

    train_cleaned.append(tweet_text)

    

for i in range(len(test_df["text"].values)):

    tweet_text = test_df["text"].values[i]

    tweet_text = tweet_text.lower()

    test_cleaned.append(tweet_text)
document = nlp.pipe(train_cleaned)

train_vector = np.array([tweet.vector for tweet in document])

print(train_vector.shape)
X = train_vector

Y = train_df["target"]



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, stratify=Y, test_size=0.1, random_state=0)



model = LogisticRegression(C=0.5)

model.fit(X_train, Y_train)



y_pred = model.predict(X_test)

print("Test accuracy : %0.2f" %(accuracy_score(Y_test, y_pred)*100))



y_train_pred = model.predict(X_train)

print("Train accuracy : %0.2f" %(accuracy_score(Y_train, y_train_pred)*100))
test_document = nlp.pipe(test_cleaned)

test_vector = np.array([tweet.vector for tweet in test_document])

test_output = model.predict(test_vector)
print(test_output)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = test_output

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
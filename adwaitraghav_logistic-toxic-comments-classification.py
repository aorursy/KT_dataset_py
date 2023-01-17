import nltk

import random

import pandas as pd

import re
df1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"])

df2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"])



# Merging the training csv files

train = pd.concat([df1, df2], axis = 0, sort = False).reset_index(drop=True)



test_translated = pd.read_csv("../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv")

test = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")

valid = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

submission = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
train.head()
test.head()
test_translated.head()
submission.head()
train_shape = train.shape[0]

test_shape = test.shape[0]

sum = train_shape + test_shape
print("    :    train    :    test")

print("rows:    {} :      {}".format(train_shape, test_shape))

print("perc:    {} :      {}".format(train_shape*100/sum,test_shape*100/sum))
targets = ["toxic"]
def cleaning(sen):

    sen = re.sub(r"what's","what is",sen)

    sen = re.sub(r"\'s'"," ",sen)

    sen = re.sub(r"\'ve'"," have ",sen)

    sen = re.sub(r"can't","cannot",sen)

    sen = re.sub(r"n't"," not ",sen)

    sen = re.sub(r"i'm","i am ",sen)

    sen = re.sub(r"\'re'"," are ",sen)

    sen = re.sub(r"\'d"," would ",sen)

    sen = re.sub(r"\'ll","will",sen)

    sen = re.sub(r"\'scuse", "excuse",sen)

    sen = re.sub("\W"," ",sen)

    sen = re.sub(r"\s+"," ",sen)

    sen = sen.strip(' ')

    return sen
cleaned_training_data = []

for i in range(len(train)):

    cleaned_comment = cleaning(train["comment_text"][i])

    cleaned_training_data.append(cleaned_comment)

train["comment_text"] = pd.Series(cleaned_training_data).astype(str)
X = train.comment_text

y = train["toxic"]

X_translated = test_translated.translated

X_multi = test.content
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(max_features=2000, min_df=2)
X_train_dtm = vector.fit_transform(X_train)

X_test_dtm = vector.fit_transform(X_test)

X_translated_dtm = vector.fit_transform(X_translated)

X_multi_dtm = vector.fit_transform(X_multi)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

model_log = LogisticRegression(C=6.0)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_train = y_train.astype('int')

y_test = y_test.astype('int')
training_prob = pd.DataFrame()

for label in targets:

    print("...On the label:{}".format(label))

    model_log.fit(X_train_dtm, y_train)

    predictions = model_log.predict(X_train_dtm)

    print("Training accuracy: {}".format(accuracy_score(y_train, predictions)))

    probability_train = model_log.predict_proba(X_train_dtm)[:,1]

    print(classification_report(y_train, predictions))

    print()

    print(confusion_matrix(y_train, predictions))

    training_prob[label] = probability_train
training_prob.head()
testing_prob = pd.DataFrame()
for label in targets:

    print("...On the label:{}".format(label))

    predictions = model_log.predict(X_test_dtm)

    print("Testing accuracy: {}".format(accuracy_score(y_test, predictions)))

    probability_test = model_log.predict_proba(X_test_dtm)[:,1]

    print(classification_report(y_test, predictions))

    print()

    print(confusion_matrix(y_test, predictions))

    testing_prob[label] = probability_test
testing_prob.head()
multi_prob = submission
for label in targets:

    predictions = model_log.predict(X_multi_dtm)

    probability_multi = model_log.predict_proba(X_multi_dtm)[:,1]

    multi_prob[label] = probability_multi
multi_prob.head()
trans_prob = submission
for label in targets:

    predictions = model_log.predict(X_translated_dtm)

    probability_trans = model_log.predict_proba(X_translated_dtm)[:,1]

    trans_prob[label] = probability_trans
trans_prob.head()
multi_prob.to_csv('submission.csv', index=False)
import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train.head()
train['len1'] = train['text'].apply(lambda x:len(str(x).split(' ')))

train['len2'] = train['selected_text'].apply(lambda x:len(str(x).split(' ')))



train.head()
train['len1'].max()
submission.head()
test.head()
train['sentiment'] = train['sentiment'].apply(lambda x: 2 if x == 'positive' else 1 if x == 'neutral' else 0)

train.head()
selected_texts = train['selected_text'].astype(str)

all_train_texts = train['text'].astype(str)

text_locations = [all_train_texts[i].find(s) for i, s in enumerate(selected_texts)]
text_locations[:5]
train['text_location'] = text_locations

train.head()
test['sentiment'] = test['sentiment'].apply(lambda x: 2 if x == 'positive' else 1 if x == 'neutral' else 0)

test['len1'] = test['text'].apply(lambda x:len(str(x).split(' ')))

test.head()
# to predict 'len2'

Y_train1 = train['len2']

X_train1 = train[['sentiment', 'len1']]

X_test = test[['sentiment', 'len1']]



# to predict 'text_location'

Y_train2 = train['text_location']

X_train2 = train[['sentiment', 'len1']]
Y_train1.head()
X_train1.head()
X_test.head()
print("The mean length of full text in the training data is " + str(round(train['len1'].mean(), 2)))
print("The median length of full text in the training data is " + str(round(train['len1'].median(), 2)))
print("The most common length of full text in the training data is " + str(round(train['len1'].mode()[0], 2)))
print("The mean length of full text in the test data is " + str(round(test['len1'].mean(), 2)))
print("The median length of full text in the test data is " + str(round(test['len1'].median(), 2)))
print("The most common length of full text in the test data is " + str(round(test['len1'].mode()[0], 2)))
print("The mean length of selected text in the training data is " + str(round(train['len2'].mean(), 2)))
print("The mean length of selected text in the training data is " + str(round(train['len2'].median(), 2)))
print("The mean length of selected text in the training data is " + str(round(train['len2'].mode()[0], 2)))
from sklearn import linear_model

import lightgbm as lgb
reg = lgb.LGBMRegressor()

#reg = linear_model.LinearRegression()

reg.fit(X_train1, Y_train1)
predicted1 = np.round(reg.predict(X_test))

predicted1[predicted1 < 1] = 1

predicted1
reg2 = lgb.LGBMRegressor()

#reg2 = linear_model.LinearRegression()

reg2.fit(X_train2, Y_train2)
predicted2 = np.round(reg2.predict(X_test))

predicted2[predicted2 < 1] = 1

predicted2
# now predctions are of the form: index of starting character + length of word

predicted = predicted1 + predicted2

predicted
sub = test[['textID', 'text']]

sub['preds'] = predicted

sub.head()
sub['text2'] = sub["text"].apply(lambda x: x.split())

sub
text2 = sub['text2']

text2
textx = sub['text'].tolist()

text_sub = [s[int(predicted2.tolist()[ind]):int(predicted2.tolist()[ind])+int(predicted1.tolist()[ind])] for ind, s in enumerate(textx)]
text_sub[:5]
text2 = [l[-int(predicted.tolist()[ind]):] for ind, l in enumerate(text2)]
text2[:5]
sub['text22'] = text2

sub.head()
sub['result'] = sub["text22"].apply(lambda x: " ".join(x))
sub.head()
submission["selected_text"] = sub['result']
submission.head()
submission.to_csv('submission.csv', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





train_tweet = pd.read_csv('../input/nlp-getting-started/train.csv')

test_tweet = pd.read_csv('../input/nlp-getting-started/test.csv')

submit_result = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
print("show top 10 train tweet data size: ", train_tweet.shape)

train_tweet.head(10)
print("show top 10 test tweet data size: ", test_tweet.shape)

test_tweet.head(10)
real_train_data = train_tweet[train_tweet['target'] == 1]

false_train_data = train_tweet[train_tweet['target'] == 0]

real_train_data.head(10)
false_train_data.head(10)
plt.bar(0, bottom=1, width=real_train_data.shape[0], height=0.5, orientation='horizontal', label='Real')

plt.bar(0, bottom=2, width=false_train_data.shape[0], height=0.5, orientation='horizontal', label='False')

plt.legend()

plt.xlabel('Number of examples')

plt.title('Propertion of examples')
def length(text):

    return len(text)
real_train_data['length'] = real_train_data['text'].apply(length)

false_train_data['length'] = false_train_data['text'].apply(length)
plt.rcParams['figure.figsize'] = (18.0, 6.0)

bins = 150

plt.hist(real_train_data['length'], alpha=0.6, bins=bins, label="TRUE")

plt.hist(false_train_data['length'], alpha=0.6, bins=bins, label="FALSE")

plt.xlabel('length')

plt.ylabel('number')

plt.xlim(0, 150)

plt.grid()

plt.legend(loc=0)

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

real_train_data_len = real_train_data['text'].str.len()

ax1.hist(real_train_data_len, color='green')

ax1.grid()

ax1.set_title('true tweets')



false_train_data_len = false_train_data['text'].str.len()

ax2.hist(false_train_data_len, color='red')

ax2.grid()

ax2.set_title('false tweets')



fig.suptitle('Charactres in tweets')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

real_train_tweet_len = real_train_data['text'].str.split().map(lambda x: len(x))

ax1.grid()

ax1.set_title('true tweets')

ax1.hist(real_train_tweet_len, color='green')



false_train_tweet_len = false_train_data['text'].str.split().map(lambda x: len(x))

ax2.grid()

ax2.set_title('false tweets')

ax2.hist(false_train_tweet_len, color='red')



fig.suptitle("Words in a tweet")
import spacy



nlp = spacy.blank("en")

textcat = nlp.create_pipe(

                "textcat",

                config={

                    "exclusive_classes": True,

                    "architecture": "bow"})

nlp.add_pipe(textcat)
textcat.add_label("True")

textcat.add_label("False")
train_tweet_data = train_tweet['text'].values

train_tweet_labels = [{'cats': {'True': target == 1,

                                'False': target == 0}}

                     for target in train_tweet['target']]
train_data = list(zip(train_tweet_data, train_tweet_labels))

train_data[100:120]
from spacy.util import minibatch

import random



def train(model, train_data, optimizer):

    losses = {}

    random.seed(1)

    random.shuffle(train_data)

    

    batches = minibatch(train_data, size=100)

    for batch in batches:

        texts, labels = zip(*batch)

        model.update(texts, labels, sgd=optimizer, losses=losses)

    

    return losses
spacy.util.fix_random_seed(1)

random.seed(1)



optimizer = nlp.begin_training()

losses = train(nlp, train_data, optimizer)

print(losses['textcat'])
def predict(model, texts):

    docs = [model.tokenizer(text) for text in texts]

    

    textcat = model.get_pipe('textcat')

    scores, _ = textcat.predict(docs)

    

    predicted_class = scores.argmax(axis=1)

    

    return predicted_class
print(test_tweet['text'].values)

# texts = [test_tweet['text'].values]

predictions = predict(nlp, test_tweet['text'].values)



for p, t in zip(predictions, test_tweet['text'].values):

    print(f"{textcat.labels[p]}: {t} \n")
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

print(submission.head(10))
negate_predictions = []

for item in predictions:

    if item == 0:

        negate_predictions.append(1)

    if item == 1:

        negate_predictions.append(0)

print(negate_predictions[0:10])
submission['target'] = negate_predictions

print(submission.head(10))
submission.to_csv("submission.csv", index=False, header=True)
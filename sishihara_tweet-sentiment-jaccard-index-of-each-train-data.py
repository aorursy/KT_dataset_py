import numpy as np

import pandas as pd
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
# Because train.text contains NaN

# https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/138213

train.dropna(subset=['text'], inplace=True)
train.head()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
score = []

for i in range(train.shape[0]):

    score.append(jaccard(train["selected_text"].values[i], train["text"].values[i]))

train['jaccard'] = score
train['jaccard'].hist()
train.sort_values('jaccard')
train.sort_values('jaccard').to_csv('train_with_jaccard.csv')
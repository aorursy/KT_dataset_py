import os

import numpy as np 

import pandas as pd



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
INPUT = "/kaggle/input/tweet-sentiment-extraction"



train = pd.read_csv(f"{INPUT}/train.csv")

test = pd.read_csv(f"{INPUT}/test.csv")

sub = pd.read_csv(f"{INPUT}/sample_submission.csv")



train.head()
corpus = train["text"].dropna().tolist()

cv = CountVectorizer(ngram_range=(1, 3), stop_words="english", min_df=5, max_df=0.8)

bow = cv.fit_transform(corpus).toarray()

bow_df = pd.DataFrame(bow, columns=cv.get_feature_names())
sentiment_dic = {v:i for i,v in enumerate(train["sentiment"].unique())}

y = train[~train["text"].isna()]["sentiment"].map(sentiment_dic).values
lr_model = LogisticRegression()

lr_model.fit(bow, y)
coef_df = pd.DataFrame(lr_model.coef_, columns=cv.get_feature_names())

coef_df
X_test = cv.transform(test["text"].tolist()).toarray()

X_test_df = pd.DataFrame(X_test, columns=cv.get_feature_names())
key_words = []

for idx in range(len(X_test_df)):

    _df = X_test_df.iloc[idx]

    _df = _df[_df != 0]

    word_coef = coef_df[_df.index].iloc[test["sentiment"].map(sentiment_dic).iloc[idx]]

    

    test_text = test.iloc[idx]["text"]

    if len(test_text.split()) <= 3:

        key_words.append(test_text)

        continue

    try:

        kw = word_coef[word_coef == word_coef.max()].index[0]

    except IndexError:

        key_words.append(test_text)

        continue

    key_words.append(kw)
sub["selected_text"] = key_words

sub.to_csv("submission.csv", index=None)

sub.head()
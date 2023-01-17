import pandas as pd

import numpy as np

from tqdm import tqdm

import re



tqdm.pandas()



import ftfy

ftfy.fix_text('Kaggle is a cool placee &lt;3')
original_df = pd.read_csv("https://raw.githubusercontent.com/Galanopoulog/DATA607-Project-4/master/TextEmotion.csv")

original_df.head()
tweet = "sooo sad i will miss you here in san diego!!!"

original_df[original_df['content'].str.lower().str.contains(tweet)]
len(original_df['sentiment'].unique())
list(original_df['sentiment'].unique())
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15,5)



%config InlineBackend.figure_format='retina'



title = "Sentiment distribution in original_df"

original_df.groupby('sentiment')['content'].count().plot.bar(color='orange', title=title);
len(original_df['author'].unique())
original_df['author'].value_counts()
TSE_DATA = "/kaggle/input/tweet-sentiment-extraction/"



train_df = pd.read_csv(TSE_DATA + "train.csv").dropna().reset_index(drop=True)

test_df = pd.read_csv(TSE_DATA + "test.csv")
size_private_df = original_df.shape[0] - train_df.shape[0] - test_df.shape[0]

size_private_df
test_df.shape[0] / 30 * 70
train_test_tweets = list(train_df['text'].str.lower()) + list(test_df['text'].str.lower())



def tweet_in_private(content):

    for tweet in train_test_tweets:

        if tweet in content:

            return False

    return True



original_df['content'] = original_df['content'].str.lower()

original_df['in_private'] = original_df['content'].progress_apply(tweet_in_private)
original_df['in_private'].value_counts()
private_df = original_df[original_df['in_private'] == True]

private_df.head()
private_df.shape
title = "Sentiment distribution in private_df"

private_df.groupby('sentiment')['content'].count().plot.bar(color='orange', title=title);
private_df['author'].value_counts()
private_df.to_csv("test_private_df.csv", index=False)
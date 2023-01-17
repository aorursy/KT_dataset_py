import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100
import markovify as mk

tweets = pd.read_csv('../input/tweets.csv', usecols = ['handle', 'text', 'is_retweet'])
tweets = tweets[tweets.is_retweet == False]
tweets.sample(8)
def tweet(tweeter):
    doc = tweets[tweets.handle.str.contains(tweeter)].text.tolist()
    text_model = mk.Text(doc) 
    print('\n', tweeter)
    for i in range(8):
        print(text_model.make_short_sentence(140))
        
tweet('Hillary')
tweet('Donald')
def subj_tweet(tweeter, subject):
    doc = tweets[tweets.handle.str.contains(tweeter)].text.tolist()
    text_model = mk.Text(doc) 
    print('\n', tweeter)
    for i in range(8):
        print(text_model.make_sentence_with_start(subject, strict=False))

subj_tweet('Hillary', 'They')
subj_tweet('Donald', 'We')
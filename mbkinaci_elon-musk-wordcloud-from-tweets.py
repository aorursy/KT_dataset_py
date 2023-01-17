#Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
dataset = pd.read_csv("../input/elonmusk_tweets.csv")

#Converting tweets into lists by split()

dataset['new_text'] = dataset['text'].apply(lambda x: x.split() )

#Removing first element of all lists,which compose of tweet words

dataset['new_text'] = dataset['new_text'].apply(lambda x: x[1:])

#Converting back to string

dataset['last'] = dataset['new_text'].apply(lambda x: " ".join(str(y) for y in x))

#Adding https" to stopwords

stopwords = set(STOPWORDS)

stopwods = stopwords.add("https")

#Converting to a big string

text = ' '.join(dataset['last'])
wc = WordCloud(background_color="black", max_words=2000,

               stopwords=stopwords)

# generate word cloud

wc.generate(text)
# show

plt.figure(1,figsize=(15, 15))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
#TeslaMotors and SpaceX are outstanding . 
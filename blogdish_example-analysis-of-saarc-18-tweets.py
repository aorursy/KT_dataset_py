import pandas as pd
tweetstable = pd.read_csv("../input/saarc87964.csv", encoding = "Latin-1")
tweetstable[0:5]
len(tweetstable.Handle)
handles = []

for handle in tweetstable.Handle:

    handles.append(handle)
import nltk
hfdist = nltk.FreqDist(handles)
hfdist.most_common(1000)
len(tweetstable.Tweet)
tfdist = nltk.FreqDist(x for x in tweetstable.Tweet)
tfdist.most_common(1000)
dfdist = nltk.FreqDist(x for x in tweetstable.Day)
dfdist.most_common(1000)
import matplotlib.pyplot as plt
%matplotlib inline
hfdist.plot(25)
dfdist.plot(25)
hashes = []

for tweet in tweetstable.Tweet:

    tokens = str(tweet).split(" ")

    for tok in tokens:

        if "#" in tok:

            hashes.append(tok)
hashdist = nltk.FreqDist(hash.lower() for hash in hashes)
hashdist.most_common(1000)
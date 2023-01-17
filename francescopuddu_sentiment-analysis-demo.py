import numpy as np
import pandas as pd 
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

dataset_filename = os.listdir("../input")[0]
dataset_path = os.path.join("..","input",dataset_filename)
df = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', encoding ="ISO-8859-1" , names=["target", "ids", "date", "flag", "user", "text"])
df = df[['target','text']]
df['target'] = df['target'].replace(4,1)
text, sentiment = list(df['text']), list(df['target'])
for t in text:
    t = t.lower()
data_neg = text[:800000]
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)

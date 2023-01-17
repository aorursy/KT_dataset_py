import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS 



import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
print(tf.__version__)
specific_wc = []

sw = list(set(stopwords.words('english')))

sw = sw + specific_wc



print(sw[:5])

print(len(sw))
df = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv", 

                 encoding = 'latin-1', header=None)

df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

df.head()
df.info()
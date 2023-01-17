print(' Problem:QIQC')
import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
import collections
import sklearn as sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.feature_extraction import text 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import gensim
from collections import defaultdict
from itertools import islice
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from keras import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D,Dense, Input, Embedding, Dropout, LSTM, CuDNNGRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


import time
%matplotlib inline
warnings.filterwarnings('ignore')
print('Libraries loaded')
!ls ../input/
!ls ../input/embeddings/

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train:",train_df.shape)
print("Test:", test_df.shape)
train_df.head(20)
train_df.dtypes
# To generate a Wordcloud of the Question text in the given data file
txtCorpus = " ".join(train_df.question_text)

wordCloud = WordCloud(width=1024, height=1024, margin=0).generate(txtCorpus)

# To display the image generated
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordCloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()

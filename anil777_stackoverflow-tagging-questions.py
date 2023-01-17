import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

import matplotlib.lines as mlines

import seaborn as sns



import warnings



import pickle

import time



import re

from bs4 import BeautifulSoup

import nltk

from nltk.tokenize import ToktokTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

from string import punctuation



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.dummy import DummyClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection

from sklearn.metrics import make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import hamming_loss

from sklearn.cluster import KMeans





import logging



from scipy.sparse import hstack



warnings.filterwarnings("ignore")

plt.style.use('bmh')

%matplotlib inline
np.random.seed(seed=11)
import os 

print(os.listdir("../input"))
df = pd.read_csv("../input/Questions.csv", encoding="ISO-8859-1")

df.head()
tags = pd.read_csv("../input/Tags.csv", encoding="ISO-8859-1", dtype={'Tag': str})

tags.head()
df.info()
tags.info()
tags['Tag'] = tags['Tag'].astype(str)
grouped_tags = tags.groupby("Id")["Tag"].apply(lambda tags: ' '.join(tags))
grouped_tags.head()
grouped_tags.reset_index()
grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags.values})

grouped_tags_final.head(5)

df.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
df = df.merge(grouped_tags_final, on='Id')

df.head(5)

new_df = df[df['Score']>5]
new_df.head()
new_df.isnull().sum()

# No missing value present 
print('Dupplicate entries: {}'.format(new_df.duplicated().sum()))

# No Duplicates Present 
new_df.drop(columns=['Id', 'Score'], inplace=True)
new_df.head()
new_df['Tags'] = new_df['Tags'].apply(lambda x: x.split())
new_df['Tags']
all_tags = [item for sublist in new_df['Tags'].values for item in sublist]
len(all_tags)
my_set = set(all_tags)

unique_tags = list(my_set)

len(unique_tags)
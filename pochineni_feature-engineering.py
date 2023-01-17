import os

import numpy as np



import pandas as pd

from tqdm import tqdm

tqdm.pandas()



from nltk import word_tokenize, pos_tag

from collections import Counter



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import required packages

#basics

import pandas as pd 

import numpy as np

from tqdm import tqdm

tqdm.pandas()



#misc

import gc

import time

import warnings



#stats

from scipy.misc import imread

from scipy import sparse

import scipy.stats as ss



#viz

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec 

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import matplotlib_venn as venn



#nlp

import string

import re    #for regex

import nltk

from nltk.corpus import stopwords

import spacy

from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

# Tweet tokenizer does not split at apostophes which is what we want

from nltk.tokenize import TweetTokenizer   



#FeatureEngineering

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split





#settings

start_time=time.time()

color = sns.color_palette()

sns.set_style("dark")

eng_stopwords = set(stopwords.words("english"))

warnings.filterwarnings("ignore")



lem = WordNetLemmatizer()

tokenizer=TweetTokenizer()



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
print("Check for missing values in Train dataset")

null_check=train.isnull().sum()

print(null_check)

print("Check for missing values in Test dataset")

null_check=test.isnull().sum()

print(null_check)

print("filling NA with \"unknown\"")

train["comment_text"].fillna("unknown", inplace=True)

test["comment_text"].fillna("unknown", inplace=True)
# Generate new Features using train data

train['total_length'] = train['comment_text'].apply(len)

train['capitals'] = train['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

train['caps_vs_length'] = train.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)

train['num_exclamation_marks'] = train['comment_text'].apply(lambda comment: comment.count('!'))

train['num_question_marks'] = train['comment_text'].apply(lambda comment: comment.count('?'))

train['num_punctuation'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))

train['num_symbols'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))

train['num_words'] = train['comment_text'].apply(lambda comment: len(comment.split()))

train['num_unique_words'] = train['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))

train['words_vs_unique'] = train['num_unique_words'] / train['num_words']

train['num_smilies'] = train['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

#Sentence count in each comment:

train['count_sent']=train['comment_text'].apply(lambda x: len(re.findall("\n",str(x)))+1)

#title case words count

train['count_words_title'] = train['comment_text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords

train['count_stopwords'] = train['comment_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words

train['mean_word_len'] = train['comment_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
features = ('total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks','num_question_marks', 'num_punctuation', 'num_words', 

            'num_unique_words','words_vs_unique', 'num_smilies', 'num_symbols', 'count_sent', 'count_words_title', 'count_stopwords', 'mean_word_len')

columns = ('target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'funny', 'wow', 'sad', 'likes', 'disagree', 'sexual_explicit','identity_annotator_count', 'toxicity_annotator_count')

rows = [{c:train[f].corr(train[c]) for c in 'columns'} for f in features]

train_correlations = pd.DataFrame(rows, index=features)
#Correlation between new features and target variable

train_correlations
#Correlation using heat map

plt.figure(figsize=(10, 6))

sns.set(font_scale=1)

ax = sns.heatmap(train_correlations, vmin=-0.1, vmax=0.1, center=0.0)
#Add term document matrix, POS and named entity recognition as features



cv = CountVectorizer()

count_feats_user= cv.fit_transform(train['comment_text'].apply(lambda x : str(x)))
count_feats_user.head()
## POS



tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(train_data) + list(test_data))
pos_tag(tokenizer.fit_on_texts)
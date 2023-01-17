!pip install pyenchant pysastrawi
!wget http://archive.ubuntu.com/ubuntu/pool/main/libr/libreoffice-dictionaries/hunspell-id_6.4.3-1_all.deb

!dpkg -i hunspell-id_6.4.3-1_all.deb
!apt update && apt install -y enchant libenchant1c2a hunspell hunspell-en-us libhunspell-1.6-0
import re

import os

import gc

import random



import numpy as np

import pandas as pd

import sklearn

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import enchant
!pip freeze > requirements.txt
print('Numpy version:', np.__version__)

print('Pandas version:', pd.__version__)

print('Scikit-Learn version:', sklearn.__version__)

print('Matplotlib version:', matplotlib.__version__)

print('Seaborn version:', sns.__version__)

print('NLTK version:', nltk.__version__)
SEED = 42



os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)
nltk.download('wordnet')
!ls -lha /kaggle/input

!ls -lha /kaggle/input/student-shopee-code-league-sentiment-analysis
df_train = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/train.csv')

df_train.sample(10)
df_train2 = pd.read_csv('/kaggle/input/shopee-reviews/shopee_reviews.csv')



def to_int(r):

    try:

        return np.int32(r)

    except:

        return np.nan



df_train2['label'] = df_train2['label'].apply(to_int)

df_train2 = df_train2.dropna()

df_train2['label'] = df_train2['label'].astype(np.int32)

df_train2
df_test = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv')

df_test.sample(10)
X_train = pd.concat([df_train['review'], df_train2['text']], axis=0)

X_train = X_train.reset_index(drop=True)

y_train = pd.concat([df_train['rating'], df_train2['label']], axis=0)

y_train = y_train.reset_index(drop=True)



X_test = df_test['review']
rating_count = y_train.value_counts().sort_index().to_list()

total_rating = sum(rating_count)

lowest_rating_count = min(rating_count)

rating_weight = [lowest_rating_count/rc for rc in rating_count]



print(rating_count)

print(total_rating)

print(rating_weight)
class_weight = np.empty((total_rating,))

for i in range(total_rating):

    class_weight[i] = rating_weight[y_train[i] - 1]
from nltk.stem import WordNetLemmatizer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory



lemmatizer = WordNetLemmatizer() # for en

factory = StemmerFactory() # for id

stemmer = factory.create_stemmer() # for id



tweet_tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)



eng_dict = enchant.Dict('en')

ind_dict = enchant.Dict('id_ID')



def remove_char(text):

    text = re.sub(r'[^a-z ]', ' ', text)

    return text





def stem_lemma(tokens):

    new_token = []

    for token in tokens:

        if eng_dict.check(token):

            new_token.append(lemmatizer.lemmatize(token))

        elif ind_dict.check(token):

            new_token.append(stemmer.stem(token))

        else:

            new_token.append(token)

    return new_token



def upper_or_lower(tokens):

    new_token = []

    for token in tokens:

        total_lower = len(re.findall(r'[a-z]',token))

        total_upper = len(re.findall(r'[A-Z]',token))

        if total_lower == 0 or total_upper == 0:

            new_token.append(token)

        elif total_lower > total_upper:

            new_token.append(token.lower())

        else:

            new_token.append(token.upper())

    return new_token

    



def preprocess(X):

    X = X.apply(tweet_tokenizer.tokenize)

    X = X.apply(lambda token: [t for t in token if t != ''])

    X = X.apply(upper_or_lower)

    X = X.apply(stem_lemma)

#     X = X.apply(lambda token: ' '.join(token)) # need to join token because sklearn tf-idf only accept string, not list of string

    

#     X = X.apply(remove_char)

    return X
X_train = preprocess(X_train)

X_test = preprocess(X_test)
X_train.sample(10)
X_train = pd.DataFrame({'X': X_train})

X_train.to_parquet('X_train.parquet', engine='pyarrow')
X_test = pd.DataFrame({'X': X_test})

X_test.to_parquet('X_test.parquet', engine='pyarrow')
y_train = pd.DataFrame({'y': y_train})

y_train.to_parquet('y_train.parquet', engine='pyarrow')
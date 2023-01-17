# Data Analysis
import pandas as pd
import numpy as np
import math

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Data Visualization for text
from PIL import Image
from os import path
import os
import random
from wordcloud import WordCloud, STOPWORDS

# Text Processing
import re
import itertools
import spacy
import string
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
from collections import Counter

# Ignore noise warning
import warnings
warnings.filterwarnings('ignore')

# Work with pickles
import pickle

pd.set_option('display.max_column', None)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#mbti_df = pd.read_csv("../data/mbti_1.csv")
mbti_df = pd.read_csv('../input/mbti-type/mbti_1.csv')
mbti_df.head()
mbti_df.shape
mbti_df.info()
mbti_df.isna().sum()
mbti_df.duplicated().sum()
mbti_df.nunique()
mbti_df.type.value_counts()
plt.figure(figsize=(18,10))
sns.countplot(y='type',data=mbti_df, order=mbti_df.type.value_counts().index)
sns.set_context('talk')
plt.title('Personality types distribution', fontsize=25)
#plt.savefig('images/output_images/mbti_count.png')
plt.show()
def var_row(row):
    lst = []
    for word in row.split('|||'):
        lst.append(len(word.split()))
    return np.var(lst)

mbti_df['words_per_comment'] = mbti_df['posts'].apply(lambda x: len(x.split())/50)
mbti_df['variance_of_word_counts'] = mbti_df['posts'].apply(lambda x: var_row(x))
mbti_df.head()
plt.figure(figsize=(18,10))
sns.swarmplot('type', 'words_per_comment', data=mbti_df)
sns.set_context('talk')
plt.title('Posts length per type', fontsize=25)
#plt.savefig('images/output_images/mbti_posts_length.png')
plt.show()
mbti_df.describe().T
mbti_df.corr()
# Read the whole text.
text = ' '.join(mbti_df['posts'])

# Generate a word cloud image
stopwords = STOPWORDS
wordcloud = WordCloud(background_color='white', width=800, height=400, stopwords=stopwords, max_words=100, repeat=False, min_word_length=4).generate(text)

# Display the generated image:
plt.figure(figsize=(18,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
sns.set_context('talk')
plt.title('Most common words', fontsize=25)
#plt.savefig('images/output_images/mbti_cloud.png')
plt.show()
mbti_text = mbti_df[['type','posts']].copy()
mbti_text = mbti_text.fillna('')
text_columns = mbti_text[['type']]
text_columns['text'] = mbti_text.iloc[:,1:].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
def clean_urls(column):
    '''
    This function takes a string and returns a string 
    with its urls removed and all the words in lowercase.
    '''
    return column.apply(lambda x: x.lower()).apply(lambda x: re.sub('http[s]?://\S+', '', x))

text_columns['text'] = clean_urls(text_columns['text'])
#raise SystemExit('Stop right there! Run cells one by one till the next heading.')
nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser']) 
nlp.max_length = 33000000
def tokenize(string):
    '''
    This function takes a sentence and returns the list of all lemma
    '''
    doc = nlp(string)
    l_token = [token.text for token in doc if not token.is_punct 
               | token.is_space | token.is_stop | token.is_digit & token.is_oov]
    return ' '.join(l_token)


text_columns['text'] = text_columns['text'].apply(lambda row: tokenize(row))
pd_token = pd.DataFrame(text_columns, columns=['type', 'text'])
pd_token.head()
#pd_token.to_pickle('data/output_pickles/token.pkl')
# Read the whole text.
text = ' '.join(pd_token['text'])

# Generate a word cloud image
stopwords = STOPWORDS
wordcloud = WordCloud(background_color='white', width=800, height=400, stopwords=stopwords, max_words=100, repeat=False, min_word_length=4).generate(text)

# Display the generated image:
plt.figure(figsize=(18,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
sns.set_context('talk')
plt.title('Most common tokenized words', fontsize=25)
#plt.savefig('images/output_images/mbti_token_cloud.png')
plt.show()
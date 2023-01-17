# from google.colab import drive
# drive.mount('/content/drive')
# on kaggle
import numpy as np #
import pandas as pd #

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = (os.path.join(dirname, filename))
# #on local 
# filepath = '../drivedata/coursea_data.csv'
# # on Google drive
# filepath = "/content/drive/My Drive/Colab Notebooks/coursera/data/coursea_data.csv"
df = pd.read_csv(filepath, index_col=0)
from collections import defaultdict
dct_convert_unit = defaultdict(lambda:0)
dct_convert_unit['k']  = 1000
dct_convert_unit['m']  = 1000*1000

df['course_students_enrolled_unit'] = df['course_students_enrolled'].str[-1].apply(lambda x : dct_convert_unit[x])
df['course_students_enrolled'] = df['course_students_enrolled'].str[:-1].astype(float)
df['course_students_enrolled'] = df['course_students_enrolled'] * df['course_students_enrolled_unit'] 
del df['course_students_enrolled_unit'] 
ls_course_Certificate_type = ['COURSE','SPECIALIZATION' ,'PROFESSIONAL CERTIFICATE']
ls_course_difficulty = [ 'Beginner', 'Intermediate', 'Advanced','Mixed']
df.shape[0]
for _col  in df.columns:
    if df[_col].dtype == 'object':  
        display(pd.DataFrame(df[_col].value_counts()))
for _col  in df.columns:
    if df[_col].dtype == 'object':  
        display(pd.DataFrame(df[_col].value_counts(normalize=True)*100).cumsum())
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
df.sort_values(by="course_students_enrolled",ascending=True)
for _col  in ['course_Certificate_type','course_difficulty']:   
    fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
    sns.countplot(x=_col, data=df, ax= ax, order = df[_col].value_counts().index, alpha=0.7)
fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
# sns.countplot(x='course_Certificate_typea', hue ='course_difficulty', order = ls_course_Certificate_type,hue_order=ls_course_difficulty,  data=df, ax= ax, alpha=0.7)
# ax.legend(loc='upper right')
df_pivot_cert_difficult = df[['course_Certificate_type', 'course_difficulty']].pivot_table(
        index='course_difficulty', columns='course_Certificate_type',
        aggfunc=len, fill_value=0)
df_pivot_cert_difficult = df_pivot_cert_difficult.loc[ls_course_difficulty,  ls_course_Certificate_type]
sns.heatmap(df_pivot_cert_difficult, annot=True, fmt="1.1f",cmap="Reds")

fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
sns.countplot(x='course_rating', data=df, ax= ax,hue = 'course_difficulty', order = sorted(list(set(df.course_rating))), hue_order=ls_course_difficulty, alpha=0.7)
ax.legend()
df_rating_dist =  df[['course_rating', 'course_difficulty']] \
    .pivot_table(
        columns='course_difficulty', index='course_rating',
        aggfunc=len, fill_value=0)

display(df_rating_dist.T)


df_rating_dist = 100*df_rating_dist/df_rating_dist.sum()
fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
ls_xticklabel = list(df_rating_dist.index)
for i , _col in enumerate(ls_course_difficulty):
    ax.bar(
        list(map(lambda x: (0.2)*i+x, range(len((ls_xticklabel))))), 
        height=df_rating_dist[_col].values,
        label=_col,
        alpha=0.6, width=0.2)
ax.set_xticks(range(len((ls_xticklabel))))
ax.set_xticklabels(ls_xticklabel)
plt.xlabel('coruse_rating')
plt.ylabel('coruse_rating')
ax.legend()

df_rating_dist =  df[['course_rating', 'course_difficulty']] \
  .groupby('course_difficulty')['course_rating'].agg(["mean","std"])
display(df_rating_dist)

fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
sns.countplot(x='course_rating', data=df, ax= ax,hue = 'course_Certificate_type', order = sorted(list(set(df.course_rating))), hue_order=ls_course_Certificate_type, alpha=0.7)
ax.legend()
df_rating_dist =  df[['course_rating', 'course_Certificate_type']] \
    .pivot_table(
        columns='course_Certificate_type', index='course_rating',
        aggfunc=len, fill_value=0)

display(df_rating_dist.T)


df_rating_dist = 100*df_rating_dist/df_rating_dist.sum()
fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
ls_xticklabel = list(df_rating_dist.index)
for i , _col in enumerate(ls_course_Certificate_type):
    ax.bar(
        list(map(lambda x: (0.2)*i+x, range(len((ls_xticklabel))))), 
        height=df_rating_dist[_col].values,
        label=_col,
        alpha=0.6, width=0.2)
ax.set_xticks(range(len((ls_xticklabel))))
ax.set_xticklabels(ls_xticklabel)
plt.xlabel('coruse_rating')
plt.ylabel('count[%]')
ax.legend()

df_rating_dist =  df[['course_rating', 'course_Certificate_type']] \
  .groupby('course_Certificate_type')['course_rating'].agg(["mean","std"])
display(df_rating_dist)

fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
_cert_type = 'PROFESSIONAL CERTIFICATE'
sns.distplot(df.loc[:, "course_students_enrolled"],ax=ax, label = _cert_type)
_cert_type
df_large = df[df.course_students_enrolled > 500000] \
  .sort_values(by="course_students_enrolled", ascending =False)
print(df_large.shape[0])
display(df_large)
df_small = df[df.course_students_enrolled < 10000]
print(df_small.shape[0])
display(df_small)
import copy

df['log_course_students_enrolled'] = np.log10(df.course_students_enrolled)
display(pd.DataFrame(df.groupby('course_Certificate_type')['log_course_students_enrolled'].agg(['mean','std','count','min','max'])))
display(df.loc[df.course_Certificate_type == "PROFESSIONAL CERTIFICATE", ["course_title","course_organization","course_students_enrolled"]])
fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
for _cert_type in ls_course_Certificate_type:
  sns.kdeplot(df.loc[df.course_Certificate_type == _cert_type, "log_course_students_enrolled"],ax=ax, label = _cert_type)
# plt.xlim(-10000,500000)
plt.xlabel('log course_students_enrolled')
plt.ylabel('p')



display(pd.DataFrame(df.groupby('course_difficulty')['course_students_enrolled'].agg(['mean','std','count','min','max'])))
fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
for _diff in ['Beginner', 'Intermediate', 'Mixed', 'Advanced']:
  sns.kdeplot(df.loc[df.course_difficulty == _diff, "course_students_enrolled"],ax=ax, label = _diff)
# plt.xlim(-10000,500000)
plt.legend()
plt.xlabel('course_students_enrolled')
plt.ylabel('p')
_col ="course_organization"
fig, ax = plt.subplots(figsize=(8,6), facecolor='w')
sns.countplot(x=_col, data=df, order=df[_col].value_counts().index, ax= ax, alpha=0.7)
df_org = pd.DataFrame(df[_col].value_counts())
mean_part = df_org.mean().values[0]
print("mean: {0}".format(mean_part))
ax.hlines(y=mean_part, xmin=0, xmax=ax.get_xlim()[1], color='r')
ax.set(xlim=(0,70))
plt.xticks(rotation= 90)
df_org = df_org.reset_index()
df_org.columns = ['course_organization', 'n_held']
df_org['course_organization_type'] = 'low_held'
df_org.loc[df_org['n_held']>mean_part,'course_organization_type']  = 'high_held'
df_with_held = pd.merge(df  , df_org, on='course_organization', how='left')
sns.pairplot(df_with_held,hue='course_organization_type',palette="husl")
print("course_students_enrolled")
df_with_held.groupby(['course_organization_type'])['course_students_enrolled'].agg(['mean','count','min','max'])
!pip install gensim
!pip install -q wordcloud
!pip install pycld2
import wordcloud
import nltk
import gensim
from gensim import corpora
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
import unicodedata
import re
import string
import pycld2 as cld2
"""
Set up NLP
"""

# Constants
# POS (Parts Of Speech) for: nouns, adjectives, verbs and adverbs
DI_POS_TYPES = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'} 
POS_TYPES = list(DI_POS_TYPES.keys())

# Constraints on tokens
MIN_STR_LEN = 3
RE_VALID = '[a-zA-Z]'

# Get stopwords, stemmer and lemmatizer
stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Remove accents function
# (Normalization Form Compatibility Decomposition)
def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters or x == " ")

def tokenize(sentence):
    # Tokenize by sentense:
    tokens = [word.lower() for sent in nltk.sent_tokenize(sentence) for word in nltk.word_tokenize(sent)]
    ls_tokenized = []
    for token in tokens:

        # Remove accents
        t = remove_accents(token)

        # Remove punctuation
        t = str(t).translate(string.punctuation)

        #remove stop word
        if t in stopwords:
            continue

        # remove signal etc...
        if not re.search(RE_VALID, t):
            continue

        # #remove too short
        # if len(t) < MIN_STR_LEN:
        #   continue

        #stem and lemma
        # t = stemmer.stem(t)
        # t = lemmatizer.lemmatize(t)#, pos=DI_POS_TYPES[])

        ls_tokenized.append(t)
    return ls_tokenized
# Detect monther language of title
df['course_title_lang'] = [cld2.detect(_title)[2][0][1]  for _title in df.course_title.values]
print("course_students_enrolled")
df.groupby(['course_title_lang'])['course_students_enrolled'].agg(['mean','count','min','max']).sort_values(by='count')
df_en = df[df.course_title_lang=='en']
dct_title = {i :tokenize(_token) for i, _token in enumerate(df_en.course_title.values)}
df_words = pd.DataFrame(nltk.FreqDist([flatten for inner in dct_title.values() for flatten in inner]),index=['count']).T.sort_values(by='count',ascending=False)
fig, ax = plt.subplots(figsize=(20,6), facecolor='w')
df_words[df_words['count']>20].plot.bar(ax=ax)
plt.xticks(rotation =75,fontsize=16)
plt.xlabel('top appear words')
plt.ylabel('count')
print(df_words.shape[0])
ls_top_words = list(df_words[df_words['count']>20].index)
df_top_words=pd.DataFrame(
    {i :len(list(set(ls_top_words) & set(ls_words) )) for i , ls_words in enumerate(dct_title.values())},
    index=['top_w_count']).T
df_with_top_words = pd.concat([df_en.reset_index(drop=True),df_top_words],axis=1)
_col = 'top_w_count'
fig, ax = plt.subplots(figsize=(20,6), facecolor='w')
df_plot = df_with_top_words.groupby([_col])['course_students_enrolled'].agg(['mean']).reset_index()
sns.regplot(data=df_plot, x='top_w_count', y='mean',ax=ax,scatter_kws={"s": 80, "color":"r"}, robust=True, ci=None)
plt.xticks(rotation =0,fontsize=16)
plt.xlim(-0.5,6.5)
plt.xlabel('#top_word in title')
plt.ylabel('mean #course_students_enrolled')

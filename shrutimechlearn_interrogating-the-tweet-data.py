import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

import time 

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from IPython.core.display import display, HTML

import plotly.graph_objects as go

import re

# Natural Language Tool Kit 

import nltk  

nltk.download('stopwords') 

from nltk.corpus import stopwords 

from nltk.stem.porter import PorterStemmer 
sns.set()
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", index_col= 'id')

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", index_col= 'id')

submission =  pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv", index_col= 'id')
train.shape
test.shape
train.info()
train.head()
train.isnull().sum()
test.isnull().sum()
train.location.unique()[-10:-1]
train['location_has_hash'] = train.location.apply(lambda x: 1 if '#'in str(x) else 0 )

test['location_has_hash'] = test.location.apply(lambda x: 1 if '#'in str(x) else 0 )
train['location_treated'] = train.location.str.lower().str.replace(r"[^A-Z|a-z|0-9]"," ").str.strip()

test['location_treated'] = test.location.str.lower().str.replace(r"[^A-Z|a-z|0-9]"," ").str.strip()
train.keyword.unique()[-10:-1]
train['keyword_has_hash'] = train.keyword.apply(lambda x: 1 if '%20'in str(x) else 0 )

train['keyword'] = train.keyword.str.replace(r"%20"," ")



test['keyword_has_hash'] = test.keyword.apply(lambda x: 1 if '%20'in str(x) else 0 )

test['keyword'] = test.keyword.str.replace(r"%20"," ")
train['keyword_treated'] = train.keyword.str.lower().str.replace(r"[^A-Z|a-z|0-9]"," ").str.strip()

test['keyword_treated'] = test.keyword.str.lower().str.replace(r"[^A-Z|a-z|0-9]"," ").str.strip()
train['text'] = train.text.str.lower().str.strip()

test['text'] = test.text.str.lower().str.strip()



train['text_has_mentions'] = train.text.apply(lambda x: 1 if '@'in str(x) else 0 )

test['text_has_mentions'] = test.text.apply(lambda x: 1 if '@'in str(x) else 0 )



train['text_mentions_count'] = train.text.apply(lambda x: str(x).count("@"))

test['text_mentions_count'] = test.text.apply(lambda x: str(x).count("@"))
train.head()
ax = sns.countplot(x = 'target', data = train )



for p in ax.patches:

    ax.annotate(f'{p.get_height():.0f}\n({p.get_height() / (train.target.count()) * 100:.1f}%)', 

                xy=(p.get_x() + p.get_width()/2., p.get_height()), ha='center', xytext=(0,5), textcoords='offset points')

ax.set_ylim(0, 2*train.target.sum())

_ = plt.title('Target Analysis')
train['target_mean'] = train.groupby('keyword_treated')['target'].transform('mean')



fig = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=train.sort_values(by='target_mean', ascending=False)['keyword_treated'],

              hue=train.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()



train.drop(columns=['target_mean'], inplace=True)
# helper refer https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud



STOPWORDS.add('https')  # remove htps to the world Cloud



def plot_world(text, bg_color):

    

    comment_words = ' '

    stopwords = set(STOPWORDS) 

    

    for val in text: 



        # typecaste each val to string 

        val = str(val) 



        # split the value 

        tokens = val.split() 



#         # Converts each token into lowercase 

#         for i in range(len(tokens)): 

#             tokens[i] = tokens[i].lower() 



        for words in tokens: 

            comment_words = comment_words + words + ' '





    wordcloud = WordCloud(width = 5000, height = 4000, 

                    background_color =bg_color, 

                    stopwords = stopwords, 

                    min_font_size = 10).generate(comment_words) 



    # plot the WordCloud image                        

    plt.figure(figsize = (12, 12), facecolor = 'k', edgecolor = 'k' ) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 



    plt.show() 
text_1 = train[train.target==1].text.values

plot_world(text_1, 'green')
text_0 = train[train.target==0].text.values

plot_world(text_0, 'red')
text_1 = train[train.target==1].keyword_treated.fillna("").values

plot_world(text_1, 'green')
text_0 = train[train.target==0].keyword_treated.fillna("").values

plot_world(text_0, 'red')
train['text_length'] = train['text'].fillna("").apply(len)

test['text_length'] = test['text'].fillna("").apply(len)



train['keyword_length'] = train['keyword_treated'].fillna("").apply(str).apply(len)

test['keyword_length'] = test['keyword_treated'].fillna("").apply(str).apply(len)



train['location_length'] = train['location'].fillna("").apply(str).apply(len)

test['location_length'] = test['location'].fillna("").apply(str).apply(len)
_ = sns.factorplot(y = 'text_length', x = 'target', data = train, kind = 'box')
_ = sns.factorplot(x = 'text_length', y = None, data = train, kind = 'count', aspect = 2.5)
_ = sns.factorplot(x = 'text_length', y = None, data = test, kind = 'count', aspect = 2.5)
_ = sns.factorplot(y = 'keyword_length', x = 'target', data = train, kind = 'box')
_ = sns.factorplot(x = 'keyword_length', y = None, data = train, kind = 'count', aspect = 2.5)
_ = sns.factorplot(x = 'keyword_length', y = None, data = test, kind = 'count', aspect = 2.5)
_ = sns.factorplot(y = 'location_length', x = 'target', data = train, kind = 'box')
_ = sns.factorplot(x = 'location_length', y = None, data = train, kind = 'count', aspect = 2.5)
_ = sns.factorplot(x = 'location_length', y = None, data = test, kind = 'count', aspect = 2.5)
print(str(100*test.location_treated.isin(train.location_treated).sum()/test.location_treated.isin(train.location_treated).count())+" % of unique test locations are from train")
print(str(test.keyword_treated.isin(train.keyword_treated).sum()*100/test.keyword_treated.isin(train.keyword_treated).count())+" % of unique test keywords are from train") 
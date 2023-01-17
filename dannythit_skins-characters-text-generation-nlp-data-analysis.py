# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing all necessary libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import lxml

from bs4 import BeautifulSoup as soup 

import requests

from wordcloud import WordCloud, ImageColorGenerator

import nltk
df = pd.read_csv('/kaggle/input/skins-season-1-script-with-character-cues/skins_gen_one.csv')
df.head()
df.shape
# Make all the text lowercase (done)

# Remove Punctuation (done)

# Remove Numerical Values (done)

# Remove non-sensical text (done)

# Tokenize Text (done)

# Remove Stop Words (done)

# Getting rid of irrevelent characters (done)



pd.set_option('display.max_rows', None)
df['Characters'].value_counts()
# Make all the text lowercase



df['Text'] = df['Text'].str.lower()
df.head()
# Remove non-sensical text

import re



df['Text'] = df['Text'].str.replace(r"\(.*\)","")
# Remove Punctuation



import string



df['Text'] = df['Text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df.head()
# Remove Numerical Values



df['Text'] = df['Text'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]) )
df.head()
# Getting rid of irrevelent characters (done)



rev_char = df['Characters'].value_counts().sort_values(ascending=False)[:10].index

df = df.loc[df['Characters'].isin(rev_char)]
df['Characters'].value_counts()
df.shape
df.head()
# Setting empty text values to NaN and dropping them 



df['Text'] = df['Text'].apply(lambda x: x if len(x) > 1 else np.nan )

df = df.dropna(axis='rows')
df.isna().sum()
speech = df.groupby(['Characters'], as_index=True)['Text'].apply(lambda x: ''.join(x)).reset_index()

speech.set_index(['Characters'], inplace=True)



speech
from sklearn.feature_extraction.text import CountVectorizer 



stop_words = nltk.corpus.stopwords.words('english')

extension = ['im', 'ill', 'ive', 'youre', 'dont', 'gonna', 'going', 'go', 'gotta', 'ok', 'got', 'like', 'yeah', 'get',

            'hes', 'oh']

stop_words.extend(extension)



cv = CountVectorizer(stop_words= stop_words)

data_cv = cv.fit_transform(speech['Text'])

data_term_matrix = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())

data_term_matrix.index = speech.index



data_term_matrix = data_term_matrix.T

data_term_matrix.head()
df = pd.read_csv('/kaggle/input/skins-season-1-script-with-character-cues/dtm.csv')
df.head()
df.rename(columns={'Unnamed: 0' : 'words'}, inplace=True)
def most_common(col):    

        

    empty_df = pd.DataFrame([])



    char_word = df[['words', col]]

    

    count = df[col].sort_values(ascending=False)[0:10]

    

    top_words = char_word.loc[char_word[col].isin(count)].sort_values(by=col, ascending=False)[0:10]

    

    word_count = str(list(zip(top_words['words'], top_words[col])))

    

    top_ten = pd.concat([empty_df, top_words])

    

    print(top_ten, '\n')
characters = df.columns[1:]
for character in characters:

    most_common(character)
text = pd.read_csv('/kaggle/input/skins-season-1-script-with-character-cues/clean_data.csv')
stop_words = nltk.corpus.stopwords.words('english')

    

extension = ['im', 'ill', 'ive', 'youre', 'dont', 'gonna', 'going', 'go', 'gotta', 'ok', 'got', 'like', 'yeah', 'get',

            'hes', 'oh']

    

stop_words.extend(extension)    
def word_cloud(character):



    words = str(text.loc[text['Characters'] == character]['Text'].values)

        

    wc = WordCloud(stopwords=stop_words, background_color = 'white', random_state=42, max_words=35,

                    min_font_size=10).generate(words)

        

    plt.imshow(wc, interpolation='bilinear')

    

    plt.axis('off')

    

    plt.title(character)
word_cloud('angie')
word_cloud('anwar')
word_cloud('cassie')
word_cloud('jal')
word_cloud('mark')
word_cloud('maxxie')
word_cloud('michelle')
word_cloud('sid')
word_cloud('tony')
word_cloud('angie')
data = pd.read_csv('/kaggle/input/skins-season-1-script-with-character-cues/skins_gen_one.csv')
data.head()
# Remove Numerical Values



data['Text'] = data['Text'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]) )
# Remove non-sensical text

import re



data['Text'] = data['Text'].str.replace(r"\(.*\)","")
characters = ['angie', 'anwar', 'cassie', 'chris', 'jal', 'mark', 'maxxie', 'michelle', 'sid', 'tony']
df = data.groupby('Characters', as_index=True)['Text'].apply(lambda x: ''.join(x)).reset_index()

df.head()
df = df.loc[df['Characters'].isin(characters)]
df['Text'] = df['Text'].apply(lambda x: x.replace('...', '.'))
df['Text'] = df['Text'].str.replace('"', '', regex=False) #removing double quotes

df['Text'] = df['Text'].str.replace("'", '', regex=False) #removing singular quotes
df.head()
# characters/text in list



def char_text(name):

    text = str(df.loc[df['Characters'] == name]['Text'].values)[3:-3]

    

    return text
# creating the markov chain

# Credit goes to Alice Zhao for providing the function for Markov Chain 



from collections import defaultdict



def markov_chain(text):

    

    words = text.split(' ')

    

    m_dict = defaultdict(list)

    

    for current_word, next_word in zip(words[0:-1], words[1:]):

        m_dict[current_word].append(next_word)



    m_dict = dict(m_dict)



    return m_dict
import random



# Credit goes to Alice Zhao for providing the function for Markov Chain 



def txt_generator(chain, count=15):

    

    current_word = random.choice(list(chain.keys()))

    

    sentence = current_word.capitalize()

    

    for i in range(count):

        word_two = random.choice(chain[current_word])

        

        current_word = word_two

        

        sentence += ' ' + word_two

        

    return sentence
txt_generator(markov_chain(char_text('angie')))
txt_generator(markov_chain(char_text('anwar')))
txt_generator(markov_chain(char_text('cassie')))
txt_generator(markov_chain(char_text('chris')))
txt_generator(markov_chain(char_text('jal')))
txt_generator(markov_chain(char_text('mark')))
txt_generator(markov_chain(char_text('maxxie')))
txt_generator(markov_chain(char_text('michelle')))
txt_generator(markov_chain(char_text('sid')))
txt_generator(markov_chain(char_text('tony')))
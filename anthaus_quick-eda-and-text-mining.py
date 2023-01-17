# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dfi = pd.read_csv('/kaggle/input/world-war-i-letters/index.csv')

dfp = pd.read_csv('/kaggle/input/world-war-i-letters/places.csv')



dfi.head(10)
print(len(dfi))

print(len(dfp))
dfi.isna().sum()
print(set(dfi['language']))

print(set(dfi['year']))

print(set(dfi['source']))
french = dfi[dfi['language'] == 'french']

english = dfi[dfi['language'] == 'english']



fig, ax = plt.subplots()

ax.pie([len(french), len(english)], labels=['French', 'English'], autopct='%1.1f%%')

ax.set_title('Part of each language in the dataset')
years = {}

for y in set(dfi['year']):

    if y > 0:

        years[y] = len(dfi[dfi['year']== y])



fig, ax = plt.subplots()

ax.bar(years.keys(), years.values())

ax.set_xlabel('Year of writing')

ax.set_ylabel('Number of letters')

ax.set_title('Number of letters by year in the dataset')
from wordcloud import WordCloud

from nltk.corpus import stopwords

from stop_words import get_stop_words



with open('/kaggle/input/world-war-i-letters/letters.json', 'r') as f:

    letters_data=f.read()

    

letters = json.loads(letters_data)



def clean_str(st, lang='english'):

    aux = st.replace('\n',' ').lower()

    res = ''

    for word in aux.split(' '):

        if word not in get_stop_words(lang):

            res += word + ' '

    return res

    

f_letters = {}

e_letters = {}



for idx, row in dfi.iterrows():

    if row['language'] == 'french':

        f_letters[row['letter_key']] = letters[row['letter_key']]

    elif row['language'] == 'english':

        e_letters[row['letter_key']] = letters[row['letter_key']]



def combine_letters(l_dict, lang='english'):

    res = ''

    for letter in l_dict.values():

        res += clean_str(letter, lang)

    return res



f_str = combine_letters(f_letters, lang='french')

e_str = combine_letters(e_letters)
fig, ax = plt.subplots()

wc_f = WordCloud(colormap='Blues').generate_from_text(f_str)

ax.imshow(wc_f)
fig, ax = plt.subplots()

wc_e = WordCloud(colormap='Reds').generate_from_text(e_str)

ax.imshow(wc_e)
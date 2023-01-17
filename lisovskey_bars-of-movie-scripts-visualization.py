from glob import glob

from os.path import basename

from collections import Counter

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import re



sns.set_style('whitegrid')

%config InlineBackend.figure_format = 'retina'
scripts = [(basename(filename).split('.')[0], open(filename).read())

           for filename in glob('../input/*.txt')]

scripts = pd.DataFrame(scripts, columns=['movie', 'text'])

scripts = scripts.sort_values('movie')

scripts = scripts.reset_index(drop=True)



plt.figure(figsize=(10, 6))

ax = sns.barplot(scripts['text'].apply(len), scripts['movie'], orient='h')

ax.set(xlabel='length');
text = '\n'.join(scripts['text'])

chars = pd.DataFrame(Counter(text).most_common(), columns=['char', 'frequency'])



letters = chars[chars['char'].str.match(r'[а-яА-Я]')].copy()

letters['uppercase'] = letters['char'].str.isupper()

letters['char'] = letters['char'].str.lower()



plt.figure(figsize=(10, 10))

sns.barplot(x='frequency', y='char', hue='uppercase', orient='h', data=letters);
digits = chars[chars['char'].str.match(r'\d')].copy()



plt.figure(figsize=(10, 3))

sns.barplot(x='frequency', y='char', orient='h', data=digits);
def split_script(script):

    return re.split(r'\s+', re.sub(r'[^а-я\s]', '', script.lower()))



stop = stopwords.words('russian')

words = pd.Series(split_script(text))

words = words[~words.isin(stop)]

common_words = words.value_counts()[1:40]



plt.figure(figsize=(10, 10))

ax = sns.barplot(x=common_words, y=common_words.index, orient='h')

ax.set(xlabel='frequency', ylabel='word');
titles = [

    'аватар',

    'джанго освобожденный',

    'эд вуд',

    'бойцовский клуб',

    'от заката до рассвета',

    'умница уилл хантинг',

    'день сурка',

    'омерзительная восьмерка',

    'залечь на дно в брюгге',

    'бесславные ублюдки',

    'убийство священного оленя',

    'матрица',

    'мюнхен',

    'криминальное чтиво',

    'бешеные псы',

    'очень страшное кино',

    'крик',

    'звездный десант',

    'три билборда',

    'дикость',

]

title_frequency = [script.count(title)

                   for title, script in zip(titles, scripts['text'].str.lower())]



plt.figure(figsize=(10, 6))

ax = sns.barplot(title_frequency, scripts['movie'], orient='h')

ax.set(xlabel='title frequency');
def mean_word_len(script):

    words = split_script(script)

    return sum(len(word) for word in words) / len(words)



plt.figure(figsize=(10, 6))

ax = sns.barplot(scripts['text'].apply(mean_word_len), scripts['movie'], orient='h')

ax.set(xlabel='mean word length');
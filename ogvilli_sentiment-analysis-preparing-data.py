import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

print(np.__version__)



import os

print(os.listdir("../input"))

np.random.seed(42)
import nltk

import re

import seaborn as sns

import matplotlib.pyplot as plt

from random import randint

from time import sleep

from itertools import tee
df_train = pd.read_csv('../input/data-sentiment-analysis/products_sentiment_train.tsv', 

                       sep = '\t', header = None, names = ['text', 'label'])
df_test = pd.read_csv('../input/data-sentiment-analysis/products_sentiment_test.tsv', sep = '\t')

df_test.columns = ['Id', 'text']
df = pd.DataFrame({'text': pd.concat([df_train['text'], df_test['text']],axis = 0)})

df.shape
sns.countplot(df_train['label']);

plt.title('Train: Target distribution');
fig = plt.figure(figsize = (15, 5));

ax1 = fig.add_subplot(121);

df_train['text'].apply(lambda x: len(x.split())).hist(bins = 20);

plt.title('Train: Number of words');



ax2 = fig.add_subplot(122);

df_test['text'].apply(lambda x: len(x.split())).hist(bins = 20);

plt.title('Test: Number of words');
print('Среднее кол-во слов в обучении и тесте:', 

      df_train['text'].apply(lambda x: len(x.split())).mean(), ' и ',

      df_test['text'].apply(lambda x: len(x.split())).mean())
print('Медианы кол-ва слов в обучении и тесте:', 

      df_train['text'].apply(lambda x: len(x.split())).median(), ' и ',

      df_test['text'].apply(lambda x: len(x.split())).median())
print('Мера разброса кол-ва слов в обучении и тесте:', 

      df_train['text'].apply(lambda x: len(x.split())).std(), ' и ',

      df_test['text'].apply(lambda x: len(x.split())).std())
print('Всего слов в трейновой выборке:', df_train['text'].apply(lambda x: len(x.split())).sum())

print('Всего слов в тестовой выборке:', df_test['text'].apply(lambda x: len(x.split())).sum())
indices = np.random.randint(low = 0, high = len(df_train), size = 20)

df_train.iloc[indices]['text'].values
def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

    return text
%%capture

!pip install googletrans
from googletrans import Translator

translator = Translator(service_urls=['translate.google.com',  'translate.google.es', 'translate.google.ru'])
def substitute_special_symb(mystring):

    return re.sub('&#', '', mystring)
print('Оригинальный вариант:', 'all in all i am pleased with the router .')



translations_es = translator.translate(['all in all i am pleased with the router .'], dest='es')

translations_en = translator.translate([d.text for d in translations_es], dest='en')



for translation in translations_en:

    print('Аугментированный текст:', translation.text)
print('Оригинальный вариант:', 'all in all i am pleased with the router .')



translations_ru = translator.translate(['all in all i am pleased with the router .'], dest='ru')

translations_en = translator.translate([d.text for d in translations_ru], dest='en')



for translation in translations_en:

    print('Аугментированный текст:', translation.text)
# augmented_train = []

# chunk_size = 50



# for i in range(38, np.int(len(df_train)/chunk_size)):

#    translations_es = translator.translate(df_train['text'].apply(substitute_special_symb).tolist()[chunk_size*i:chunk_size*(i+1)], dest='es')

#    sleep(randint(1, 5))

#    translations_en = translator.translate([d.text for d in translations_es], dest='en')

#    sleep(randint(1, 5))



#    for translation in translations_en:

#        augmented_train.append(translation.text)

        

#    print(i+1, ' iteration has been ended')
# with open('augmented_train.txt', 'w') as f:

#     for item in augmented_train:

#         f.write("%s\n" % item)
# augmented_test = []

# chunk_size = 50



# for i in range(8, np.int(len(df_test)/chunk_size)):

#     translations_es = translator.translate(df_test['text'].apply(substitute_special_symb).tolist()[chunk_size*i:chunk_size*(i+1)], dest='es')

#     sleep(randint(1, 4))

#     translations_en = translator.translate([d.text for d in translations_es], dest='en')

#     sleep(randint(1, 4))



#     for translation in translations_en:

#         augmented_test.append(translation.text)

        

#     print(i+1, ' iteration has been ended')
# with open('augmented_test.txt', 'w') as f:

#     for item in augmented_test:

#         f.write("%s\n" % item)
print(os.listdir("../input/augmented-data"))
df_train_augmented = pd.read_csv('../input/augmented-data/augmented_train.txt', sep = '\n', 

                              header = None, names = ['text'])

df_train_augmented['label'] = df_train.label
df_test_augmented = pd.read_csv('../input/augmented-data/augmented_test.txt', sep = '\n', 

                              header = None, names = ['text'])
df['text_translated'] = pd.DataFrame({'text': pd.concat([df_train_augmented['text'], df_test_augmented['text']], axis = 0)})

df.shape
%%capture

!pip install autocorrect
%%capture

!pip install pyspellchecker
from gensim.models import Word2Vec, KeyedVectors

from autocorrect import spell, Speller

from spellchecker import SpellChecker
%%time

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)

# path = '../input/gensim-word-vectors/'

GLOVE_TWITTER = '../input/gensim-word-vectors/glove-twitter-100/glove-twitter-100'

twitter_model = KeyedVectors.load_word2vec_format(GLOVE_TWITTER)
%%time

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)

GOOGLE_NEWS = '../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin.gz'

news_model = KeyedVectors.load_word2vec_format(GOOGLE_NEWS, binary=True)
def get_similar_bag_of_words_ordered(phrase, topn = 5, model = twitter_model):  

    bag_words = []

    ordered_bag_words = []

    # считаем вектор по каждому слову из фразы

    for word in phrase.split():

        try:

            bag_words.append([item[0] for item in model.most_similar([word], topn = topn)]) 

        except KeyError:

            try:

                subbag = [item[0]  for item in model.most_similar([spell(word)], topn = topn-1)]

                subbag.append(spell(word))

                bag_words.append(subbag)

            except KeyError:

                continue

                

    for i in range(topn):

        for w in bag_words:

            ordered_bag_words.append(w[i])

        

    return ' '.join([w for w in ordered_bag_words])
get_similar_bag_of_words_ordered('treee grow slowly')
def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

    return text
%%time

%%capture



# на основе твитов

df['text_w2v_twitter'] = df['text'].apply(preprocessor).apply(get_similar_bag_of_words_ordered)
%%time

%%capture



# на основе новостей

def get_similar_bag_of_words_news_ordered(phrase):

    return get_similar_bag_of_words_ordered(phrase = phrase, topn = 5, model = news_model)



df['text_w2v_news'] = df['text'].apply(preprocessor).apply(get_similar_bag_of_words_news_ordered)
print(df.shape)
df.head()
df.to_csv('enriched_train_test_text.csv', index = 'false')
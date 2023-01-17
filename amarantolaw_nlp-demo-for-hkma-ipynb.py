import nltk

import csv

import seaborn as sns

import os

os.chdir('../input')

# Reference article: https://www.scmp.com/news/china/diplomacy/article/3017772/senior-chinese-diplomat-warns-disastrous-consequences-if-us
with open('trade_war.csv', 'rb') as csvfile:

    text = str(csvfile.read())

    print(text)
tokens = nltk.word_tokenize(text)

print(tokens)
#NAMED ENTITY RECOGNITION



for sent in nltk.sent_tokenize(text):

   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):

      if hasattr(chunk, 'label'):

         print(chunk.label(), ' '.join(c[0] for c in chunk))
named_entity_count_dict = {}



for sent in nltk.sent_tokenize(text):

   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):

      if hasattr(chunk, 'label'):

            entity = ' '.join(c[0] for c in chunk)

            named_entity_count_dict[entity] = named_entity_count_dict.get(entity, 0) + 1

            

named_entity_count_dict

data = [[value, key] for key, value in named_entity_count_dict.items()]

data = sorted(data,reverse=True)[:10]

data
sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(11.7,8.27)})

ax = sns.barplot(x=[x[0] for x in data], y=[x[1] for x in data])
# SENTIMENT ANALYSIS

positive_dictonary = [ 'good', 'great', 'wonderful', 'awesome', 'outstanding', 'fantastic', 'terrific', 'nice',]

negative_dictionary = [ 'bad', 'terrible','disastrous', 'stupid']

neutral_dictionary = [ 'place','the','paper','was','is','actors','did','know','words','not']
positive_words = [word for word in tokens if word in positive_dictonary]

negative_words = [word for word in tokens if word in negative_dictionary]

neutral_words = [word for word in tokens if word in neutral_dictionary]



print("positive matches: ", positive_words)

print("negatives matches: ", negative_words)

print("neutral matches: ", neutral_words)
tags = nltk.pos_tag(tokens)
set([i for i in tags if i[1] in ['JJ','JJR','JJS']])
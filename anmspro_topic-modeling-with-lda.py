import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models


import gensim
dataset = pd.read_csv('../input/voted-kaggle-dataset/voted-kaggle-dataset.csv')
print(dataset['Description'][0])
dataset.head()

dataset.iloc[0:3,:]
dataset.info()

dataset.columns.values
modified_dataset = dataset.loc[ : , ['Title','Subtitle','Description']]
modified_dataset.iloc[0:5,:]
modified_dataset.info()

modified_dataset.isnull()
print(modified_dataset.isnull().sum())
print("total null values : ",sum(modified_dataset.isnull().values.ravel()))
print("total number of rows containing null values : ", sum([True for idx,row in modified_dataset.iterrows() if any(row.isnull())]))
modified_dataset = modified_dataset.dropna()
print("total null values : ",sum(modified_dataset.isnull().values.ravel()))
print("total number of rows containing null values : ", sum([True for idx,row in modified_dataset.iterrows() if any(row.isnull())]))
modified_dataset.info()

import string
import re

remove_digits = str.maketrans('', '', string.digits) # Set of all digits
for column in ['Title','Subtitle','Description']:
  modified_dataset[column] = modified_dataset[column].map(lambda x : x.lower())
modified_dataset.iloc[0:3,:]

exclude = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
for column in ['Title','Subtitle','Description']:
  modified_dataset[column] = modified_dataset[column].map(lambda x : x.translate(remove_digits))
  modified_dataset[column] = modified_dataset[column].map(lambda x : re.sub(str(exclude), '', x))
modified_dataset.iloc[0:3,:]
tag_dataset = dataset['Tags']
tag_dataset.isnull().sum()
#tag_dataset=tag_dataset.dropna()
tag_dataset.isnull().sum()
print(len(tag_dataset))
def convert(lst): 
    return (lst.split("\n"))
unique_tag = []
for i in range(len(tag_dataset)):
  tag_string = str(tag_dataset[i])
  if tag_string != "nan" :
    tag_word=convert(tag_string)
    for j in range(len(tag_word)):
      if tag_word[j] not in unique_tag:
        unique_tag.append(tag_word[j])
print(len(unique_tag))

for i in range(len(unique_tag)):
  print(unique_tag[i])
import nltk
nltk.download('punkt')
tokenized_dataframe =  modified_dataset.apply(lambda row: nltk.word_tokenize(row['Description']), axis=1)
print(type(tokenized_dataframe))
tokenized_dataframe[0:3]
#single word check......
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
print(ps.stem('contains'))
def lemmatize_text(text):
    return [ps.stem(w)  for w in text if len(w)>5]
nltk.download('wordnet')
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
stemmed_dataset = tokenized_dataframe.apply(lemmatize_text)
type(stemmed_dataset)
stemmed_dataset[0:3]
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#dataset_words=''
#for column in ['Title','Subtitle','Description']:
dataset_words=''.join(list(str(stemmed_dataset.values)))
print(type(dataset_words))
wordcloud = WordCloud(width = 800, height = 500, 
                background_color ='white',  
                min_font_size = 10).generate(dataset_words) 

plt.figure(figsize = (5, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

dictionary_of_words = gensim.corpora.Dictionary(stemmed_dataset)

type(dictionary_of_words)
len(dictionary_of_words)
# Creatig coupus which contains  of word id mapping with word_frequency--->(word_id, word_frequency) 
word_corpus = [dictionary_of_words.doc2bow(word) for word in stemmed_dataset]
for corp in word_corpus[:1]:
  for id, freq in corp:
    print(dictionary_of_words[id],freq)
lda_model = gensim.models.ldamodel.LdaModel(corpus=word_corpus,
                                           id2word=dictionary_of_words,
                                           num_topics=329, 
                                           random_state=101,
                                           update_every=1,
                                           chunksize=300,
                                           passes=50,
                                           alpha='auto',
                                           per_word_topics=True)
lda_model.print_topics()

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
from gensim.models.coherencemodel import CoherenceModel

coherence_val = CoherenceModel(model=lda_model, texts=stemmed_dataset, dictionary=dictionary_of_words, coherence='c_v').get_coherence()

print('Coherence Score: ', coherence_val)
coherence_value = []
for topic_number in range(10,331,10):
  lda_model = gensim.models.ldamodel.LdaModel(corpus=word_corpus,
                                           id2word=dictionary_of_words,
                                           num_topics=topic_number, 
                                           random_state=101,
                                           update_every=1,
                                           chunksize=100,
                                           passes=50,
                                           alpha='auto',
                                           per_word_topics=True)
  models.append(lda_model)
  coherence_model_lda = CoherenceModel(model=lda_model, texts=stemmed_dataset, dictionary=dictionary_of_words, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  coherence_value.append(coherence_lda)
  print("number of topics ",topic_number,"coherence_value :" , coherence_lda)
lda_model1 = gensim.models.ldamodel.LdaModel(corpus=word_corpus,
                                           id2word=dictionary_of_words,
                                           num_topics=60, 
                                           random_state=1,
                                           update_every=1,
                                           chunksize=100,
                                           passes=50,
                                           alpha='auto',
                                           per_word_topics=True)
from gensim.models.coherencemodel import CoherenceModel


# Compute Coherence Score
cohr_val = CoherenceModel(model=lda_model1, texts=stemmed_dataset, dictionary=dictionary_of_words, coherence='c_v').get_coherence()

print('\nCoherence Score: ', cohr_val)
from gensim.test.utils import common_corpus, common_dictionary
lda_multicore_model = gensim.models.ldamulticore.LdaMulticore(corpus=word_corpus, 
                                                              num_topics=60, 
                                                              id2word=dictionary_of_words,                                                             
                                                              chunksize=100, 
                                                              passes=50,                                
                                                              alpha='symmetric',
                                                              eta=0.1,
                                                              decay=0.5, 
                                                              offset=1.0, 
                                                              gamma_threshold=0.001,
                                                              random_state=101,
                                                              minimum_probability=0.01,
                                                              minimum_phi_value=0.01,
                                                              per_word_topics=False)
from gensim.models.coherencemodel import CoherenceModel


# Compute Coherence Score
cohr_lda_multicore_model1 = CoherenceModel(model=lda_multicore_model, texts=stemmed_dataset, dictionary=dictionary_of_words, coherence='c_v').get_coherence()

print('\nCoherence Score: ', cohr_lda_multicore_model1)
from gensim.test.utils import common_corpus, common_dictionary
lda_multicore_model2 = gensim.models.ldamulticore.LdaMulticore(corpus=word_corpus, 
                                                              num_topics=329, 
                                                              id2word=dictionary_of_words,                                                             
                                                              chunksize=100, 
                                                              passes=50,                                
                                                              alpha='symmetric',
                                                              eta=0.1,
                                                              decay=0.5, 
                                                              offset=1.0, 
                                                              gamma_threshold=0.001,
                                                              random_state=101,
                                                              minimum_probability=0.01,
                                                              minimum_phi_value=0.01,
                                                              per_word_topics=False)
from gensim.models.coherencemodel import CoherenceModel


# Compute Coherence Score
cohr_lda_multicore_model2 = CoherenceModel(model=lda_multicore_model2, texts=stemmed_dataset, dictionary=dictionary_of_words, coherence='c_v').get_coherence()

print('\nCoherence Score: ', cohr_lda_multicore_model2)

v = lda_model[word_corpus[2]]
print(type(lda_model[word_corpus[2]]))
z=sorted(v[0], key=lambda tup: -1*tup[1])
print(z)
print(v[0])
for  index,score in sorted(lda_model[word_corpus[2]][0], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
topics = lda_model.show_topics(formatted=False)
topic_words = dict(topics[0][1])
wordcloud.generate_from_frequencies(topic_words, max_font_size=100)
plt.figure(figsize = (5, 5), facecolor = None) 
plt.title("topic 0")
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()
fig = plt.figure(figsize=(15,15),frameon=0)
a = fig.add_subplot(1, 2, 1)
topic_words = dict(topics[1][1])
wordcloud.generate_from_frequencies(topic_words, max_font_size=100)
imgplot = plt.imshow(wordcloud)
a.set_title('topic 0')

a = fig.add_subplot(1, 2, 2)
topic_words = dict(topics[2][1])
wordcloud.generate_from_frequencies(topic_words, max_font_size=100)
imgplot = plt.imshow(wordcloud)

a.set_title('topic 1')

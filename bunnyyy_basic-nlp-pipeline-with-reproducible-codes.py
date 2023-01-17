import numpy as np 

import pandas as pd 
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import emoji

import string



def preprocess_data(data, remove_stop = True):

    

    data = re.sub('https?://\S+|www\.\S+', '', data)

    data = re.sub('<.*?>', '', data)

    emoj = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    emoj.sub(r'', data)

    data = data.lower()

    data = data.translate(str.maketrans('','', string.punctuation))

    data = re.sub(r'\[.*?\]', '', data)

    data = re.sub(r'\w*\d\w*','', data)

    

    words= data.split()

    

    if(remove_stop):

        words = [w for w in words if w not in ENGLISH_STOP_WORDS]

        words = [w for w in words if len(w) > 2]  # remove a,an,of etc.

    

    words= ' '.join(words)

    

    return words
from nltk.stem import WordNetLemmatizer 

from nltk.stem import PorterStemmer 

from nltk.tokenize import RegexpTokenizer





def tokenizer(words):

    tokenizer = RegexpTokenizer(r'\w+')

    words= tokenizer.tokenize(words)

    

    return words



def lemmatize(words):

    lemmatizer = WordNetLemmatizer() 

    lem= []

    for w in words:

        lem.append(lemmatizer.lemmatize(w))

    return lem



def stemming(words):

    ps = PorterStemmer() 

    stem= []

    for w in words:

        stem.append(ps.stem(w))

    return stem  
stemming(tokenizer(preprocess_data("@water #dream hi 19 :) hello where are you going be there tomorrow happening")))
import spacy



nlp = spacy.load('en_core_web_lg')



text = """London is the capital and most populous city of England and 

the United Kingdom.  Standing on the River Thames in the south east 

of the island of Great Britain, London has been a major settlement 

for two millennia. It was founded by the Romans, who named it Londinium.

The City of Westminster is also an Inner London borough holding city status.

London is governed by the mayor of London and the London Assembly.

London has a diverse range of people and cultures, and more than 300 languages are spoken in the region.

"""

doc = nlp(text)

from spacy import displacy



for entity in doc.ents:

    print(f"{entity.text} ({entity.label_})")



displacy.render(doc, style="ent") #this needs to be closed
i=0

for token in doc:

    if i<10:

        print(token.text, token.pos_)

    i+=1
doc1 = nlp("Gandhiji was born in Porbandar in 1869.")



displacy.render(doc1, style="dep")



#sentence_spans = list(doc.sents) # To show large text dependency parsing, uncomment this.

def worldcloud(word_list):

    #wordcloud = WordCloud()

    #wordcloud.fit_words(dict(count(word_list).most_common(40)))

    wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                

                min_font_size = 10).generate(word_list)



    fig=plt.figure(figsize=(10, 10))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()
worldcloud(' '.join([token.text for token in doc if token.pos_ in ['NOUN']]))
import gensim



word2vec_path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"

word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):

    if len(tokens_list)<1:

        return np.zeros(k)

    if generate_missing:

        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]

    else:

        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]

   

    length = len(vectorized)

    summed = np.sum(vectorized, axis=0)

    averaged = np.divide(summed, length)

    return averaged
tokens_list= stemming(tokenizer(preprocess_data("@water #dream hi 19 :) hello where are you going be there tomorrow happening")))
tokens_list
get_average_word2vec(tokens_list, word2vec) #This will give an average of all of the token
import nltk
import csv
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation
import keras.utils as kutils
df_ep4 = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeIV.txt', sep =' ', header=0, escapechar='\\')
df_ep5 = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeV.txt', sep =' ', header=0, escapechar='\\')
df_ep6 = pd.read_csv('../input/star-wars-movie-scripts/SW_EpisodeVI.txt', sep =' ', header=0, escapechar='\\')
df_ep4
sw_scripts = list(pd.concat([df_ep4 , df_ep5, df_ep6]).dialogue.values)
sw_scripts[0]
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string

all_sents = [[w.lower() for w in word_tokenize(sen) if not w in string.punctuation] \
             for sen in sw_scripts]

x = []
y = []

print(all_sents[:10])

for sen in all_sents:
    for i in range(1, len(sen)):
        x.append(sen[:i])
        y.append(sen[i])
        

print(x[:10])
print(y[:10])
from sklearn.model_selection import train_test_split
import numpy as np

all_text = [c for sen in x for c in sen]
all_text += [c for c in y]

all_text.append('UNK') # Palavra desconhecida

words = list(set(all_text))
        
word_indexes = {word: index for index, word in enumerate(words)}      

max_features = len(word_indexes)

x = [[word_indexes[c] for c in sen] for sen in x]
y = [word_indexes[c] for c in y]

print(x[:10])
print(y[:10])

y = kutils.to_categorical(y, num_classes=max_features)

maxlen = max([len(sen) for sen in x])

print(maxlen)

x = pad_sequences(x, maxlen=maxlen)
x = pad_sequences(x, maxlen=maxlen)

print(x[:10,-10:])
print(y[:10,-10:])
embedding_size = 10

model = Sequential()
    
# Add Input Embedding Layer
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    
# Add Hidden Layer 1 - LSTM Layer
model.add(LSTM(100))
model.add(Dropout(0.1))
    
# Add Output Layer
model.add(Dense(max_features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
model.fit(x, y, epochs=10, verbose=5)
import pickle

print("Saving model...")
model.save('shak-nlg.h5')

with open('shak-nlg-dict.pkl', 'wb') as handle:
    pickle.dump(word_indexes, handle)

with open('shak-nlg-maxlen.pkl', 'wb') as handle:
    pickle.dump(maxlen, handle)
print("Model Saved!")
import pickle

model = keras.models.load_model('shak-nlg.h5')
maxlen = pickle.load(open('shak-nlg-maxlen.pkl', 'rb'))
word_indexes = pickle.load(open('shak-nlg-dict.pkl', 'rb'))
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
stopwords_list = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

def my_tokenizer(doc):
    words = word_tokenize(doc)
    
    pos_tags = pos_tag(words)
    
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]
    
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]
    
    lemmas = []
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN
        
        lemmas.append(lemmatizer.lemmatize(w[0], pos))

    return lemmas
tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(sw_scripts))
print(tfidf_matrix.shape)
def get_word_by_index(index, word_indexes):
    for w, i in word_indexes.items():
        if index == i:
            return w
        
    return None
def closest_response(question):
    query_vect = tfidf_vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, tfidf_matrix)
    max_similarity = np.argmax(similarity, axis=None)
    return ' '.join(sw_scripts[max_similarity+1].split(' ')[:20])
def chatbot_answer(phrase):
    sample_seed = closest_response(phrase)
    sample_seed_vect = np.array([[word_indexes[c] \
                                  if c in word_indexes.keys() else \
                                 word_indexes['UNK'] \
                                 for c in word_tokenize(sample_seed)]])

    test = pad_sequences(sample_seed_vect, \
                        maxlen=maxlen, \
                        padding='pre')
    
    predicted = []
    i = 0
    while i < 20:
        predicted = model.predict_classes(
                                pad_sequences(sample_seed_vect, \
                                               maxlen=maxlen, \
                                               padding='pre'),\
                                verbose=0)
        new_word = get_word_by_index(predicted[0], word_indexes)
        sample_seed += ' ' + new_word

        sample_seed_vect = np.array([[word_indexes[c] \
                                  if c in word_indexes.keys() else \
                                 word_indexes['UNK'] \
                                 for c in word_tokenize(sample_seed)]])
        i += 1
        
    gen_text = ''
    for index in sample_seed_vect[0][:20]:
        gen_text += get_word_by_index(index, word_indexes) + ' '
        
    output = ''
    for i, gen in enumerate(gen_text.split(' ')):
        if gen == 'UNK':
            output += sample_seed.split(' ')[i] + ' '
        else:
            output += gen + ' '
    print('[SW]Chatbot:', output)
print('Type here:')
z = input()
chatbot_answer(z)
# Dupla:

# Francis Pimentel

# Vinicius Vieira
import numpy as np

import pandas as pd

import keras

from keras.models import Sequential

from keras.layers.recurrent import LSTM

from keras.preprocessing.sequence import pad_sequences

from keras.layers.embeddings import Embedding

from keras.layers.core import Activation, Dense, Dropout

from keras.layers.wrappers import TimeDistributed

from keras.layers.core import Dense, Activation

import keras.utils as kutils
df1 = pd.read_table('../input/SW_EpisodeIV.txt',delim_whitespace=True, header=0, escapechar='\\')

df2 = pd.read_table("../input/SW_EpisodeV.txt",delim_whitespace=True, header=0, escapechar='\\')

df3 = pd.read_table("../input/SW_EpisodeVI.txt",delim_whitespace=True, header=0, escapechar='\\')
df1.info()
df2.info()
df3.info()
all_dialogues = list(pd.concat([df1, df2, df3]).dialogue.values)

print('Tamanho: ', len(all_dialogues))

print(all_dialogues[:10])
from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

import string



all_sents = [[w.lower() for w in word_tokenize(sen) if not w in string.punctuation]\

            for sen in all_dialogues]



x = []

y = []



print(all_sents[:10])



print('\n')

for sen in all_sents:

    for i in range(1, len(sen)):

        x.append(sen[:i])

        y.append(sen[i])

        

print(x[:10])

print('\n')

print(y[:10])
from sklearn.model_selection import train_test_split

import numpy as np



all_text = [c for sen in x for c in sen]

all_text += [c for c in y]



all_text.append('UNK')



print(all_text[:10])
words = list(set(all_text))

print(words[:10])
word_indexes = {word: index for index, word in enumerate(words)}



max_features = len(word_indexes)



print(max_features)
x = [[word_indexes[c] for c in sen] for sen in x]

y = [word_indexes[c] for c in y]



print(x[:10])

print(y[:10])
y = kutils.to_categorical(y, num_classes=max_features)

print(y[:10])
maxlen = max([len(sen) for sen in x])

print(maxlen)

x = pad_sequences(x, maxlen=maxlen)



print(x[:10, -10:])

print(x[:10, -10:])
embedding_size = 10



model = Sequential()



model.add(Embedding(max_features, embedding_size, \

                    input_length=maxlen))



model.add(LSTM(100))

model.add(Dropout(0.1))

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
sample_seed = 'may the force be with'
sample_seed_vect = np.array([[word_indexes[c] \

                              if c in word_indexes.keys() else \

                             word_indexes['UNK'] \

                             for c in word_tokenize(sample_seed)]])

                             

print(sample_seed_vect)
sample_seed_vect = pad_sequences(sample_seed_vect, maxlen=maxlen)

print(sample_seed_vect)
predicted = model.predict_classes(sample_seed_vect, verbose=0)

print(predicted)
def get_word_by_index(index, word_indexes):

    for w, i in word_indexes.items():

        if index == i:

            return w

        

    return None
for p in predicted:

    print(get_word_by_index(p, word_indexes))
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

tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(all_dialogues))

print(tfidf_matrix.shape)
# Dada uma entrada, retorna o proximo dialogo, limitado a 20 palavras

def find_closest_response(question):

    query_vect = tfidf_vectorizer.transform([question])

    similarity = cosine_similarity(query_vect, tfidf_matrix)

    max_similarity = np.argmax(similarity, axis=None)

    

    print('Entrada do usuário:', question)

    print('Diálogo mais próximo encontrado:', all_dialogues[max_similarity])

    print('Similaridade: {:.2%}'.format(similarity[0, max_similarity]))

    print('Diálogo seguinte:', all_dialogues[max_similarity+1])

    return ' '.join(all_dialogues[max_similarity+1].split(' ')[:20])
sample_seed = find_closest_response('may the force be with you')
sample_seed
def obter_resposta_chatbot(entrada):

    sample_seed = find_closest_response(entrada)

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

    print('RESPOSTA CHATBOT:', output)
obter_resposta_chatbot('may the force be with you')
obter_resposta_chatbot('darth vader')
obter_resposta_chatbot('The Force will be with you. Always')
obter_resposta_chatbot('I find your lack of faith disturbing')
obter_resposta_chatbot('Now, young Skywalker, you will die.')
obter_resposta_chatbot('There’s always a bigger fish.')
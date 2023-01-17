import pandas as pd

from nltk.tokenize import MWETokenizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import numpy as np

from matplotlib import pyplot as plt

#from tqdm import tqdm_notebook

import tqdm

from tqdm.notebook import tqdm_notebook

from ipywidgets import IntProgress

import tensorflow as tf

from keras.models import Sequential



from keras import regularizers, initializers, optimizers, callbacks

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.utils.np_utils import to_categorical

from keras.layers import *

from keras.models import Model

stop_words = set(stopwords.words('english'))  # For stopwords

lemmatizer = WordNetLemmatizer()  # For lemmatization

tokenizer = MWETokenizer([('all', 'over'), ('all', 'right'), ('and', 'how'), ('arrival', 'time'), ('as', 'well')

                             , ('arrive', 'at'), ('at', 'least'), ('at', 'the', 'same', 'time'), ('at', 'times')

                             , ('at', 'will'), ('be', 'on'), ('belong', 'to'), ('bring', 'up'), ('car', 'rental')

                             , ('clock', 'in'), ('clock', 'on'), ('close', 'to'), ('coming', 'back'), ('coming', 'back')

                             , ('connecting', 'flight'), ('day', 'of', 'the', 'week'), ('day', 'return')

                             , ('departure', 'time'), ('direct', 'flight'), ('do', 'in'), ('early', 'on'), ('eat', 'on')

                             , ('economy', 'class'), ('equal', 'to'), ('find', 'out'), ('first', 'class'), ('fly', 'on')

                             , ('fort', 'worth'), ('get', 'down'), ('get', 'on'), ('get', 'to'), ('go', 'after')

                             , ('go', 'around'), ('go', 'for'), ('go', 'in'), ('go', 'into'), ('go', 'on')

                             , ('go', 'through'), ('go', 'to'), ('go', 'with'), ('have', 'on'), ('in', 'flight')

                             , ('in', 'for'), ('in', 'on'), ('kansas', 'city'), ('kind', 'of'), ('las', 'vegas')

                             , ('light', 'time'), ('live', 'in'), ('local', 'time'), ('lock', 'in'), ('long', 'beach')

                             , ('look', 'at'), ('look', 'like'), ('looking', 'for'), ('los', 'angeles'), ('many', 'a')

                             , ('more', 'than'), ('morning', 'time'), ('new', 'jersey'), ('new', 'york'),

                          ('number', '1')

                             , ('new', 'york', 'city'), ('nonstop', 'flight'), ('north', 'carolina'), ('of', 'late')

                             , ('on', 'air'), ('on', 'that'), ('on', 'the', 'way'), ('on', 'time'), ('or', 'so')

                             , ('out', 'of'), ('per', 'se'), ('r', 'and', 'b'), ('ring', 'up'), ('round', 'trip')

                             , ('salt', 'lake', 'city'), ('san', 'diego'), ('san', 'francisco'), ('san', 'jose')

                             , ('seating', 'capacity'), ('show', 'business'), ('show', 'up'), ('some', 'other')

                             , ('sort', 'of'), ('st.', 'louis'), ('st.', 'paul'), ('st', '.', 'peter'),

                          ('st.', 'petersburg')

                             , ('take', 'in'), ('stand', 'for'), ('stop', 'over'), ('take', 'ten'), ('take', 'to')

                             , ('thank', 'you'), ('the', 'city'), ('time', 'of', 'arrival'), ('time', 'zone')

                             , ('to', 'and', 'fro'), ('to', 'that'), ('to', 'wit'), ('travel', 'to'), ('turn', 'around')

                             , ('turn', 'in'), ('turn', 'to'), ('type', 'o'), ('up', 'on'), ('used', 'to'),

                          ('very', 'much')

                             , ('very', 'well')], separator=' ')   # Multi word tokens
class_names = ['atis_cheapest', 'atis_quantity', 'atis_flight_no', 'atis_capacity', 'atis_flight_time', 'atis_meal'

    , 'atis_city', 'atis_restriction', 'atis_distance', 'atis_airport', 'atis_abbreviation',

               'atis_ground_fare', 'atis_aircraft', 'atis_ground_service', 'atis_airfare', 'atis_airline',

               'atis_flight']



prototypes = ['cheap', 'count', 'number', 'seating capacity', 'time', 'meal', 'municipality', 'restriction', 'distance',

              'airport', 'code', 'service fee', 'kind of', 'service', 'fare', 'airline', 'flight']



synsets_of_prototypes = [300937468, 113612964, 106436708, 105113179, 200680466, 107589261, 108242502,

                         100809843, 105091408, 102695091, 106680062, 113346869, 400018764, 108202965,

                         113329169, 102692940, 108237455]   # Prototip synsetler
MAX_NB_WORDS = 100

MAX_SEQUENCE_LENGTH = 500

VALIDATION_SPLIT = 0.2

EMBEDDING_DIM = 300
vectors =  pd.read_csv("../input/disambg/vectors300D.csv")

lemma2synset = pd.read_csv("../input/disambg/word2synset.csv")

atis_train = pd.read_csv("../input/disambg/atis_intents_train.csv")

atis_test = pd.read_csv("../input/disambg/atis_intents_test.csv")
y = atis_train['intent'].values

comments_train = atis_train['vocab']

comments_test = atis_test['vocab']



comments_train = list(comments_train)  # Gereksiz

# Test için

for x in range(10):

    print(comments_train[x])
# Tokenization

def clean_text(text):

    word_tokens = tokenizer.tokenize(text.split())  # Tokenization

    tokens = [w for w in word_tokens if not w in stop_words]

    return tokens
# Tokenların olası tüm synsetleri bulunuyor

def find_synsets(tokens):

    storage = []  # It will store the synsets of lemmas

    count = 0



    for token in tokens:

        lemma = lemmatizer.lemmatize(token)  # Lemmatization

        synsets = lemma2synset[lemma2synset['lemma'] == lemma]

        if synsets.empty:  # For unknown synset

            print("Token  -- ", token, " --couldn't found in the wordnet!")

            continue

        else:

            storage.append([])



        for syn in synsets['synsetid']:

            storage[count].append(syn)



        count += 1

    return storage

# Burda prototiplere en kısa yollar hesaplanarak en ideal synsetler seçiliyor ( her token için)

# Disamb: en kısa yolu oluşturan sysnetler

def disambiguation(storage):

    disamb = []

    for numb in range(len(storage)):

        dist2 = []



        for amb_synset in storage[numb]:

            sum = 0

            temp = vectors[vectors['SynsetID'] == amb_synset]

            temp = temp.reset_index()

            p1 = temp.loc[0].as_matrix(columns=temp.columns[2:])

            for prototype in synsets_of_prototypes:

                temp2 = vectors[vectors['SynsetID'] == prototype]

                temp2 = temp2.reset_index()

                p2 = temp2.loc[0].as_matrix(columns=temp2.columns[2:])

                squared_dist = ((p1 - p2) ** 2)

                dnmm = np.sum(squared_dist)

                dist = np.sqrt(dnmm)

                sum = sum + dist

            dist2.append(sum)

        index = dist2.index(min(dist2))

        disamb.append(storage[numb][index])

    return disamb
num_of_sentence = 0

sequences = []

keys = []

values = []





atis_train = atis_test.head(20)   # İlk 20 cümle

for sentence in atis_train['vocab']:





    tokens = clean_text(sentence)

    storage = find_synsets(tokens)

    disamb = disambiguation(storage)



    sequences.append([])



    # Veriyi Lstme verebilmek için gerekli adımların en başında kelimeleri nümerik sayılar çeviriyorduk

    # Burada bizim verimiz kelime yerine synsetler olduğundan, synsetlerin indexlerini aldım.

    # Örnek want to eat --> ['20', '1564'] (sayılar rastgele)

    for syn in disamb:

        index = vectors.index[vectors['SynsetID'] == syn].tolist()

        synset = vectors.SynsetID[vectors['SynsetID'] == syn]

        sequences[num_of_sentence].extend(index)  # Index bazlı sequenceler, her cümle için

        keys.extend(index)   # Index

        values.extend(synset)  # ve indexin temsil ettiği synset



    num_of_sentence += 1



display(sequences)
data_dict = pd.DataFrame({'keys': keys, 'values': values})

data_dict.drop_duplicates(keep="first",inplace=True)

data_dict.set_index('keys', inplace=True)

word_index = data_dict.to_dict()   # Dict'in indexi atisteki kelimelerin synsetlerinin indexlerini gösteriyor, değerler ise synsetleri



print('Sample sequences and label:', sequences[1], y[1])

print('Vocabulary size:', len(word_index['values']))   # Dictionary vocabları "values" indexinde tutuyor.

data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', y.shape)

display(data)

# Elif'in gönderdiği kodun devamı bu koda uygulanabilir.
indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = y[indices]
num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])

x_train = data[: -num_validation_samples]

y_train = labels[: -num_validation_samples]

x_val = data[-num_validation_samples: ]

y_val = labels[-num_validation_samples: ]



display(x_train)

display(y_train)
print('Tokenized sentences: \n', data[10])

print('One hot label: \n', labels[10])

embeddings_index = {}



for index, row in vectors.iterrows():

    values = row[1:].tolist()

    embeddings_index[index] = np.asarray(values, dtype='float32')



print("Done.\n Proceeding with Embedding Matrix...", end="")
# Aşağıdaki döngüde embedding matrixte vektörler tutuluyor

# ancak normal koddan farklı olarak synsetlerin indexlerini "index_number" listinde tutuyorum

index_number = []

count = 0

embedding_matrix = np.random.random((len(word_index['values']) + 1, EMBEDDING_DIM))

for word, i in word_index['values'].items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[count] = embedding_vector  # Her synsetin vektörü

        index_number.append(word)       # Synsetlerin indexleri

        count += 1

print(" Completed!")
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(word_index['values']) + 1,

                           EMBEDDING_DIM,

                           weights = [embedding_matrix],

                           input_length = MAX_SEQUENCE_LENGTH,

                           trainable=False,

                           name = 'embeddings')

embedded_sequences = embedding_layer(sequence_input)
# Buradan sonrası düzenlenmeli

x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

#preds = Dense(6, activation="sigmoid")(x) #toxix datasetini kullanırken

preds = Dense(1, activation="sigmoid")(x)

model = Model(sequence_input, preds)

model.compile(loss = 'binary_crossentropy',

             optimizer='adam',

             metrics = ['accuracy'])

model.summary()

tf.keras.utils.plot_model(model)

# Burada hata alıyorum

print('Training progress:')

history = model.fit(x_train, y_train, epochs = 2, batch_size=32, validation_data=(x_val, y_val))

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, label='Training loss')

plt.plot(epochs, val_loss, label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show();

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']
plt.plot(epochs, accuracy, label='Training accuracy')

plt.plot(epochs, val_accuracy, label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend()

plt.show();
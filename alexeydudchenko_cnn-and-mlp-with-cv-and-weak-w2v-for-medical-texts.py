# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Keras

from keras import Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, Conv1D, Conv2D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding

from keras import optimizers

import pandas

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score
#get data in words

some_data = pandas.read_csv('../input/diagnoses/database_983_24_2_artifsampl_2.csv',

                            sep=' ; ', encoding = 'utf-8', engine='python',index_col=False)
some_data.head()
labels = some_data.iloc[:,0]

samples = some_data.iloc[:,1]
num_of_diagnoses = len(set(labels))
print(labels[1])



#convert labels to categorical

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

encoder = LabelEncoder()

encoder.fit(labels)

encoded_labels = encoder.transform(labels)

Y_encoded = np_utils.to_categorical(encoded_labels)



print(Y_encoded[1])
#get the vocabulary size

text = ""

for sample in samples:

    text = text + sample



from keras.preprocessing.text import text_to_word_sequence



words = set(text_to_word_sequence(text))

temp_set_of_words = words.copy()



for word in temp_set_of_words:

    if len(word) < 4:

        words.remove(word)

         

vocab_size = len(words)

vocab_size
#Word2Vec load

def words_in_sample(samples):

    """Max number of words in one sample"""

    max_len = 0

    for sample in samples:

        cur_sample = sample.split()

        max_len = len(cur_sample) if len(cur_sample) > max_len else max_len

    return max_len



max_words_in_sample = words_in_sample(samples)

max_words_in_sample
from gensim.models import Word2Vec

#загрузить word2vec

word2vecModel = Word2Vec.load('../input/word2vec-model2/model2.bin')

print(word2vecModel)
import re

def custom_tokenize(samples):

    output_matrix = []

    for s in samples:

        indices = []

        for w in s.split():

            w = re.sub(r'[^\w\s]','',w).lower()

            if w in word2vecModel.wv.vocab:

                indices.append(word2vecModel.wv.vocab[w].index)

        output_matrix.append(indices)

    return output_matrix

    

# Encode docs with our special "custom_tokenize" function

encoded_samples_ge = custom_tokenize(samples)
def get_embedded_samples(samples, word2vecModel, words_in_sample):

    """get word2vec embeddings for given samples and words absent in given word2vec model"""

    new_x = np.zeros((samples.shape[0], word2vecModel.vector_size*words_in_sample))

    absent_words = []

    i = 0

    for sample in samples:

        current_sample = text_to_word_sequence(sample)

        newcur_x = np.zeros((1, word2vecModel.vector_size*max_words_in_sample))

        j = 0

        for word in current_sample:

            if word in word2vecModel.wv.vocab:

                newcur_x[:, j:j+word2vecModel.vector_size] = (word2vecModel[word])

                j += word2vecModel.vector_size

            else:

                absent_words.append(word)

        new_x[i] = newcur_x

        i += 1

    return new_x, absent_words



new_x, absentWords = get_embedded_samples(samples, word2vecModel, max_words_in_sample)
'''define the model'''

def create_network():

    model_MLP = Sequential()

    model_MLP.add(Dense(int(word2vecModel.vector_size*max_words_in_sample*0.8), 

                        input_dim = new_x.shape[1], 

                        activation='relu'))

    model_MLP.add(Dense(300, activation='relu'))

    model_MLP.add(Dense(num_of_diagnoses, activation='softmax'))



    model_MLP.compile(optimizer='adam', 

                      loss='categorical_crossentropy',

                      metrics=['acc'])

    return model_MLP
n_splits = 10

n_epochs = 50
kf = KFold(n_splits=n_splits, shuffle = True, random_state = 2)

kf.get_n_splits(new_x)



f1_score_all = []

for train_index, test_index in kf.split(new_x):

    X_train, X_test = new_x[train_index], new_x[test_index]

    y_train, y_test = Y_encoded[train_index], Y_encoded[test_index]

    

    model_CNN = create_network()



    history_CNN = model_CNN.fit(X_train, y_train,                    

                                verbose=0, 

                                epochs = n_epochs,

                                validation_data = (X_test, y_test))

    

    fig = plt.figure()

    plt.plot(history_CNN.history['acc'])

    plt.plot(history_CNN.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='lower right')

    plt.show()

    

    print("acc ", model_CNN.evaluate(X_test, y_test, verbose = 0)[1])

    pred_cnn = model_CNN.predict_classes(X_test)

    metrica = f1_score(np.argmax(y_test,axis =1), pred_cnn, average='micro')

    print("F1 ", metrica)

    f1_score_all.append(metrica)
sum(f1_score_all)/len(f1_score_all)
#from sklearn.model_selection import train_test_split

#split dataset

#x_train, x_test, y_train, y_test = train_test_split(new_x, Y_encoded, test_size=0.3, random_state=77)
"""convert flat 1000 to 100*100"""

#x_train_CNN = x_train.reshape(x_train.shape[0], 50, 10, 1)

#x_test_CNN = x_test.reshape(x_test.shape[0], 50, 10, 1)

x_CNN = new_x.reshape(new_x.shape[0], 50, 10, 1)
from keras.layers import Flatten, Dropout, Conv2D, MaxPooling2D

from keras.optimizers import RMSprop
def create_network():

    model_CNN = Sequential([



        Conv2D(32, (3, 3), 

               activation='relu', 

               input_shape=(50, 10, 1)),

        MaxPooling2D(pool_size=(2, 2)),

        Dropout(0.25),



        Conv2D(64, (3, 3), activation='relu'),

        MaxPooling2D(pool_size=(2, 2)),

        Dropout(0.25),



        Flatten(),

        Dense(256, activation='relu'),

        Dropout(0.5),

        Dense(num_of_diagnoses, activation='softmax'),])





    model_CNN.compile(optimizer=RMSprop(lr=0.001),

                      loss='categorical_crossentropy',

                      metrics=['accuracy'])

    

    return model_CNN
n_splits = 10

n_epochs = 80



kf = KFold(n_splits=n_splits, shuffle = True, random_state = 2)

kf.get_n_splits(x_CNN)



f1_score_all = []

for train_index, test_index in kf.split(x_CNN):

    X_train, X_test = x_CNN[train_index], x_CNN[test_index]

    y_train, y_test = Y_encoded[train_index], Y_encoded[test_index]

    

    model_CNN = create_network()



    history_CNN = model_CNN.fit(X_train, y_train,                    

                                verbose=0, 

                                epochs = n_epochs,

                                validation_data = (X_test, y_test))

    

    fig = plt.figure()

    plt.plot(history_CNN.history['acc'])

    plt.plot(history_CNN.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='lower right')

    plt.show()

    

    print("acc ", model_CNN.evaluate(X_test, y_test, verbose = 0)[1])

    pred_cnn = model_CNN.predict_classes(X_test)

    metrica = f1_score(np.argmax(y_test,axis =1), pred_cnn, average='micro')

    print("F1 ", metrica)

    f1_score_all.append(metrica)
sum(f1_score_all)/len(f1_score_all)
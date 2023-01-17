!mkdir log

!mkdir model

!mkdir submission
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import log_loss



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding

from keras.layers import Bidirectional

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Dropout

from keras.backend import tensorflow_backend

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.callbacks import CSVLogger



from gensim.models import KeyedVectors



import pandas

import numpy
def build_model(n_word, n_dim, n_hidden, syn0=None, trainable=True):

    model = Sequential()



    if syn0 is not None:

        model.add(Embedding(input_dim=n_word+1, output_dim=n_dim, weights=[syn0], trainable=trainable))

        

    else:

        model.add(Embedding(input_dim=n_word+1, output_dim=n_dim, trainable=trainable))



    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(n_hidden)))    

    model.add(Dense(50))

    model.add(Dense(3, activation='softmax'))

    return model
def build_embedding(n_word, n_dim, pretrain=False):

    syn0 = numpy.random.random((n_word+1, n_dim))



    if pretrain:

        embedding_model = KeyedVectors.load(f'embedding/glove.6B.{n_dim}d')

        for word, index in tokenizer.word_index.items():

            try:

                vector = embedding_model.word_vec(word)

                index = tokenizer.word_index[word]

                syn0[index, :] = vector



            except Exception as e:

                pass



    return syn0
def experiment(n_word, n_dim, n_hidden, pretrain=True, trainable=True, batch_size=128):

    syn0  = build_embedding(n_word, n_dim, pretrain=pretrain)

    

    if pretrain:

        model = build_model(n_word, n_dim, n_hidden, syn0=syn0, trainable=trainable)

    

    else:

        model = build_model(n_word, n_dim, n_hidden, trainable=trainable)



    model_name = f'modelBiLSTM.embedding{n_dim}.n_hidden{n_hidden}.trainable{trainable}.pretrain{pretrain}'  # NOQA

    print()

    print()

    print('# params')

    print(f'n_word     : {n_word}')

    print(f'n_dim      : {n_dim}')

    print(f'n_hidden   : {n_hidden}')

    print(f'pretrain   : {pretrain}')

    print(f'trainable  : {trainable}')

    print(f'destination: {model_name}')

    

    callbacks = []

    callbacks.append(EarlyStopping(patience=3))

    callbacks.append(ModelCheckpoint(filepath=f'model/{model_name}.hdf5', save_best_only=True))

    callbacks.append(CSVLogger(filename=f'log/{model_name}.csv'))



    model.summary()

    model.compile('adam', 'categorical_crossentropy')

    model.fit(X_train, y_train, batch_size=batch_size, epochs=100,

              validation_data=[X_val, y_val], callbacks=callbacks)

    model.load_weights(f'model/{model_name}.hdf5')

    

    y_val_pred_prob = model.predict(X_val, batch_size=batch_size)

    val_loss = log_loss(y_val, y_val_pred_prob)

    

    print(f'{model_name}: {val_loss}')

    

    y_test_prob = model.predict(X_test, batch_size=batch_size)

    test_data['EAP'] = y_test_prob[:, 0]

    test_data['HPL'] = y_test_prob[:, 1]

    test_data['MWS'] = y_test_prob[:, 2]



    test_data[['id', 'EAP', 'HPL', 'MWS']].to_csv(f'submission/{model_name}.csv', index=False)

    

    return 0
train_data = pandas.read_csv('../input/train.csv', index_col=False)

test_data  = pandas.read_csv('../input/test.csv',  index_col=False)
train_data.head()
test_data.head()
all_text = pandas.concat([train_data.text, test_data.text])

n_train = train_data.shape[0]



print(f'n_train: {n_train}')
tokenizer = Tokenizer()

tokenizer.fit_on_texts(all_text)



labelbinarizer = LabelBinarizer()

labelbinarizer.fit(train_data['author'])



X = tokenizer.texts_to_sequences(train_data.text)

X = pad_sequences(X)



y = labelbinarizer.fit_transform(train_data['author'])
X_train, X_val, y_train, y_val = train_test_split(X, y)

n_word   = len(tokenizer.word_index)



print(f'vocaburary size: {n_word}')
X_test = tokenizer.texts_to_sequences(test_data.text)

X_test = pad_sequences(X_test)
experiment(n_word, n_dim=50, n_hidden=50, pretrain=False, trainable=True)
# use pre-trained word-embeddings

experiment(n_word, n_dim=50, n_hidden=50, pretrain=True, trainable=True) # not tested on kernel
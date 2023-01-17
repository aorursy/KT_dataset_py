import os

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# To get reproducible results

np.random.seed(0)

tf.set_random_seed(0)



print(os.listdir("../input"))
test_file = '../input/movie-review-sentiment-analysis-kernels-only/test.tsv'

test = pd.read_csv(test_file, delimiter='\t').fillna('')

x_test = test.values[:, 2]



train_file = '../input/movie-review-sentiment-analysis-kernels-only/train.tsv'

train = pd.read_csv(train_file, delimiter='\t').fillna('')

x_train = train.values[:, 2]

y_train = train.values[:, 3]



print('x_test count: {}'.format(len(x_test)))

print('x_train count: {}'.format(len(x_train)))

print('y_train count: {}'.format(len(y_train)))
from keras.preprocessing import text, sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



max_length = 60

max_features = 20000



x_all = []

x_all.extend(x_test)

x_all.extend(x_train)



tk = Tokenizer(num_words=max_features, lower=True, filters='\n\t')

tk.fit_on_texts(x_all)

x_train_seq = tk.texts_to_sequences(x_train)

x_test_seq = tk.texts_to_sequences(x_test)



np_x_train = pad_sequences(x_train_seq, maxlen=max_length,  padding='post')

np_x_test = pad_sequences(x_test_seq, maxlen=max_length,  padding='post')

np_y_train = to_categorical(y_train)



print ('np_x_train shape: {}'.format(np_x_train.shape))

print ('np_x_test shape: {}'.format(np_x_test.shape))

print ('np_y_train shape: {}'.format(np_y_train.shape))
import tqdm



word_dict = tk.word_index

embedding_dim = 300

embeddings_index = {}



with open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec', 'r') as f:

    lines = f.readlines()



for i in tqdm.tqdm(range(len(lines))):

    values = lines[i].rstrip().rsplit(' ')

    word = values[0]

    embeddings_index[word] = np.asarray(values[1:], dtype='float32')



max_features = min(max_features, len(word_dict) + 1)

embedding_matrix = np.zeros((max_features, embedding_dim))



for word, i in word_dict.items():

    if i >= max_features:

        break



    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector



print('Embedding matrix: {}'.format(embedding_matrix.shape))
from keras.models import Model

from keras.layers import *

from keras.callbacks import EarlyStopping



def one_input_classifier(index, classifier_type, input_length, max_features, class_num, embedding_dim, embedding_matrix):

    inputs = Input(shape=(input_length,), name='input_1')

    embeddings = Embedding(max_features, embedding_dim,

                           weights=[embedding_matrix], input_length=input_length,

                           trainable=False, name='embedding_1')(inputs)

    x = SpatialDropout1D(0.3, name='spatial_dropout1d_1')(embeddings)



    if classifier_type == 'bilstm_cnn':

        x = Bidirectional(CuDNNLSTM(128, name='lstm_1', return_sequences=True), name='bidirectional_1')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Conv1D(128, 5, activation='relu', name='conv1d_1')(x)

        x = Conv1D(128, 3, activation='relu', name='conv1d_2')(x)

        x = Conv1D(128, 1, activation='relu', name='conv1d_3')(x)

        x = Dropout(0.25, name='dropout_2')(x)



        x = GlobalMaxPooling1D(name='global_maxpool1d_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_3')(x)



    elif classifier_type == 'bilstm_cnn_x2':

        x = Bidirectional(CuDNNLSTM(64, name='lstm_1', return_sequences=True), name='bidirectional_1')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Conv1D(64, 5, activation='relu', name='conv1d_1')(x)

        x = Conv1D(64, 4, activation='relu', name='conv1d_2')(x)

        x = Conv1D(64, 3, activation='relu', name='conv1d_3')(x)

        x = Dropout(0.25, name='dropout_2')(x)



        x = Bidirectional(CuDNNLSTM(128, name='lstm_2', return_sequences=True), name='bidirectional_2')(x)

        x = Dropout(0.25, name='dropout_3')(x)

        x = Conv1D(128, 3, activation='relu', name='conv1d_4')(x)

        x = Conv1D(128, 2, activation='relu', name='conv1d_5')(x)

        x = Conv1D(128, 1, activation='relu', name='conv1d_6')(x)

        x = Dropout(0.25, name='dropout_4')(x)



        x = GlobalMaxPooling1D(name='global_maxpool1d_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_5')(x)



    elif classifier_type == 'lstm_cnn':

        x = CuDNNLSTM(128, name='lstm_1', return_sequences=True)(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Conv1D(128, 5, activation='relu', name='conv1d_1')(x)

        x = Conv1D(128, 3, activation='relu', name='conv1d_2')(x)

        x = Conv1D(128, 1, activation='relu', name='conv1d_3')(x)

        x = Dropout(0.25, name='dropout_2')(x)



        x = GlobalMaxPooling1D(name='global_maxpool1d_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_3')(x)



    elif classifier_type == 'cnn_bilstm':

        x = Conv1D(128, 5, activation='relu', name='conv1d_1')(x)

        x = Conv1D(128, 3, activation='relu', name='conv1d_2')(x)

        x = Conv1D(128, 1, activation='relu', name='conv1d_3')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Bidirectional(CuDNNLSTM(128, name='lstm_1', return_sequences=True), name='bidirectional_1')(x)

        x = Dropout(0.25, name='dropout_2')(x)



        x = GlobalMaxPooling1D(name='global_maxpool1d_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_3')(x)



    elif classifier_type == 'cnn_bilstm_x2':

        x = Conv1D(64, 5, activation='relu', name='conv1d_1')(x)

        x = Conv1D(64, 4, activation='relu', name='conv1d_2')(x)

        x = Conv1D(64, 3, activation='relu', name='conv1d_3')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Bidirectional(CuDNNLSTM(64, name='lstm_1', return_sequences=True), name='bidirectional_1')(x)

        x = Dropout(0.25, name='dropout_2')(x)



        x = Conv1D(128, 3, activation='relu', name='conv1d_4')(x)

        x = Conv1D(128, 2, activation='relu', name='conv1d_5')(x)

        x = Conv1D(128, 1, activation='relu', name='conv1d_6')(x)

        x = Dropout(0.25, name='dropout_3')(x)

        x = Bidirectional(CuDNNLSTM(128, name='lstm_2', return_sequences=True), name='bidirectional_2')(x)

        x = Dropout(0.25, name='dropout_4')(x)



        x = GlobalMaxPooling1D(name='global_maxpool1d_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_5')(x)



    elif classifier_type == 'cnn_lstm':

        x = Conv1D(128, 5, activation='relu', name='conv1d_1')(x)

        x = Conv1D(128, 3, activation='relu', name='conv1d_2')(x)

        x = Conv1D(128, 1, activation='relu', name='conv1d_3')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = CuDNNLSTM(128, name='lstm_1', return_sequences=True)(x)

        x = Dropout(0.25, name='dropout_2')(x)



        x = GlobalMaxPooling1D(name='global_maxpool1d_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_3')(x)



    elif classifier_type == 'bilstm_only':

        x = Bidirectional(CuDNNLSTM(128, name='lstm_1'), name='bidirectional_1')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_2')(x)



    elif classifier_type == 'lstm_only':

        x = CuDNNLSTM(128, name='lstm_1')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_2')(x)



    elif classifier_type == 'cnn_only':

        x = Conv1D(128, 5, activation='relu', name='conv1d_1')(x)

        x = Conv1D(128, 3, activation='relu', name='conv1d_2')(x)

        x = Conv1D(128, 1, activation='relu', name='conv1d_3')(x)

        x = Dropout(0.25, name='dropout_1')(x)



        x = GlobalMaxPooling1D(name='global_maxpool1d_1')(x)

        x = Dense(32, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_2')(x)



    elif classifier_type == 'dense_only':

        x = Flatten(name='flatten_1')(x)

        x = Dense(1024, activation='relu', name='dense_1')(x)

        x = Dropout(0.25, name='dropout_1')(x)

        x = Dense(128, activation='relu', name='dense_2')(x)

        x = Dropout(0.25, name='dropout_2')(x)

        x = Dense(32, activation='relu', name='dense_3')(x)

        x = Dropout(0.25, name='dropout_3')(x)



    preds = Dense(class_num, activation='softmax', name='preds')(x)

    model = Model(inputs=inputs, outputs=preds, name='model_{}'.format(index))

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1,

                               mode='min', baseline=None, restore_best_weights=True)



class_num = np_y_train.shape[1]

epochs = 32

batch_size = 1024

validation_split = 0.2

classifier_num = 10



print('Classes: {}'.format(class_num))

print('Epochs: {}'.format(epochs))

print('Batch size: {}'.format(batch_size))

print('Validation split: {:.1}'.format(validation_split))

print('Classifiers: {}'.format(classifier_num))
classifiers = []

classifier_types = ['bilstm_cnn', 'bilstm_cnn_x2', 'lstm_cnn', 'cnn_bilstm', 'cnn_bilstm_x2', 'cnn_lstm', 'bilstm_only', 'lstm_only', 'cnn_only', 'dense_only']



for i in tqdm.tqdm(range(len(classifier_types))):

    classifiers.append(one_input_classifier(i, classifier_types[i], max_length, max_features,

                                            class_num, embedding_dim, embedding_matrix))



for i in range(classifier_num):

    classifiers[i].summary()

    hist = classifiers[i].fit(np_x_train, np_y_train, validation_split=validation_split,

                              callbacks=[early_stopping], epochs=epochs, batch_size=batch_size, verbose=1)

    classifiers[i].trainable = False



    print('{}'.format(classifier_types[i]))

    print('min loss ({}): {:.4}'.format(i, min(hist.history['loss'])))

    print('min val_loss ({}): {:.4}'.format(i, min(hist.history['val_loss'])))

    print('max acc ({}): {:.4}'.format(i, max(hist.history['acc'])))

    print('max val_acc ({}): {:.4}'.format(i, max(hist.history['val_acc'])))



y_pred_list = []



for i in range(classifier_num):

    y_pred = classifiers[i].predict(np_x_test, batch_size=1024, verbose=1)

    y_pred_list.append(y_pred)
test_num = np_x_test.shape[0]

y_pred_class = np.ndarray(shape=(test_num,), dtype=np.int32)



for i in range(test_num):

    votes = []



    for j in range(classifier_num):

        vote = y_pred_list[j][i].argmax(axis=0).astype(int)

        votes.append(vote)



    vote_final = max(set(votes), key=votes.count)

    y_pred_class[i] = vote_final



mapping = {phrase: sentiment for _, _, phrase, sentiment in train.values}



# Overlapping

for i, phrase in enumerate(test.Phrase.values):

    if phrase in mapping:

        y_pred_class[i] = mapping[phrase]



test['Sentiment'] = y_pred_class

test[['PhraseId', 'Sentiment']].to_csv('submission.csv', index=False)

test.head()
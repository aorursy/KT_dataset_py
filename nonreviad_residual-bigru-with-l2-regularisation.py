import numpy as np
import pandas as pd
import keras
from keras.layers import GRU, Bidirectional, Dense, Reshape, Input, Embedding, LSTM, Add, CuDNNGRU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', delimiter='\t')
df.head()
vocab_size = 15000
max_len = 50
embedding_size = 200
hidden_size = 128
tok = keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='-\t\n')
tok.fit_on_texts(df['Phrase'])
import os
from tqdm import tqdm
embeddings_index = {}
with open(os.path.join('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt')) as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
embedding_matrix = np.zeros((vocab_size, embedding_size))
for word, i in tqdm(tok.word_index.items()):
    if i >= vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
X_train = tok.texts_to_sequences(df['Phrase'])
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
y_train = df['Sentiment']
y_train = keras.utils.to_categorical(y_train, num_classes=5)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8)
def create_trained_residual_model(lambda_reg = 0.01, regulariser_type = keras.regularizers.l2, weights_filename = "gru.hd5", cell_type = GRU, num_epochs=40):
    regulariser = regulariser_type(lambda_reg)
    input_layer = Input(shape=(max_len,))
    save_cb = ModelCheckpoint(filepath=weights_filename,monitor='val_acc', save_best_only=True, save_weights_only=True)
    embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix])(input_layer)

    recurrent_layer_one = Bidirectional(cell_type(hidden_size, return_sequences=True, kernel_regularizer=regulariser))(embedding_layer)
    recurrent_layer_two = Bidirectional(cell_type(hidden_size, return_sequences=True, kernel_regularizer=regulariser))(recurrent_layer_one)
    merged_one = Add()([recurrent_layer_two, recurrent_layer_one]) # Residual link

    recurrent_layer_three = Bidirectional(cell_type(hidden_size, return_sequences=True, kernel_regularizer=regulariser))(merged_one)
    recurrent_layer_four = Bidirectional(cell_type(hidden_size, return_sequences=True, kernel_regularizer=regulariser))(recurrent_layer_three)
    recurrent_layer_last = Add()([recurrent_layer_three, recurrent_layer_four]) # Residual link

    flattened_layer = Reshape((-1,))(recurrent_layer_last)
    output_layer = Dense(5, activation='softmax', kernel_regularizer=regulariser)(flattened_layer)
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, batch_size=512, epochs=num_epochs,validation_data=(X_val,y_val),callbacks=[save_cb])
    model.load_weights('./' + weights_filename)
    return model

# Ignore this
# def create_trained_simple_bidirectional_model(lambda_reg = 0.01, regulariser_type = keras.regularizers.l2, weights_filename = "gru.hd5", cell_type = GRU, num_epochs=40):
#     regulariser = regulariser_type(lambda_reg)
#     input_layer = Input(shape=(max_len,))
#     save_cb = ModelCheckpoint(filepath=weights_filename,monitor='val_acc', save_best_only=True, save_weights_only=True)
#     embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix])(input_layer)
#     recurrent_layer = Bidirectional(cell_type(hidden_size, return_sequences=True, kernel_regularizer=regulariser))(embedding_layer)
#     flattened_layer = Reshape((-1,))(recurrent_layer)
#     output_layer = Dense(5, activation='softmax', kernel_regularizer=regulariser)(flattened_layer)
#     model = Model(input_layer, output_layer)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#     model.fit(X_train, y_train, batch_size=512, epochs=num_epochs,validation_data=(X_val,y_val),callbacks=[save_cb])
#     model.load_weights('./' + weights_filename)
#     return model

# def create_trained_simple_model(lambda_reg = 0.01, regulariser_type = keras.regularizers.l2, weights_filename = "gru.hd5", cell_type = GRU, num_epochs=40):
#     regulariser = regulariser_type(lambda_reg)
#     input_layer = Input(shape=(max_len,))
#     save_cb = ModelCheckpoint(filepath=weights_filename,monitor='val_acc', save_best_only=True, save_weights_only=True)
#     embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix])(input_layer)
#     recurrent_layer = cell_type(hidden_size, kernel_regularizer=regulariser)(embedding_layer)
#     output_layer = Dense(5, activation='softmax', kernel_regularizer=regulariser)(recurrent_layer)
#     model = Model(input_layer, output_layer)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#     model.fit(X_train, y_train, batch_size=512, epochs=num_epochs,validation_data=(X_val,y_val),callbacks=[save_cb])
#     model.load_weights('./' + weights_filename)
#     return model
# model_lstm = create_trained_simple_model(lambda_reg = 0.05, regulariser_type = keras.regularizers.l2, weights_filename = "lstm.hd5", cell_type = LSTM, num_epochs=30)
model_gru = create_trained_residual_model(lambda_reg = 0.1, regulariser_type = keras.regularizers.l2, weights_filename = "gru.hd5", cell_type = CuDNNGRU, num_epochs=30)
# model_bilstm = create_trained_simple_bidirectional_model(lambda_reg = 0.05, regulariser_type = keras.regularizers.l2, weights_filename = "bilstm.hd5", cell_type = LSTM, num_epochs=30)
# y_pred = y_pred1 = model_lstm.predict(X_val)

# y_pred1 = np.argmax(y_pred1, axis=1)
# y_pred1 = to_categorical(y_pred1, num_classes=5)
# print("LSTM accuracy: {}".format(accuracy_score(y_pred1, y_val)))

# y_pred2 = model_gru.predict(X_val)
# y_pred += y_pred2
# y_pred2 = np.argmax(y_pred2, axis=1)
# y_pred2 = to_categorical(y_pred2, num_classes=5)
# print("Residual BiGRU accuracy: {}".format(accuracy_score(y_pred2, y_val)))

# y_pred3 = model_bilstm.predict(X_val)
# y_pred += y_pred3
# y_pred3 = np.argmax(y_pred3, axis=1)
# y_pred3 = to_categorical(y_pred3, num_classes=5)
# print("BiLSTM accuracy: {}".format(accuracy_score(y_pred3, y_val)))

# y_pred = np.argmax(y_pred, axis=1)
# y_pred = to_categorical(y_pred, num_classes=5)
# print("Mixture accuracy: {}".format(accuracy_score(y_pred, y_val)))
df_test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', delimiter='\t')
X_test = tok.texts_to_sequences(df_test['Phrase'])
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')
# y_test  = model_lstm.predict(X_test)
y_test = model_gru.predict(X_test)
# y_test += model_bilstm.predict(X_test)
y_test = np.argmax(y_test, axis=1)
df_out = pd.DataFrame(data={'PhraseId':df_test['PhraseId'], 'Sentiment':y_test})
df_out.to_csv('submission.csv', index=False)
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN, Bidirectional, LSTM, Activation, SpatialDropout1D
from keras.optimizers import Adam
from keras.datasets import imdb
from keras.callbacks import  TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from keras.regularizers import l2
max_features = 5000
no_classes = 1
max_length = 500
batch_size = 32
embedding_size = 32
dropout_rate = 0.4
no_epochs = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print(x_train[7])
print(y_train[7])
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[7]])
print(decoded_review)
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

print(x_train[7])
print(x_train.shape)
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
print(y_train.shape)
print(y_test.shape)
x_val = x_train[:5000]
partial_x_train = x_train[5000:]
y_val = y_train[:5000]
partial_y_train = y_train[5000:]

print(x_val.shape)
print(partial_x_train.shape)
print(y_val.shape)
print(partial_y_train.shape)
LSTM_model = Sequential()
LSTM_model.add(Embedding(max_features, embedding_size, input_length=max_length))
LSTM_model.add(SpatialDropout1D(dropout_rate))
LSTM_model.add(Bidirectional(LSTM(16,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),dropout=0.2, recurrent_dropout=0.2)))
LSTM_model.add(Dense(16,activation="relu", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
LSTM_model.add(Dropout(dropout_rate))
LSTM_model.add(Activation('relu'))
LSTM_model.add(Dropout(dropout_rate))
LSTM_model.add(Dense(1))

LSTM_model.add(Dense(no_classes, activation='sigmoid'))
LSTM_model.summary()

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)
LSTM_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard('./logs/SimpleRNN')
LSTM_model.save("myRNNSimplemodel.h5")

history = LSTM_model.fit(partial_x_train, partial_y_train, batch_size=batch_size, verbose=2, epochs=no_epochs, validation_data=(x_val, y_val), shuffle=True,callbacks = [tensorboard])
results = LSTM_model.evaluate(x_test, y_test)
plt.clf()
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, (len(history_dict['accuracy']) + 1))
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
prediction = LSTM_model.predict(x_test)
y_pred = (prediction > 0.5)

from sklearn.metrics import f1_score


print('F1-score: {0}'.format(f1_score(y_pred, y_test)))


import pandas as pd

pred = np.round(y_pred).astype(int)

pred = np.array(pred)
target = np.array(y_test)

df = pd.DataFrame(list(zip(target, pred)), columns = ['Target', 'Pred']) 
df.to_csv('final_preds.csv', index=False)
df.tail()


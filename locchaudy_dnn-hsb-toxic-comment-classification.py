import sys, os, re, csv, codecs, numpy as np, pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, SimpleRNN, TimeDistributed, ConvLSTM2D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import plot_model
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

print(train.shape)
print(test.shape)
train.head()
test.head()
train['comment_text'][0]
train['comment_text'][1689]
test['comment_text'][4]
test['comment_text'][0]
train.isnull().any().sum()

test.isnull().any().sum()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
index = tokenizer.word_index
print(index['the'])
print(index['cat'])
print(index['cars'])

len(list_tokenized_train[1])
len(list_tokenized_train[250])
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))
plt.axvline(x=200, color='gray')
plt.show()
maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
print(X_t[0])
print(len(X_t[0]))
# For the Input 
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

# != layers 

x = Dense(6, activation="tanh")(x)
x = Dense(6, activation="tanh")(x)


# For the Output
x = GlobalMaxPool1D()(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, dpi=96)
batch_size = 64
epochs = 5
history = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
history_dict = history.history
history_dict.keys()

acc_basic = history.history['accuracy']
val_loss_basic = history.history['val_loss']
loss_basic = history.history['loss']
val_acc_basic = history.history['val_accuracy']

epochs = range(1, len(acc_basic) + 1)

plt.subplot(221)
plt.plot(epochs, loss_basic, 'b', label='Training loss')
plt.plot(epochs, val_loss_basic, 'g', label='Validation loss')

plt.title('Training vs validation (loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(224)

plt.plot(epochs, acc_basic, 'b', label='Training accuracy')
plt.plot(epochs, val_acc_basic, 'red', label='Validation accuracy')

plt.title('Training vs Validation (accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
score, test_acc_basic = model.evaluate(X_t,y,verbose =1)
print(test_acc_basic)
y_pred = model.predict(X_te, verbose = 1)
submission = pd.DataFrame(columns=['id'] + list_classes)
submission['id'] = test['id'].values 
submission[list_classes] = y_pred
submission.to_csv("./submission_basicmodel.csv", index=False)
# For the Input 
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

# != layers 


x = SimpleRNN(6,return_sequences=True)(x)
x = Dense(6, activation="sigmoid")(x)


# For the Output
x = GlobalMaxPool1D()(x)
model_rnn = Model(inputs=inp, outputs=x)
model_rnn.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
plot_model(model_rnn, to_file='model_gru.png', show_shapes=True, show_layer_names=True, dpi=96)
batch_size = 64
epochs = 5
history = model_rnn.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
history_dict = history.history
history_dict.keys()

acc_rnn = history.history['accuracy']
loss_rnn = history.history['loss']
val_acc_rnn = history.history['val_accuracy']
val_loss_rnn = history.history['val_loss']

epochs = range(1, len(acc_rnn) + 1)

plt.subplot(221)
plt.plot(epochs, loss_rnn, 'b', label='Training loss')
plt.plot(epochs, val_loss_rnn, 'g', label='Validation loss')

plt.title('Training vs validation (loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(224)

plt.plot(epochs, acc_rnn, 'b', label='Training accuracy')
plt.plot(epochs, val_acc_rnn, 'red', label='Validation accuracy')

plt.title('Training vs Validation (accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
score, test_acc_rnn = model_rnn.evaluate(X_t,y,verbose =1)
y_pred = model_rnn.predict(X_te, verbose = 1)
submission = pd.DataFrame(columns=['id'] + list_classes)
submission['id'] = test['id'].values 
submission[list_classes] = y_pred
submission.to_csv("/kaggle/working/submission_model_RNN.csv", index=False)
# For the Input 
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

# != layers 

x = LSTM(6, return_sequences=True)(x)
x = Dense(6, activation="sigmoid")(x)


# For the Output
x = GlobalMaxPool1D()(x)
model_lstm = Model(inputs=inp, outputs=x)
model_lstm.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
plot_model(model_lstm, to_file='model_lstm.png', show_shapes=True, show_layer_names=True, dpi=96)
batch_size = 64
epochs = 5
history = model_lstm.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
history_dict = history.history
history_dict.keys()

acc_lstm = history.history['accuracy']
val_acc_lstm = history.history['val_accuracy']
val_loss_lstm = history.history['val_loss']
loss_lstm = history.history['loss']

epochs = range(1, len(acc_lstm) + 1)

plt.subplot(221)
plt.plot(epochs, loss_lstm, 'b', label='Training loss')
plt.plot(epochs, val_loss_lstm, 'g', label='Validation loss')

plt.title('Training vs validation (loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(224)

plt.plot(epochs, acc_lstm, 'b', label='Training accuracy')
plt.plot(epochs, val_acc_lstm, 'red', label='Validation accuracy')

plt.title('Training vs Validation (accuracy)')
plt.xlabel('Accuracy')
plt.ylabel('Loss')
plt.legend()

plt.show()
score, test_acc_lstm = model_lstm.evaluate(X_t,y,verbose =1)
test_acc_lstm
y_pred = model_lstm.predict(X_te, verbose = 1)
submission = pd.DataFrame(columns=['id'] + list_classes)
submission['id'] = test['id'].values 
submission[list_classes] = y_pred
submission.to_csv("./submission_model_LSTM.csv", index=False)
# For the Input 
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

# != layers 

x = GRU(10,return_sequences=True, activation="sigmoid")(x)
x = Dense(6, activation="sigmoid")(x)


# For the Output
x = GlobalMaxPool1D()(x)
model_gru = Model(inputs=inp, outputs=x)
model_gru.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
plot_model(model_gru, to_file='model_gru.png', show_shapes=True, show_layer_names=True, dpi=96)
batch_size = 64
epochs = 5
history = model_gru.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
history_dict = history.history
history_dict.keys()

acc_gru = history.history['accuracy']
loss_gru = history.history['loss']
val_acc_gru = history.history['val_accuracy']
val_loss_gru = history.history['val_loss']

epochs = range(1, len(acc_gru) + 1)

plt.subplot(221)
plt.plot(epochs, loss_gru, 'b', label='Training loss')
plt.plot(epochs, val_loss_gru, 'g', label='Validation loss')

plt.title('Training vs validation (loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(224)

plt.plot(epochs, acc_gru, 'b', label='Training accuracy')
plt.plot(epochs, val_acc_gru, 'red', label='Validation accuracy')

plt.title('Training vs Validation (accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
score, test_acc_gru = model_gru.evaluate(X_t,y,verbose =1)
test_acc_gru
y_pred = model_gru.predict(X_te, verbose = 1)
submission = pd.DataFrame(columns=['id'] + list_classes)
submission['id'] = test['id'].values 
submission[list_classes] = y_pred
submission.to_csv("./submission_model_GRU.csv", index=False)
comp = pd.DataFrame(columns = ['Models'] + ['Training Accuracy (avg)'] + ['Validation Accuracy (avg)'] + ['Test Accuracy'] + ['Time /epochs (avg)']+ ['Kaggle Score'])
comp['Models'] = ['Baseline',"LSTM",'GRU','SimpleRNN']
comp['Training Accuracy (avg)'] = [mean(acc_basic),mean(acc_lstm),mean(acc_gru),mean(acc_rnn)]
comp['Validation Accuracy (avg)'] = [mean(val_acc_basic), mean(val_acc_lstm), mean(val_acc_gru), mean(val_acc_rnn)]
comp['Test Accuracy'] = [test_acc_basic, test_acc_lstm, test_acc_gru, test_acc_rnn]
comp['Time /epochs (avg)'] = ["54s", "208s", "246.2s", "151.4s"]
comp['Kaggle Score'] = ["0.89238","0.92040","0.95050","0.91707"]
comp.to_csv("./Models_comp.csv",index = False)
displaycomp = pd.read_csv('/kaggle/working/Models_comp.csv')
print(displaycomp)
predict = pd.read_csv('/kaggle/working/submission_model_GRU.csv')
predict_val = predict[list_classes].values
test['comment_text'][4]
predict_val[4]
test['comment_text'][0]
predict_val[0]
import numpy as np

import pandas as pd
data_dir = '../input/hate_speech.csv'

data = pd.read_csv(data_dir)
texts = np.array(data['post'])

labels = np.array(data['label'])
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



maxlen = 100

samples = texts.shape[0]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index



data = pad_sequences(sequences, maxlen)

print(data.shape)

print(labels.shape)
x_train = data[:7925]

y_train = labels[:7925]



x_val = data[7925:8925]

y_val = labels[7925:8925]



x_test = data[8925:]

y_test = labels[8925:]



print(x_train.shape)

print(x_val.shape)

print(x_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Embedding, SimpleRNN, LSTM, GRU, Bidirectional

model = Sequential()

model.add(Embedding(1+len(word_index), 16))

model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.5, return_sequences=True)))

model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.5)))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
def cnfmatrix(y_test,results):

    fp = 0.0

    fn = 0.0

    tp = 0.0

    tn = 0.0

    t = 0.0

    n = 0.0

    results.shape

    for i in range(results.shape[0]):

        if y_test[i]==1 and results[i]==1:

            tp+=1

            t+=1

        elif y_test[i]==1 and results[i]==0:

            fn+=1

            t+=1

        elif y_test[i]==0 and results[i]==1:

            fp+=1

            n+=1

        elif y_test[i]==0 and results[i]==0:

            tn+=1

            n+=1

    print(tp/results.shape[0],fp/results.shape[0])

    print(fn/results.shape[0],tn/results.shape[0])

    Precision  = tp/(tp+fp)

    Recall = tp/(tp+fn)

    print("Precision: ",Precision,"Recall: ",Recall)

    f1score = (2*Precision*Recall)/(Precision+Recall)

    print("f1score: ",f1score)

    print("accuracy: ",(tp+tn)/results.shape[0])

    print("hate_acc: ", (tp)/t)

    print("non_hate_acc: ", (tn)/n)
predictions = model.predict(x_test)
results = []

for prediction in predictions:

    if prediction < 0.5:

        results.append(0)

    else:

        results.append(1)

        

results = np.array(results)
cnfmatrix(y_test, results)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
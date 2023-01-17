import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import confusion_matrix



from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling1D

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.models import model_from_json

import re

import warnings 

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/Reviews.csv')

data.columns
data = data[['Text','Score']]
data.head()
row_select_1 = data['Score'] < 3

row_select_2 = data['Score'] == 3

data['sentiment'] = pd.Series(['Positive']*len(data.index))

data.loc[row_select_1,'sentiment'] = 'Negative'

data.loc[row_select_2,'sentiment'] = 'Neutral'
data = data[['Text','sentiment']]

data.columns = ['text', 'sentiment']
# Embedding

max_features = 20000

maxlen = 100

embedding_size = 128



# Convolution

kernel_size = 5

filters = 64

pool_size = 4



# LSTM

lstm_output_size = 70



# Training

batch_size = 30

epochs = 2
data = data[data.sentiment != "Neutral"]

data['text'] = data['text'].apply(lambda x: str(x).lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))



print(data[ data['sentiment'] == 'Positive'].size)

print(data[ data['sentiment'] == 'Negative'].size)



for idx,row in data.iterrows():

    row[0] = row[0].replace('rt',' ')

    

tokenizer = Tokenizer(nb_words = max_features, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X, maxlen = maxlen)
Y = pd.get_dummies(data['sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
print('Build model...')



model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))

model.add(Dropout(0.25))

model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))

model.add(MaxPooling1D(pool_size=pool_size))

model.add(LSTM(lstm_output_size))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
print('Train...')

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,shuffle=True, validation_data=(X_test, Y_test))
yhat = model.predict(X_test, verbose = 2, batch_size = batch_size)

from sklearn import metrics

print(metrics.classification_report(Y_test[:,1], np.round(yhat[:,1]) ,target_names = ["negative", "positive"]))
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



score = ['negative', 'positive']



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greys):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(set(score)))

    plt.xticks(tick_marks, score, rotation=45)

    plt.yticks(tick_marks, score)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

# Compute confusion matrix

cm = confusion_matrix(Y_test[:,1], np.round(yhat[:,1]))

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm)    



cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure()

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')



plt.show()
# serialize model to JSON

model_json = model.to_json()

with open("model_conv_lstm.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model_conv_lstm.h5")
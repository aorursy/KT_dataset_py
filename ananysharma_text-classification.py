# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
x_train_word = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
x_test_word = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")
list_classes = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
y_train = x_train_word[list_classes].values
x_train = x_train_word["comment_text"]
x_test = x_test_word["content"]
# x_test1 = []
# from googletrans import Translator
# translator = Translator()
# for i in range (0,len(x_test)):
#     sent = translator.translate(x_test[i])
#     x_test1.append(sent.text)
    
# print(x_test1[0])
# # print(translator.translate("Doctor"))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False)

tokenizer.fit_on_texts(list(x_train))
tokenized_train = tokenizer.texts_to_sequences(x_train)
tokenized_test = tokenizer.texts_to_sequences(x_test)
index = tokenizer.word_index
average = np.mean([len(seq) for seq in tokenized_train])
stdev = np.std([len(seq) for seq in tokenized_train])
max_len = int(average + stdev * 3)
print()
processed_X_train = pad_sequences(tokenized_train, maxlen=max_len, padding='post', truncating='post')
processed_X_test = pad_sequences(tokenized_test, maxlen=max_len, padding='post', truncating='post')
embeddings_index = {}
f = open(os.path.join('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
Embedding_dim = 200
embedding_matrix = np.zeros((len(index) + 1, Embedding_dim))
for word, i in index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
from keras.layers import Embedding

layer = Embedding(len(index)+1,Embedding_dim,weights = [embedding_matrix],input_length =max_len,trainable = True)
import keras.backend
from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense, Conv1D, MaxPooling1D
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Nadam


model = Sequential()
model.add(layer)
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(3))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())


model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='sigmoid'))

model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint
def loss(y_true,y_pred):
    return keras.backend.binary_crossentropy(y_true,y_pred)


saved_model = "weights_base.best.hdf5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]
import tensorflow as tf
p = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(p[0], 'GPU')
from sklearn.model_selection import train_test_split
[X, X_val, y, y_val] = train_test_split(processed_X_train, y_train, test_size=0.03, shuffle=False)
model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])
history = model.fit(X, y, batch_size=1280, epochs=3,validation_data=(X_val, y_val),verbose=1,shuffle=True)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['train','test'],loc = 'best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['train','test'],loc = 'best')
plt.show()
def predi(string):
    newString = [string]
    newString = tokenizer.texts_to_sequences(newString)
    newString = pad_sequences(newString,maxlen = max_len,padding = "post",truncating = "post")
    prediction = model.predict(newString)
#     toxic = float(prediction[0][0])
#     if toxic>0.5:
#         return 1.0
#     else:
#         return 0.0
    
    print("Severe_Toxic % is {:.0%}".format(prediction[0][1]))
    print("Obscene % is {:.0%}".format(prediction[0][2]))
    print("Threat % is {:.0%}".format(prediction[0][3]))
    print("Insult % is {:.0%}".format(prediction[0][4]))
    print("Identity Hate % is {:.0%}".format(prediction[0][5]))      
sample  = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
sample.head
# predic = []
# for sent in x_test:/

#     predic.append(predi(sent))
predic = model.predict(processed_X_test)
output = pd.DataFrame({"Id":x_test_word.id,"toxic":predic[0][1]})
output.to_csv('my_submission.csv',index = False)

# data = { 'Loss' : hist.history['loss'],
#         'val_loss': hist.history['val_loss'],
#        'train_accuracy': hist.history['acc']}
# # df = pd.read_csv("/output/kaggle/working/my_submissions.csv")
# # x_test[3]
# print(data)
model.evaluate(X,y)
predi("kill you")

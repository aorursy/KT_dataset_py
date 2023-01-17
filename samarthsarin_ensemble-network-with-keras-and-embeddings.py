import numpy as np

import pandas as pd 

import os

print(os.listdir("../input"))



from string import punctuation

from keras.models import Sequential,Model

from keras.layers import Embedding,Input,Activation,Flatten,CuDNNLSTM,Dense,Dropout,Bidirectional,LSTM,MaxPool1D

from keras.layers import Convolution1D,GlobalAveragePooling1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import LeakyReLU

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import matplotlib.pyplot as plt

import re

import gc

import seaborn as sns

%matplotlib inline

tqdm.pandas()
df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
df.head()
def target(value):

    if value>=0.5:

        return 1

    else:

        return 0
df['target'] = df['target'].apply(target)
x = df['comment_text']

y = df['target']
sns.countplot(y)
def cleaning(text):

    text = text.lower()

    text = re.sub(r'\W+',' ',text)

    return text
x = x.progress_apply(cleaning)
f = open('../input/gloveembeddings/glove.6B.100d.txt')

embedding_values = {}

for line in tqdm(f):

    value = line.split(' ')

    word = value[0]

    coef = np.array(value[1:],dtype = 'float32')

    embedding_values[word]=coef
token = Tokenizer()
token.fit_on_texts(x)
sequence = token.texts_to_sequences(x)
len(sequence)
vocab_size = len(sequence)+1
pad_seq = pad_sequences(sequence,maxlen = 100)
all_emb = np.stack(embedding_values.values())

all_mean,all_std = all_emb.mean(),all_emb.std()

all_mean,all_std
embedding_matrix = np.random.normal(all_mean,all_std,(vocab_size,100))

for word,i in tqdm(token.word_index.items()):

    values = embedding_values.get(word)

    if values is not None:

        embedding_matrix[i] = values
model1 = Sequential()
model1.add(Embedding(vocab_size,100,input_length = 100,weights = [embedding_matrix],trainable = False))
model1.add(Bidirectional(CuDNNLSTM(100,return_sequences=True)))

model1.add(Convolution1D(64,7,padding='same'))

model1.add(GlobalAveragePooling1D())
model1.add(Dense(128))

model1.add(LeakyReLU())
model1.add(Dense(64,activation = 'relu'))
model1.add(Dense(1,activation = 'sigmoid'))
model1.compile(optimizer = 'adam',loss='binary_crossentropy',metrics = ['accuracy'])
x_train,x_test,y_train,y_test = train_test_split(pad_seq,y,test_size = 0.15,random_state = 42)
del (x,y)

gc.collect()
history = model1.fit(x_train,y_train,epochs = 5,batch_size=128,validation_data=(x_test,y_test))
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['loss']

training_loss = values['val_loss']

epochs = range(5)
plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
test.head()
X = test['comment_text']
test_sequence = token.texts_to_sequences(X)

test_pad_seq = pad_sequences(test_sequence,maxlen = 100)
prediction1 = model1.predict(test_pad_seq)
submission1 = pd.DataFrame([test['id']]).T

submission1['prediction'] = prediction1
submission1.to_csv('submission.csv', index=False)
# model2 = Sequential()

# model2.add(Embedding(vocab_size,100,input_length=100,weights = [embedding_matrix],trainable = False))

# model2.add(CuDNNLSTM(75,return_sequences=True))

# model2.add(CuDNNLSTM(75))

# model2.add(Dense(128,activation='relu'))

# model2.add(Dropout(0.3))

# model2.add(Dense(1,activation='sigmoid'))

# model2.compile(optimizer= 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# history = model2.fit(x_train,y_train,batch_size = 128,epochs = 5,validation_data = (x_test,y_test))
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['loss']

training_loss = values['val_loss']

epochs = range(5)
plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
prediction2 = model2.predict(test_pad_seq)

submission2 = pd.DataFrame([test['id']]).T

submission2['prediction'] = prediction2

submission2.to_csv('submission_model2.csv', index=False)
# model3 = Sequential()

# model3.add(Embedding(vocab_size,100,input_length=100,weights = [embedding_matrix],trainable = False))

# model3.add(Convolution1D(32,5,activation='relu'))

# model3.add(MaxPool1D(2,2))

# model3.add(Convolution1D(64,5,activation='relu'))

# model3.add(MaxPool1D(2,2))

# model3.add(Flatten())

# model3.add(Dense(128,activation='relu'))

# model3.add(Dropout(0.2))

# model3.add(Dense(1,activation='sigmoid'))

# model3.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# history = model3.fit(x_train,y_train,batch_size = 128,epochs = 5,validation_data = (x_test,y_test))
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['loss']

training_loss = values['val_loss']

epochs = range(5)
plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
prediction3 = model3.predict(test_pad_seq)

submission3 = pd.DataFrame([test['id']]).T

submission3['prediction'] = prediction3

submission3.to_csv('submission_model3.csv', index=False)
prediction4 = 0.33*submission1['prediction']+0.33*submission2['prediction']+0.33*submission3['prediction']

submission4 = pd.DataFrame([test['id']]).T

submission4['prediction'] = prediction4

submission4.to_csv('submission4.csv', index=False)
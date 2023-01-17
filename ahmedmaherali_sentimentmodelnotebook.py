import pandas as pd 

df=pd.read_csv("../input/20190928-reviews.csv")
df.dropna(subset=["title"],axis=0,inplace=True)

df.dropna(subset=["rating"],axis=0,inplace=True)

df.reset_index(drop=True,inplace=True)
data=[]

labels=[]
for index, row in df.iterrows():

    if row['verified']==True:

        print(row['title'])

        data.append(row['title'])

        labels.append(row['rating'])

data
!pip install nltk

print("fi")

!pip install tensorflow

print("fins")

!pip install keras

print("finshed")

!python -m nltk.downloader all
from sklearn.model_selection import train_test_split

 

train_data=data[0:round(len(data)*0.8)]

test_data=data[round(len(data)*0.8):]

train_labels=labels[0:round(len(data)*0.8)]

test_labels=labels[round(len(data)*0.8):]

train_data
test_data.reset_index(drop=True,inplace=True)

train_data.reset_index(drop=True,inplace=True)

train_labels.reset_index(drop=True,inplace=True)

test_labels.reset_index(drop=True,inplace=True)

test_labels
test_labels
import string

newData=[]

for review in train_data:

    

    word_tokens = word_tokenize(review)

    s=""

    for w in word_tokens :

        if not w in stop_words:

            lemmed=lem.lemmatize(w,"v")

            s+=(lemmed.lower()+' ')

            s=s.translate(str.maketrans('', '', string.punctuation))

    newData.append(s)

print("FIN")
newData


stop_words = set(stopwords.words('english'))

lem=WordNetLemmatizer()

newTestData=[]

for review in test_data:

    

    word_tokens = word_tokenize(review)

    s=""

    for w in word_tokens :

        if not w in stop_words:

            lemmed=lem.lemmatize(w,"v")

            s+=(lemmed.lower()+' ')

            s=s.translate(str.maketrans('', '', string.punctuation))

    newTestData.append(s)

print("FIN")
newTestData
train_data
from keras.utils import to_categorical

train_labels=to_categorical(train_labels)

train_labels
test_labels=to_categorical(test_labels,num_classes=6)
test_labels
train_data=np.array(newData)

test_data=np.array(newTestData)

train_labels=np.array(train_labels)

test_labels=np.array(test_labels)

  
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout


tokenizer = Tokenizer(num_words=len(train_data))

tokenizer.fit_on_texts(train_data)
import pickle



# saving

with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



# loading

#with open('tokenizer.pickle', 'rb') as handle:

#    tokenizer = pickle.load(handle)


 

x_train = tokenizer.texts_to_matrix(train_data, mode='tfidf')

x_test = tokenizer.texts_to_matrix(test_data, mode='tfidf')

 

model = Sequential()

model.add(Dense(512, input_shape=(len(x_train),)))

model.add(Activation('relu'))

model.add(Dropout(0.3))





model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.3))





model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.3))



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.3))





model.add(Dense(6))

model.add(Activation('softmax'))

model.summary()

 

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

 

history = model.fit(x_train, train_labels,

                    epochs=5,

                    verbose=1,

                    validation_data=(x_test,test_labels)

                    )
loss,acc=model.evaluate(x_test,test_labels)

print("loss: ",loss)

print("acc : ",acc)
model.save('/kaggle/working/mostAccDense.h5') 
import numpy as np

testrevs=[]

testrevs.append("I currently have samsung note 2, its really a sliod phone and intersting i love it")

testrevs.append("I have samsung s20")

testrevs.append("I currently have infinix note2, its battery cant work hardly i suffer with its software")

testrevs.append("i hate samsung")

testrevs.append("the phone shape ok")

testrevs.append("ok")

testrevs=np.array(testrevs)

x_test = tokenizer.texts_to_matrix(testrevs, mode='tfidf')

predictions=model.predict(x_test)

for pred in predictions:

  print(np.argmax(pred))


stop_words = set(stopwords.words('english'))

lem=WordNetLemmatizer()

test=[]

i=0

for review in data:

    

    word_tokens = word_tokenize(review)

    s=""

    for w in word_tokens :

        if not w in stop_words:

            lemmed=lem.lemmatize(w,"v")

            s+=(lemmed.lower()+' ')

    test.append(s)

    if i==100:

        break

    i+=1

print("FIN")


 

test = tokenizer.texts_to_matrix(test, mode='tfidf')



test
prediction=model.predict(test)

i=0

for p in prediction:

    print(np.argmax(p),labels[i])

    i+=1
from __future__ import print_function

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling1D

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

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



print('Build model...')

model = Sequential()

model.add(Embedding(max_features, 512))

model.add(Dropout(0.25))



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.3))



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.3))





model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(6))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

print('Train...')

model.fit(x_train, train_labels,batch_size=batch_size,epochs=3,validation_data=(x_test, test_labels))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)



print('Test score:', score)

print('Test accuracy:', acc)

model.save('LSTMPlusDense81.h5') 
totest=data[0]
from __future__ import print_function



from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling1D



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

epochs = 5







print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



print('Build model...')



model = Sequential()

model.add(Embedding(max_features, 128))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))



model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.3))





model.add(Dense(5))

model.add(Activation('softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



print('Train...')

model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)

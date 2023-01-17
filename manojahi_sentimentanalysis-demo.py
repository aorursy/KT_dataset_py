import pandas as pd
test_data = pd.read_csv('../input/train.tsv', sep='\t')

#print(test_data)

#pd.read_csv('data/train.tsv', sep='\t')

testdf = test_data.values



Xtrain = testdf[:,2] #This will have all rows with index 2 col (Our reviews)

print(Xtrain)



ytrain = testdf[:,3] #This will all rows with index 3 col (reviews label)

#print(ytrain)
from numpy import array

from numpy import asarray

from numpy import zeros

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Embedding
t = Tokenizer() #Technique to convert text list to text list index
t.fit_on_texts(Xtrain) 
vocab_size = len(t.word_index) + 1

print(vocab_size)
# integer encode the documents

encoded_docs = t.texts_to_sequences(Xtrain)

print(encoded_docs[0])
item = max(Xtrain, key=len)

print(len(item))
# pad documents to a max length of 4 words

max_length = 20

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_docs)
# load the whole embedding into memory

embeddings_index = dict()

f = open('../input/glove.6B.100d.txt', encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs

embedding_matrix = zeros((vocab_size, 100))

for word, i in t.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# define model

#model = Sequential()

#e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)

#model.add(e)

#model.add(Flatten())

#model.add(Dense(1, activation='sigmoid'))

# compile the model

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

#print(model.summary())



model = Sequential()

e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=False)

model.add(e)

model.add(Flatten())

model.add(Dense(32, input_dim=20, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(5, activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(ytrain)
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils



# encode class values as integers

encoder = LabelEncoder()

encoder.fit(ytrain)

encoded_Y = encoder.transform(ytrain)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
# fit the model

model.fit(padded_docs, dummy_y, epochs=500, verbose=1)
from keras.models import load_model
model.save('sentiment.h5')
# integer encode the documents

Xtest = ['bad movie ever in the history', 'awesome movie, very good', 'bad', 'what a awesome movie', 'movie was good']



test_docs = t.texts_to_sequences(Xtest)

#print(test_docs)



test_padded_docs = pad_sequences(test_docs, maxlen=max_length, padding='post')

#print(test_padded_docs)



a = model.predict(test_padded_docs)

#print(a)



for x in a:

    m = max(x)

    print(m, [i for i, j in enumerate(x) if j == m][0])
testing_data = pd.read_csv('../input/test.tsv', sep='\t')

testingdf = testing_data.values



XphraseID = testingdf[:,0]

Xtest = testingdf[:,2]

print(Xtest)

#print(XphraseID)
import csv

test_docs = t.texts_to_sequences(Xtest)

#print(test_docs)



test_padded_docs = pad_sequences(test_docs, maxlen=max_length, padding='post')

#print(test_padded_docs)



a = model.predict(test_padded_docs)

#print(a)



submission = open('../input/Submission.csv','w')

columnTitleRow = "PhraseId, Sentiment\n"

submission.write(columnTitleRow)



for counter, x in enumerate(a):

    m = max(x)

    print(XphraseID[counter], m, [i for i, j in enumerate(x) if j == m][0])

    #submission.write(str(XphraseID[counter])+','+str([i for i, j in enumerate(x) if j == m][0])+'\n')

    PhraseId = str(XphraseID[counter])

    Sentiment = str([i for i, j in enumerate(x) if j == m][0])

    row = PhraseId + "," + Sentiment + "\n"

    submission.write(row)

    

submission.close()    

    
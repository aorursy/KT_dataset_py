#Okay, I was getting ahead of myself.  Let's stick to the assignment
#the task from day #5 in this tutorial:
#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
import pandas
# define documents

docs = ['You are an annoying, whining man-child',
        'I am sure the pulse setting on your shower head will be devastated!',
        'I think your credit card statement would beg to differ.',
        'Did Santa finally bring you that Y-chromosome you always wanted?',
        'Should I talk slower or get a nurse who speaks fluent Moron?',
        'Elmo is so happy to see you! Welcome to Sesame Street!',
        'Elmo loves to stay nice and clean!',
        'It’s time to make up a musical!',
        'Oh look, it’s Mr. Noodle’s brother, Mr. Noodle.',
        'Elmo loves you!'] #First five are Dr. Cox from Scrubs, last 5 are Elmo from Sesame Street
# define class labels
print('doc type: ', type(docs))
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 1000

tokenMaker = Tokenizer()
tokenMaker.fit_on_texts(docs)

encoded_docs = tokenMaker.texts_to_sequences(docs)

max_length = 12
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)

print("shapes: ", padded_docs.shape, labels.shape)

model = Sequential()
model.add(Embedding(vocab_size, 12, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=10, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)
print('Accuracy: %f' % (accuracy*100))
print('Loss/ ', loss)
#I learned a lot from https://rajmak.in/2017/12/07/text-classification-classifying-product-titles-using-convolutional-neural-network-and-word2vec-embedding/
from gensim.models import KeyedVectors
from numpy import array
import numpy
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

EMBEDDING_DIMENSIONS=300

docs = ['You are an annoying, whining man-child',
        'I am sure the pulse setting on your shower head will be devastated!',
        'I think your credit card statement would beg to differ.',
        'Did Santa finally bring you that Y chromosome you always wanted?',
        'Should I talk slower or get a nurse who speaks fluent Moron?',
        'Elmo is so happy to see you! Welcome to Sesame Street!',
        'Elmo loves to stay nice and clean!',
        'It’s time to make up a musical!',
        'Oh look, it’s Mr. Noodle’s brother, Mr. Noodle.',
        'Elmo loves you!']
# define class labels
print('doc type: ', type(docs))
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 1000

word_model = KeyedVectors.load("../input/gensim-embeddings-dataset/GoogleNews-vectors-negative300.gensim")

print('Model type: ', type(word_model))
print('Found %s word vectors of Keyed Vectors' % len(word_model.vocab))

#print('Monkey: ', model['Monkey'], ' shape: ', model['Monkey'].shape)

vectors = [[word for word in line.split()] for line in docs]
for badchar in ['?', '!', '.', ',', '’', '-']:
    vectors = [[d.replace(badchar, '') for d in doc] for doc in vectors] 
print(vectors)
for badword in ['']: #I suppose these were removed from the training set?  
    for a in range(len(vectors)):   #They were wonky making when I made my own encoding
        if badword in vectors[a]: 
            vectors[a].remove(badword)
        if badword in vectors[a]: #This is not the most elegant solution, could've googled it, but remove was only removing the first instance
            vectors[a].remove(badword)
            print(a, badword)
print(vectors)

max_length = max(len(words) for words in vectors) + 1

print("Max length: ", max_length)

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(vectors)

numeric_sequences = tokenizer.texts_to_sequences(vectors)

#print(numeric_sequences)

total_words = len(tokenizer.word_index)
print("Total unique words: ", total_words)

#encoded_vectors = [[word_model[word] for word in vector] for vector in vectors]

#max_length = 12
padded_docs = pad_sequences(numeric_sequences, maxlen=max_length, padding='post')
#print(padded_docs)

print('shapes: ', padded_docs.shape, labels.shape)

embedding_matrix = numpy.zeros((vocab_size, EMBEDDING_DIMENSIONS))

for word, i in tokenizer.word_index.items():
    if word in word_model.vocab:
        #print("word: ", word)
        embedding_matrix[i] = word_model.word_vec(word)
print('Null word embeddings: %d' % numpy.sum(numpy.sum(embedding_matrix, axis=1) == 0))


embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                            embedding_matrix.shape[1], # or EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)


#print(vocab_size)
model = Sequential()
#model.add(Embedding(vocab_size, 12, input_length=max_length))
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=13, verbose=1)
# evaluate the model

loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)
print('Accuracy: %f' % (accuracy*100))
print('Loss/ ', loss)

test_tokens = tokenizer.texts_to_sequences(['God I would love to kick you in your dumb dumb face', 'Jesus loves you', 'lets be friends', 'get bent'])
padded_test = pad_sequences(test_tokens, maxlen=max_length, padding='post')
test_labels = array([1,0,0, 1])
loss, accuracy = model.evaluate(padded_test, test_labels, verbose=1)
print('Test Accuracy: %f' % (accuracy*100))
print('Test Loss: ', loss)
print('Confidence: ', model.predict(padded_test))


#A slightly modified version of the task from day #5 in this tutorial:
#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
import pandas
# define documents

reviews = pandas.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
reviews.describe()


reviews['sentiment'].replace({'negative':0, 'positive':1}, inplace=True)
#docs = ['Well done!',
#        'Good work',
#        'Great effort',
#        'nice work',
#        'Excellent!',
#        'Weak Sauce',
#        'Poor effort!',
#        'not good',
#        'poor work',
#        'Could have done better.']
# define class labels
docs = reviews['review'].values.tolist()
#print(docs)
print('doc type: ', type(docs))
labels = reviews['sentiment']
#words = docs.split()
#labels = array([1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 250000


docs = [d.replace("<br />", "  ") for d in docs] #remove line breaks, probably meaningless
docs = [d.replace("\\", "  ") for d in docs] #removespecial char, probably meaningless

print(docs[54])

encoded_docs = [one_hot(d, vocab_size) for d in docs]
#print(encoded_docs)

longest = max(len(a) for a in encoded_docs)
print('Longest review: ', longest)

max_length = 2500
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)

X_Train, X_Validation, Y_Train, Y_Validation = train_test_split(padded_docs, labels, test_size=0.02, random_state=1)
model = Sequential()
model.add(Embedding(vocab_size, 12, input_length=max_length))
#4 = 87.7%
#16 = 89.4%
#64 = 88.8%
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(X_Train, Y_Train, epochs=10, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(X_Validation, Y_Validation, verbose=1)
print('Accuracy: %f' % (accuracy*100))
print('Loss/ ', loss)

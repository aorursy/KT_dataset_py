import numpy



numpy.random.seed(0)

from keras.models import Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.initializers import glorot_uniform

numpy.random.seed(1)
import os

data_dir = '../Desktop/datasets_and_embeddings/glove.6B.100d.txt'



embeddings_index = {}

word_to_index = {}

index_to_word = {}

# f = open(os.path.join(data_dir, 'glove.6B.100d.txt'))

f = open(data_dir, encoding="utf8")

i = 0

for line in f:

    values = line.split()

    word = values[0]

    word_to_index[word] = i

    index_to_word[i] = word

    embedding = numpy.asarray(values[1:], dtype='float32')

    embeddings_index[word] = embedding

    i+=1

f.close()



# for testing purpose to make sure we are getting correct word vectors

print('Found %s word vectors.' % len(embeddings_index))

print("sample embedding of word 'bangalore' is", embeddings_index['bangalore'])

word = "cucumber"

index = 50000

print("the index of", word, "in the vocabulary is", word_to_index[word])

print("the", str(index+1) + "th word in the vocabulary is", index_to_word[index])
import csv



filename = '../Desktop/datasets_and_embeddings/helpline_datasets.csv'

X_train_list = []

Y_train_list = []

with open(filename,'r',encoding="utf-8") as csvfile:

    csvreader = csv.reader(csvfile)

    for row in csvreader:

        X_train_list.append(row[0])

        Y_train_list.append(row[1])

            

X_train = numpy.array(X_train_list) # converting list into 1-D vector representation   

Y_train = numpy.array(Y_train_list) # converting list into 1-D vector representation

num_training_sets = 0

try:

    num_training_sets = numpy.prod(X_train.shape)

    print("total datasets in training: ",num_training_sets)

    if num_training_sets != numpy.prod(X_train.shape):

        raise ValueError

except ValueError:

    print("dimensions of datasets won't match, verify once!!!!!")

maxLen = len(max(X_train, key=len).split())

index = 9

print("sample of training set in 9th index: ",X_train[index], Y_train[index])
def sentences_to_indices(X, word_to_index, max_len):

    m = X.shape[0] #number of training examples

    X_indices = numpy.zeros((m, max_len))

    for i in range(m):

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.

        sentence_words = X[i].lower().split()

        j = 0

        for w in sentence_words:

            X_indices[i,j] = word_to_index[w]

            j = j+1

    return X_indices
X1 = numpy.array(["for the past several days I'm suffering from a lot of mental tension due to ongoing audits in office late night",

                 "I got bullied yesterday from a bunch of senior guys during interval"])

X1_indices = sentences_to_indices(X1,word_to_index, max_length = 21)

print("X1=",X1)

print("X1_indices=",X1_indices)
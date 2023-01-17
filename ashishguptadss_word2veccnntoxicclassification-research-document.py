from gensim.models import word2vec



import gensim

from gensim import summarization

import nltk

from nltk.corpus import stopwords

import sklearn

import numpy as np

from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

stopw = set(nltk.corpus.stopwords.words('english'))
def readfile(file):

    f = open(file, 'r', encoding = 'utf8')

    text = f.read()

    sentences = nltk.sent_tokenize(text)

    print(len(sentences))

    

    data = []

    for sent in sentences:

        words = nltk.word_tokenize(sent)

        words = [w.lower() for w in words if len(w)>2 and w not in stopw ]

        data.append(words)

    print(len(data[1]))

    return data



text = readfile("../input/toxic-comment-classification/text.txt")

print(text)
from gensim.models import Word2Vec



model = Word2Vec(text, size = 300, window = 10, min_count = 1)



print(model['coding'].shape)
words = list(model.wv.vocab)

print(words[:10])
def read_glove_vecs(glove_file):

    with open(glove_file, 'r') as f:

        words = set()

        word_to_vec_map = {}

        

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

            

    return words, word_to_vec_map

# path_to_data= './data/glove.6B.50d.txt'

# words, w2vModel = read_glove_vecs(path_to_data)
# print(len(words))
# w2vModel['hello'].shape
# from gensim.models import KeyedVectors

# filename = 'GoogleNews-vectors-negative300.bin'

# w2vModel = KeyedVectors.load_word2vec_format(filename, binary=True)



# (w2vModel.word_vec('bird'))[:10] # Google's word embedding embedding matrix

#  w2vModel.wv
# from gensim.models import KeyedVectors

# filename = 'somefilename'

# model = KeyedVectors.load_word2vec_format(filename, binary= True)
# !ls './Toxic-Comment-Classification/'
import pandas as pd

import numpy as np

import re

AVG_SENT_LEN = 70

SIZE_OF_EACH_WORD_EMBEDDING = 300

PATH_TO_WORD_EMBEDDINGS = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'

TRAIN_CSV = 'train.csv'

TEST_CSV = "test.csv"

TEST_LABELS_CSV = "test_labels.csv"



def load_data(trainPath, testPath, testLabelsPath):

    train = pd.read_csv("../input/toxic-comment-classification/train.csv").comment_text

    test = pd.read_csv("../input/toxic-comment-classification/test.csv").comment_text

    testLabels = pd.read_csv("../input/toxic-comment-classification/test_labels.csv")

    trainLabels = pd.read_csv("../input/toxic-comment-classification/train.csv").values[:,2:]

    

    return (train, test, trainLabels, testLabels)



train, test, trainLabels, testLabels = load_data(TRAIN_CSV, TEST_CSV, TEST_LABELS_CSV)
print(train.shape, trainLabels.shape)

train.head(3)
test.tail(5)
testLabels.head()



# Clearly, not a useful info, this is just for the output submission
data = train.values

print(data.shape)
minLen = 10000

minLenSent = ''

maxLenSent = ''

maxLen = -10000

len_data = len(data)

sum_of_lengths_of_sentences = 0



sentences_lengths = []

for sent in data:

    sent = re.findall(r"[\w']+", sent)

    sum_of_lengths_of_sentences +=len(sent)

    sentences_lengths.append(len(sent))

    if(len(sent))>maxLen:

        maxLen = len(sent)

        maxLenSent = sent

    if(len(sent))<minLen:

        minLen = len(sent)

        minLenSent = sent

        

avg_len = sum_of_lengths_of_sentences/len_data

print('average sentence length',avg_len)

print('Max length, minlength, min length sentence is : ',maxLen, minLen, minLenSent)

sentences_lengths = np.array(sentences_lengths)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(0)

sns.distplot(sentences_lengths)

plt.figure(1)

sns.boxplot(sentences_lengths)

plt.show()
# Dependencies

import numpy as np

import pandas as pd

from gensim.models import KeyedVectors

import keras
AVG_SENT_LEN = 70

SIZE_OF_EACH_WORD_EMBEDDING = 300

PATH_TO_WORD_EMBEDDINGS = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'

TRAIN_CSV = '../input/toxic-comment-classification/train.csv'

TEST_CSV = "../input/toxic-comment-classification/test.csv"

TEST_LABELS_CSV = "../input/toxic-comment-classification/test_labels.csv"
def load_data(trainPath, testPath, testLabelsPath):

    train = pd.read_csv("../input/toxic-comment-classification/train.csv").comment_text

    test = pd.read_csv("../input/toxic-comment-classification/test.csv").comment_text

    testLabels = pd.read_csv("../input/toxic-comment-classification/test_labels.csv").values[:,1:]

    trainLabels = pd.read_csv("../input/toxic-comment-classification/train.csv").values[:,2:]

    

    return (train, test, trainLabels, testLabels)



train, test, trainLabels, testLabels = load_data(TRAIN_CSV, TEST_CSV, TEST_LABELS_CSV)
print(train.shape, test.shape, trainLabels.shape, testLabels.shape)



for row in trainLabels[:10]:

    print(row)
# train labels have to be added with another category

def addSeventhRow(labels):

    output = []

    for row in labels:

        if row.any() == False:

            row = np.append(row, [1], axis = 0)

        else:

            row = np.append(row, [0], axis = 0)

        output.append(row)

    output = np.array(output)

    return output



trainLabels = addSeventhRow(trainLabels)

testLabels = addSeventhRow(testLabels)



print(trainLabels.shape, testLabels.shape) # should show (--, 7)
for row in trainLabels[:10]:

    print(row)
data = train.values
def data_to_seq(data):

    # from keras.utils import 

    seqs = []

    for sent in data:

        seq = keras.preprocessing.text.text_to_word_sequence(sent, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

        seqs.append(seq)

    return seqs



def sentence_to_seq(sent):

    return keras.preprocessing.text.text_to_word_sequence(sent, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

    

sequences = data_to_seq(data)

print('Data before sequencing: ',data[0][:100], end = ' ')

print('...')

print('Data after processing: ',sequences[0])

print(len(sequences[0]), len(sequences[1]))
def getWordEmbeddingModel(path_to_embedding_bin = ''):

    filename = path_to_embedding_bin

    w2vModel = KeyedVectors.load_word2vec_format(filename, binary=True)

    return w2vModel

# this may take some time

w2vModel = getWordEmbeddingModel(PATH_TO_WORD_EMBEDDINGS)
ytrain_embeddings = []

trainLabelIndex = 0
print(trainLabels[7])
SAMPLES = 2000
# def getSequenceEmbeddings(sequences, w2vModel):

#     embeddings = []

#     trainLabelIndex= 0

#     ytrain_embeddings = []

#     for seq in sequences:

#         sequences_embedding = []

#         for word in seq:

#             try:

#                 wordvec = w2vModel.word_vec(word)

#                 sequences_embedding.append(wordvec)

# #                 print(len(sequences_embedding))

#             except KeyError:

# #                 print(word + ' not in word2vec...skipping')

#                 continue

#         # Pad the seq_embeddings

        

#         if len(sequences_embedding) is 0:

#             continue

            

#         sequences_embedding = np.array(sequences_embedding)

#         # to print the sequence embedding length

# #         print(sequences_embedding.shape[0])

#         if sequences_embedding.shape[0]<=AVG_SENT_LEN:

#             temp = sequences_embedding

#             zero_pad = np.zeros((AVG_SENT_LEN - temp.shape[0], SIZE_OF_EACH_WORD_EMBEDDING))

#             sequences_embedding = np.append(temp,zero_pad, axis = 0)

#         else:

#             sequences_embedding = sequences_embedding[:70]

#         trainLabelIndex+=1

#         embeddings.append(sequences_embedding)

#         ytrain_embeddings.append(trainLabels[trainLabelIndex])

#     return np.array(embeddings), np.array(ytrain_embeddings)



# embeddings, labels= getSequenceEmbeddings(sequences[:SAMPLES], w2vModel )

# print(labels.shape)

# print(embeddings.shape)
from collections import Counter

from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE

from keras.models import Sequential

from keras.layers import Activation, Input, Dense, Dropout, MaxPool1D, MaxPool2D, Conv2D, Flatten, Embedding

import keras

from keras.optimizers import Adam


input_words = Input(shape = (AVG_SENT_LEN, SIZE_OF_EACH_WORD_EMBEDDING,1))



conv1 = Conv2D(1, kernel_size= (3,SIZE_OF_EACH_WORD_EMBEDDING), strides = 1, padding = 'same')(input_words)

conv2 = Conv2D(1, kernel_size= (4,SIZE_OF_EACH_WORD_EMBEDDING), strides = 1, padding = 'same')(input_words)

conv3 = Conv2D(1, kernel_size= (5,SIZE_OF_EACH_WORD_EMBEDDING), strides = 1, padding = 'same')(input_words)



# Add max pool here



concat = keras.layers.Concatenate()([conv1, conv2, conv3])

# pool = MaxPool1D(3)(concat)

flat = Flatten()(concat)



dense0 = Dense(16)(flat)

dense1 = Dense(7, activation = 'sigmoid')(dense0)

from keras.models import Model

model = Model(inputs = input_words, outputs = dense1)



adam = Adam(lr = 0.0001)

model.compile("adam", loss = 'binary_crossentropy')
# We have to embed the labels into the embeddings, then pass it through the random over sampler 

# Then extract those labels , train the network(CNN)

# Perform the random oversampling



def getSequenceEmbeddings(sequences, w2vModel):

    embeddings = []

    trainLabelIndex= 0

    ytrain_embeddings = []

    for seq in sequences:

        sequences_embedding = []

        for word in seq:

            try:

                wordvec = w2vModel.word_vec(word)

                sequences_embedding.append(wordvec)

#                 print(len(sequences_embedding))

            except KeyError:

#                 print(word + ' not in word2vec...skipping')

                continue

        # Pad the seq_embeddings

        

        if len(sequences_embedding) is 0:

            continue

            

        sequences_embedding = np.array(sequences_embedding)

        # to print the sequence embedding length

#         print(sequences_embedding.shape[0])

        if sequences_embedding.shape[0]<=AVG_SENT_LEN:

            temp = sequences_embedding

            zero_pad = np.zeros((AVG_SENT_LEN - temp.shape[0], SIZE_OF_EACH_WORD_EMBEDDING))

            sequences_embedding = np.append(temp,zero_pad, axis = 0)

        else:

            sequences_embedding = sequences_embedding[:70]

        trainLabelIndex+=1

        embeddings.append(sequences_embedding)

        ytrain_embeddings.append(trainLabels[trainLabelIndex])

    return np.array(embeddings), np.array(ytrain_embeddings)



def prepare_data_for_over_sampling(embeddings, labels):

    embeddings_with_labels = []

    for i in range(embeddings.shape[0]): 

        label = [labels[i] for x in range(AVG_SENT_LEN)] # make a 70*7 array

        labeled_embedding = np.append(embeddings[i], label, axis = 1) # Simple append 

        embeddings_with_labels.append(labeled_embedding) # New embedding matrix

    embeddings_with_labels = np.array(embeddings_with_labels) # Make to numpy array



    binary_embedding_labels = []

    for i in range(embeddings.shape[0]):

        binary_embedding_labels = []

        row_= 0

        for row in labels:

            if row[:6].any() == False:

                row = 1 # toxic

            else:

                row = 0 # non toxic

            binary_embedding_labels.append(row)

        binary_embedding_labels = np.array(binary_embedding_labels)



    return embeddings_with_labels, binary_embedding_labels

    

    

def random_over_sample_dataset(embeddings_with_labels,binary_embedding_labels):

    ros = RandomOverSampler(random_state=42)

    # 2d array is accepted hence we reshaped it

    X_res, y_res = ros.fit_resample(embeddings_with_labels.reshape(embeddings_with_labels.shape[0],-1), binary_embedding_labels)

    # just reshaped back

    X_res = X_res.reshape(X_res.shape[0], 70,-1)

    return (X_res, y_res)

    

def extract_labels(embeddings_with_labels):

    embeddings_without_labels = []

    output_labels = []

    for i in range(embeddings_with_labels.shape[0]): 

        label = embeddings_with_labels[i][0][SIZE_OF_EACH_WORD_EMBEDDING:]

        true_embeddings = embeddings_with_labels[i,:,:SIZE_OF_EACH_WORD_EMBEDDING]

        embeddings_without_labels.append(true_embeddings)

        output_labels.append(label)

    output_labels = np.array(output_labels)

    embeddings_without_labels = np.array(embeddings_without_labels)

    

    

    return embeddings_without_labels, output_labels
LEN_TRAIN_DATA = len(sequences)

SEQ_TO_PROCESS = 1000

def performIteration():

    start = 0

    while(start<= LEN_TRAIN_DATA):

        end = start+SEQ_TO_PROCESS

        

        embeddings, labels = getSequenceEmbeddings(sequences[start:end], w2vModel)

        embeddings_with_labels , binary_embedding_labels = prepare_data_for_over_sampling(embeddings, labels)

        X_res, y_res = random_over_sample_dataset(embeddings_with_labels, binary_embedding_labels)

        oversampled_embeddings, oversampled_labels = extract_labels(X_res)

        

        print(oversampled_embeddings.shape, oversampled_labels.shape)

        model.fit(oversampled_embeddings.reshape(-1,70,300,1), oversampled_labels, epochs = 10, verbose = 1, validation_split = 0.15)        

        

        start+=SEQ_TO_PROCESS

    

performIteration()
from sklearn.metrics import fbeta_score



start =1001

end = 1100

test_data = data[start:end]

test_seq = sequences[start:end]



count = 0

correct_prediction = 0

embeddings, labels = getSequenceEmbeddings(test_data,w2vModel)

output = model.predict(embeddings.reshape(-1,70,300,1)) # One of 6 results



predictions = []

for i in range(start,end):

    print(test_data[i-start][:100])

    print(np.argmax(output[i-start]), end  = " ")

    print(np.argmax(labels[i-start])) 
# embeddings.shape
# # We have to embed the labels into the embeddings, then pass it through the random over sampler 

# # Then extract those labels , train the network(CNN)



# def prepare_data_for_over_sampling(embeddings, labels):

#     embeddings_with_labels = []

#     for i in range(embeddings.shape[0]): 

#         label = [labels[i] for x in range(AVG_SENT_LEN)] # make a 70*7 array

#         labeled_embedding = np.append(embeddings[i], label, axis = 1) # Simple append 

#         embeddings_with_labels.append(labeled_embedding) # New embedding matrix

#     embeddings_with_labels = np.array(embeddings_with_labels) # Make to numpy array



# #     print(embeddings_with_labels.shape)

    



#     binary_embedding_labels = []

#     for i in range(embeddings.shape[0]):

#         binary_embedding_labels = []

#         row_= 0

#         for row in labels:

#             if row[:6].any() == False:

#                 row = 1 # toxic

#             else:

#                 row = 0 # non toxic

#             binary_embedding_labels.append(row)

#         binary_embedding_labels = np.array(binary_embedding_labels)

# #     print(binary_embedding_labels)



#     return embeddings_with_labels, binary_embedding_labels



# embeddings_with_labels , binary_embedding_labels = prepare_data_for_over_sampling(embeddings, labels)

# embeddings_with_labels.shape, binary_embedding_labels.shape, labels.shape
# # Perform the random oversampling

# from collections import Counter

# from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE



# def random_over_sample_dataset(embeddings_with_labels,binary_embedding_labels):

#     ros = RandomOverSampler(random_state=42)

#     # 2d array is accepted hence we reshaped it

#     X_res, y_res = ros.fit_resample(embeddings_with_labels.reshape(embeddings_with_labels.shape[0],-1), binary_embedding_labels)

#     # just reshaped back

#     X_res = X_res.reshape(X_res.shape[0], 70,-1)

#     return (X_res, y_res)

    

# X_res, y_res = random_over_sample_dataset(embeddings_with_labels, binary_embedding_labels)

# X_res.shape, y_res.shape
#  def extract_labels(embeddings_with_labels):

#     embeddings_without_labels = []

#     output_labels = []

#     for i in range(embeddings_with_labels.shape[0]): 

#         label = embeddings_with_labels[i][0][SIZE_OF_EACH_WORD_EMBEDDING:]

#         true_embeddings = embeddings_with_labels[i,:,:SIZE_OF_EACH_WORD_EMBEDDING]

#         embeddings_without_labels.append(true_embeddings)

#         output_labels.append(label)

#     output_labels = np.array(output_labels)

#     embeddings_without_labels = np.array(embeddings_without_labels)

    

    

#     return embeddings_without_labels, output_labels





# oversampled_embeddings, oversampled_labels = extract_labels(X_res)

# print(oversampled_embeddings.shape, oversampled_labels.shape)
# print(len(oversampled_embeddings[0][2]))
from keras.models import Sequential

from keras.layers import Activation, Input, Dense, Dropout, MaxPool1D, MaxPool2D, Conv2D, Flatten, Embedding

import keras

from keras.optimizers import Adam


input_words = Input(shape = (AVG_SENT_LEN, SIZE_OF_EACH_WORD_EMBEDDING,1))



conv1 = Conv2D(1, kernel_size= (3,SIZE_OF_EACH_WORD_EMBEDDING), strides = 1, padding = 'same')(input_words)

conv2 = Conv2D(1, kernel_size= (4,SIZE_OF_EACH_WORD_EMBEDDING), strides = 1, padding = 'same')(input_words)

conv3 = Conv2D(1, kernel_size= (5,SIZE_OF_EACH_WORD_EMBEDDING), strides = 1, padding = 'same')(input_words)



# Add max pool here



concat = keras.layers.Concatenate()([conv1, conv2, conv3])

# pool = MaxPool1D(3)(concat)

flat = Flatten()(concat)



dense0 = Dense(16)(flat)

dense1 = Dense(7, activation = 'sigmoid')(dense0)

from keras.models import Model

model = Model(inputs = input_words, outputs = dense1)



adam = Adam(lr = 0.0001)

model.compile("adam", loss = 'binary_crossentropy')
hist = model.fit(embeddings.reshape(-1,70,300,1), labels, epochs = 10)
import matplotlib.pyplot as plt

import seaborn as sns



sns.lineplot(data = np.array(hist.history['loss']))

plt.xlabel("Epochs")

plt.ylabel("Loss over several epochs")

plt.title("Loss plot")

plt.show()
# # Take data not trained upon



# # Might not work just yet, but hopefully it will

# start =1001

# end = 1003

# test_data = data[start:end]

# test_seq = sequences[start:end]



# embeddings, labels = getSequenceEmbeddings(test_data,w2vModel)

# output = model.predict(embeddings.reshape(-1,70,300,1)) # One of 6 results

# for i in range(start, end):

#     first_result = output[i-start]

# #     print(test_data[i-start]) # the comment

#     print(first_result) # the review

#     print(trainLabels[i-start]) # The output, *7 -> [0,0,0,0,0,0]



# # Take data not trained upon



# Might not work just yet, but hopefully it will



from sklearn.metrics import fbeta_score



start =1001

end = 1100

test_data = data[start:end]

test_seq = sequences[start:end]



count = 0

correct_prediction = 0

embeddings, labels = getSequenceEmbeddings(test_data,w2vModel)

output = model.predict(embeddings.reshape(-1,70,300,1)) # One of 6 results



predictions = []

for i in range(start,end):

    print(test_data[i-start][:100])

    print(np.argmax(output[i-start]), end  = " ")

    print(np.argmax(labels[i-start]))   

    

# embeddings, labels = getSequenceEmbeddings(test_data,w2vModel)

# output = model.predict(embeddings.reshape(-1,70,300,1)) # One of 6 results

# for i in range(start, end):

#     first_result = output[i-start]

#     print(np.argmax(np.array(first_result)), end = " ") # the comment

# #     print(first_result[:15]) # the review

#     print('Predicted class: ', np.argmax(np.array(trainLabels[i-start]))) # The output, *7 -> [0,0,0,0,0,0]
import imblearn
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,

                           n_redundant=0, n_repeated=0, n_classes=3,

                           n_clusters_per_class=1,

                           weights=[0.01, 0.05, 0.94],

                           class_sep=0.8, random_state=0)
X.shape, y.shape
import pandas as pd

pd.DataFrame(y).info()
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)



print(X_resampled.shape, y_resampled.shape)
from collections import Counter

print(sorted(Counter(y_resampled).items()))
from collections import Counter
#IMPORTING ALL THE REQUIRED PACKAGES.
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from subprocess import check_output
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import codecs
import keras
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.callbacks import EarlyStopping
print(check_output(["ls", "../input"]).decode("utf8"))
stopwords0 = set(stopwords.words('english'))
#SETTING GLOBAL VARIABLES
EMBEDDINGDIM = 300
MAXVOCABSIZE = 175303 
MAXSEQLENGTH = 200 
batchsize = 256 
epochs = 3
#READING AND SETTING UP THE TRAIN.CSV FILE
traincomments = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv", sep=',', header=0)
traincomments.columns=['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print("num train: ", traincomments.shape[0])
traincomments.head()
labelnames = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
ytrain = traincomments[labelnames].values
#READING AND SETTING UP THE TEST.CSV FILE
testcomments = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv", sep=',', header=0)
testcomments.columns=['id', 'comment_text']
print("num test: ", testcomments.shape[0])
testcomments.head()
#CLEANING UP THE TEXT
#Function to clean up the text
def standardizetext(df, textfield):
    df[textfield] = df[textfield].str.replace(r"http\S+", "")
    df[textfield] = df[textfield].str.replace(r"http", "")
    df[textfield] = df[textfield].str.replace(r"@\S+", "")
    df[textfield] = df[textfield].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[textfield] = df[textfield].str.replace(r"@", "at")
    df[textfield] = df[textfield].str.lower()
    return df
#Cleaning the train data and making the new CSV file -> train_clean_data.csv
traincomments.fillna('_NA_')
traincomments = standardizetext(traincomments, "comment_text")
traincomments.to_csv("traincleandata.csv")
traincomments.head()
#Cleaning the test data and making the new CSV file -> test_clean_data.csv
testcomments.fillna('_NA_')
testcomments = standardizetext(testcomments, "comment_text")
testcomments.to_csv("testcleandata.csv")
testcomments.head()
#TOKENIZING THE TEXT
tokenizer = RegexpTokenizer(r'\w+')
cleantraincomments = pd.read_csv("traincleandata.csv")
cleantraincomments['comment_text'] = cleantraincomments['comment_text'].astype('str') 
cleantraincomments.dtypes
cleantraincomments["tokens"] = cleantraincomments["comment_text"].apply(tokenizer.tokenize)
# delete Stop Words
cleantraincomments["tokens"] = cleantraincomments["tokens"].apply(lambda vec: [word for word in vec if word not in stopwords0])
cleantraincomments.head()
cleantestcomments = pd.read_csv("testcleandata.csv")
cleantestcomments['comment_text'] = cleantestcomments['comment_text'].astype('str') 
cleantestcomments.dtypes
cleantestcomments["tokens"] = cleantestcomments["comment_text"].apply(tokenizer.tokenize)
cleantestcomments["tokens"] = cleantestcomments["tokens"].apply(lambda vec: [word for word in vec if word not in stopwords0])
cleantestcomments.head()
alltrainingwords = [word for tokens in cleantraincomments["tokens"] for word in tokens]
trainingsentencelengths = [len(tokens) for tokens in cleantraincomments["tokens"]]
TRAININGVOCAB = sorted(list(set(alltrainingwords)))
print("%s words total, with a vocabulary size of %s" % (len(alltrainingwords), len(TRAININGVOCAB)))
print("Max sentence length is %s" % max(trainingsentencelengths))
alltestwords = [word for tokens in cleantestcomments["tokens"] for word in tokens]
testsentencelengths = [len(tokens) for tokens in cleantestcomments["tokens"]]
TESTVOCAB = sorted(list(set(alltestwords)))
print("%s words total, with a vocabulary size of %s" % (len(alltestwords), len(TESTVOCAB)))
print("Max sentence length is %s" % max(testsentencelengths))
#WORD2VEC
word2vecpath = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vecpath, binary=True)
def getaverageword2vec(tokenslist, vector, generatemissing=False, k=300):
    if len(tokenslist)<1:
        return np.zeros(k)
    if generatemissing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokenslist]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokenslist]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged
#GETTING EMBEDDINGS
def getword2vecembeddings(vectors, cleancomments, generatemissing=False):
    embeddings = cleancomments['tokens'].apply(lambda x: getaverageword2vec(x, vectors, 
                                                                          generatemissing=generatemissing))
    return list(embeddings)
#TRAIN EMBEDDING
trainingembeddings = getword2vecembeddings(word2vec, cleantraincomments, generatemissing=True)
tokenizer = Tokenizer(num_words=MAXVOCABSIZE, lower=True, char_level=False)
tokenizer.fit_on_texts(cleantraincomments["comment_text"].tolist())
trainingsequences = tokenizer.texts_to_sequences(cleantraincomments["comment_text"].tolist())

trainwordindex = tokenizer.word_index
print('Found %s unique tokens.' % len(trainwordindex))

traincnndata = pad_sequences(trainingsequences, maxlen=MAXSEQLENGTH)

trainembeddingweights = np.zeros((len(trainwordindex)+1, EMBEDDINGDIM))
for word,index in trainwordindex.items():
    trainembeddingweights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDINGDIM)
print(trainembeddingweights.shape)
testsequences = tokenizer.texts_to_sequences(cleantestcomments["comment_text"].tolist())
testcnndata = pad_sequences(testsequences, maxlen=MAXSEQLENGTH)
#DEFINING THE CNN
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv==True:
        x = Dropout(0.5)(l_merge)  
    else:
        # Original Yoon Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Finally, we feed the output into a Sigmoid layer.
    # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0) 
    # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model
x_train = traincnndata
y_tr = ytrain
model = ConvNet(trainembeddingweights, MAXSEQLENGTH, len(trainwordindex)+1, EMBEDDINGDIM, 
                len(list(labelnames)), False)
#DEFINING CALLBACKS
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbackslist = [earlystopping]
#TRAINING THE NETWORK
hist = model.fit(x_train, y_tr, epochs=epochs, callbacks=callbackslist, validation_split=0.1, shuffle=True, batch_size=batchsize)
ytest = model.predict(testcnndata, batch_size=1024, verbose=1)
#CREATING THE SUBMISSION.CSV FILE
submissiondf = pd.DataFrame(columns=['id'] + labelnames)
submissiondf['id'] = testcomments['id'].values 
submissiondf[labelnames] = ytest 
submissiondf.to_csv("./cnn_submission.csv", index=False)
#GENERATING THE GRAPHS
plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.title('CNN sentiment')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()
plt.figure()
plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
plt.title('CNN sentiment')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
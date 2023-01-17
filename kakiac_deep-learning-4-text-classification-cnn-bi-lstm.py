# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Add
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
import gensim
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import codecs
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

stop_words = set(stopwords.words('english'))
# Any results you write to the current directory are saved as output.
EMBEDDING_DIM = 300 # how big is each word vector
MAX_VOCAB_SIZE = 175303 # how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH = 200 # max number of words in a comment to use

#training params
batch_size = 256 
num_epochs = 2 
train_comments = pd.read_csv("../input/manifestos-en/manifesots_en.csv", sep=',', header=0)
train_comments.columns=['text', 'cmp_code', 'eu_code', 'pos', 'manifesto_id', 'party', 'date', 'language', 'source', 'has_eu_code', 'is_primary_doc', 'may_contradict_core_dataset', 'md5sum_text', 'url_original', 'md5sum_original', 'annotations', 'handbook', 'is_copy_of', 'title', 'id']
#'NA', '0', '101', '102', '103', '104', '105', '106', '107', '108', '109', '201', '202', '203', '204', '301', '302', '303', '304', '305','401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '501', '502', '503', '504', '505', '506', '507', '601', '602', '603', '604', '605', '606', '607', '608', '701', '702', '703', '704', '705', '706', '707', '708')
print("num train: ", train_comments.shape[0])
train_comments.head()
# check that the values of cmp_code are of the right type
print(train_comments.cmp_code[1])
print(type(train_comments["cmp_code"]))

#turns values of cmp_code from object to a list, comma separated
builder_list = [] #creates an empty list
# loop - for every entry in the cmp_code column, add value to end of list, separated by commas
for data in train_comments["cmp_code"]: 
    builder_list.append(str(data))
",".join(builder_list)

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define dataset 
data = builder_list
values = array(data)
print(values[1])
# encode codes to integer  
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# print the second entry as an integer code
print(integer_encoded[1])
# encode integer codes to binary
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print the second entry as a code expressed in binary
print(onehot_encoded[1])

# invert the vector to output the code. This throws an error of Deprecation (DeprecationWarning: The truth value 
# of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` 
# to check that an array is not empty.)- will need to go back to it to check.

# print(type(onehot_encoded))
# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# print(inverted[1])
#label_names = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '201', '202', '203', '204', '301', '302', '303', '304', '305','401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '501', '502', '503', '504', '505', '506', '507', '601', '602', '603', '604', '605', '606', '607', '608', '701', '702', '703', '704', '705', '706', '707', '708']
#y_train = train_comments[label_names].values

Y_train = onehot_encoded
print(Y_train[1])

# trying to add the vectors in the main dataset
#clean_train_comments["codes_encoded"] = clean_train_comments["tokens"].apply(lambda vec: [word for word in vec if word not in stop_words]) 
#clean_train_comments.head()
#test_comments = pd.read_csv("../input/manifestos-aus/test-aus.csv", engine='python', sep=',', header=0)
#print("ok")
#test_comments.columns=['text', 'cmp_code', 'eu_code']
#print("num test: ", test_comments.shape[0])
#test_comments.head()

#This is no longer needed, as I will create the test on the fly from the dataset above (manifesots_en.csv)
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df
train_comments.fillna('_NA_')
train_comments.fillna('NaN')
train_comments = standardize_text(train_comments, "text")
train_comments.to_csv("train_clean_data.csv")
train_comments.head()
#test_comments.fillna('_NA_')
#test_comments = standardize_text(test_comments, "text")
#test_comments.to_csv("test_clean_data.csv")
#test_comments.head()

#This is no longer needed, as I will create the test on the fly from the dataset above (manifesots_en.csv)
tokenizer = RegexpTokenizer(r'\w+')
clean_train_comments = pd.read_csv("train_clean_data.csv")
clean_train_comments['text'] = clean_train_comments['text'].astype('str') 
clean_train_comments.dtypes
clean_train_comments["tokens"] = clean_train_comments["text"].apply(tokenizer.tokenize)
# delete Stop Words
clean_train_comments["tokens"] = clean_train_comments["tokens"].apply(lambda vec: [word for word in vec if word not in stop_words])
   
clean_train_comments.head()
#This is no longer needed, as I will create the test on the fly from the dataset above (manifesots_en.csv)
#clean_test_comments = pd.read_csv("test_clean_data.csv")
#clean_test_comments['text'] = clean_test_comments['text'].astype('str') 
#clean_test_comments.dtypes
#clean_test_comments["tokens"] = clean_test_comments["text"].apply(tokenizer.tokenize)
#clean_test_comments["tokens"] = clean_test_comments["tokens"].apply(lambda vec: [word for word in vec if word not in stop_words])

#clean_test_comments.head()
all_training_words = [word for tokens in clean_train_comments["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in clean_train_comments["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
#print(clean_train_comments["tokens"])
#all_test_words = [word for tokens in clean_test_comments["tokens"] for word in tokens]
#test_sentence_lengths = [len(tokens) for tokens in clean_test_comments["tokens"]]
#TEST_VOCAB = sorted(list(set(all_test_words)))
#print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
#print("Max sentence length is %s" % max(test_sentence_lengths))

word2vec_path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)
training_embeddings = get_word2vec_embeddings(word2vec, clean_train_comments, generate_missing=True)
# test_embeddings = get_word2vec_embeddings(word2vec, clean_test_comments, generate_missing=True)
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, char_level=False)
tokenizer.fit_on_texts(clean_train_comments["text"].tolist())
training_sequences = tokenizer.texts_to_sequences(clean_train_comments["text"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))

train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
#print(train_cnn_data[:4])

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))

for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights[1])
print("-----------=====-----------")
print(train_embedding_weights.shape)
#test_sequences = tokenizer.texts_to_sequences(clean_test_comments["text"].tolist())
#print(clean_test_comments["text"][4])
#print(test_sequences[4])
#test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

#I opted for splitting the train set in two parts : a small fraction (20%) became the validation set which the model is evaluated and the rest (80%) is used to train the model.

#Since our dataset is not balanced (explain) , a random split of the train set causes some labels to be over represented in the validation set and we end up with an unbalanced dataset. A simple random split could cause inaccurate evaluation during the validation, hence to avoid that, we use stratify = True option in train_test_split function (**Only for >=0.17 sklearn versions**).
# Split dataset into training and test/validation

from sklearn.model_selection import train_test_split
# Set the random seed
random_seed = 2

Y_train = onehot_encoded #(defined above)
#X_train = clean_train_comments["tokens"]
X_train = train_embedding_weights
print(X_train[0])
print(Y_train[0])
print(clean_train_comments["cmp_code"][0])
print(clean_train_comments["text"][0])
print(clean_train_comments["tokens"])

print("num X_train: ", X_train.shape[0])
print("num Y_train: ", Y_train.shape[0])
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)

#print(X_val)
#print(Y_val)
print("num start dataset: ", clean_train_comments.shape[0])
print("num X_val: ", X_val.shape[0])
print("num Y_val: ", Y_val.shape[0])
print("num X_train: ", X_train.shape[0])
print("num Y_train: ", Y_train.shape[0])

print(X_train)








from keras.layers.merge import concatenate, add

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    #the filter
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    #the unknown image
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    #the merge function of the first convolution 
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3,4,5] # in the loop, first apply 3 as size, then 4 then 5

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        #kernel is the filter
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    
    # activated if extra_convoluted is true at the def
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
x_train = train_cnn_data
y_tr = y_train
print(len(list(label_names)))
model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, len(list(label_names)), False)
#define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
# I opted for splitting the train set in two parts : a small fraction (20%) became the validation set which the model is 
# evaluated and the rest (80%) is used to train the model.
# Since our dataset is not balanced (explain) , a random split of the train set causes some labels to be over represented 
# in the validation set and we end up with an unbalanced dataset. A simple random split could cause inaccurate evaluation 
# during the validation, hence to avoid that, we use stratify = True option in train_test_split function 
# (**Only for >=0.17 sklearn versions**).



# Split dataset into training and test/validation

from sklearn.model_selection import train_test_split
# Set the random seed
random_seed = 2

Y_train = onehot_encoded #(defined above)
X_train = clean_train_comments["tokens"]
print(X_train[0])
print(Y_train[0])
print(clean_train_comments["cmp_code"][0])
print(clean_train_comments["text"][0])

print("num X_train: ", X_train.shape[0])
print("num Y_train: ", Y_train.shape[0])
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed) # can also use random_state=random_seed as an option if dataset is balanced

#print(X_val)
#print(Y_val)
print("num start dataset: ", clean_train_comments.shape[0])
print("num X_val: ", X_val.shape[0])
print("num Y_val: ", Y_val.shape[0])
print("num X_train: ", X_train.shape[0])
print("num Y_train: ", Y_train.shape[0])
hist = model.fit(x_train, y_tr, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, batch_size=batch_size)
y_test = model.predict(test_cnn_data, batch_size=1024, verbose=1)
print(y_test)
#create a submission
submission_df = pd.DataFrame(columns=['id'] + label_names)
submission_df['id'] = test_comments['id'].values 
submission_df[label_names] = y_test 
submission_df.to_csv("./cnn_submission.csv", index=False)
#generate plots
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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
EMBEDDING_FILE = '../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin.gz'
train = pd.read_csv('../input/manifestos-aus/train-aus.csv')
test = pd.read_csv('../input/manifestos-aus/test-aus.csv')
train["text"].fillna("fillna")
test["text"].fillna("fillna")
X_train = train["text"].str.lower()
y_train = train[['101', '102', '103', '104', '105', '106', '107', '108', '109', '201', '202', '203', '204', '301', '302', '303', '304', '305','401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '501', '502', '503', '504', '505', '506', '507', '601', '602', '603', '604', '605', '606', '607', '608', '701', '702', '703', '704', '705', '706', '707', '708']].values

X_test = test["text"].str.lower()
max_features=100000
maxlen=150
embed_size=300
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
tok=text.Tokenizer(num_words=max_features,lower=True)
tok.fit_on_texts(list(X_train)+list(X_test))
X_train=tok.texts_to_sequences(X_train)
X_test=tok.texts_to_sequences(X_test)
x_train=sequence.pad_sequences(X_train,maxlen=maxlen)
x_test=sequence.pad_sequences(X_test,maxlen=maxlen)
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
word_index = tok.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
sequence_input = Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.1)(x)
preds = Dense(6, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])
model.summary()
batch_size = 128
epochs = 4
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=233)
# filepath="../input/best-model/best.hdf5"
#filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
callbacks_list = [ra_val,checkpoint, early]
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)
#Loading model weights
#model.load_weights(filepath) #try this with and without; with is bi-LSTM with convolution
print('Predicting....')
y_pred = model.predict(x_test,batch_size=1024,verbose=1)
# Write scores to file
submission = pd.read_csv('../input/manifestos-aus/sample_submission.csv')
submission[['101', '102', '103', '104', '105', '106', '107', '108', '109', '201', '202', '203', '204', '301', '302', '303', '304', '305','401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '501', '502', '503', '504', '505', '506', '507', '601', '602', '603', '604', '605', '606', '607', '608', '701', '702', '703', '704', '705', '706', '707', '708']] = y_pred
submission.to_csv('submission.csv', index=False)
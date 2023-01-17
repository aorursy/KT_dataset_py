import numpy as np

import pandas as pd

from IPython.display import display
# load training variants

train = pd.read_csv('../input/training_variants')

# load training text

train_txt_ = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

# merge text & variants

train = pd.merge(train, train_txt_, how='left', on='ID').fillna('')

# clean up

del train_txt_

# print train data info

display(train.info())



# load test variants from stage 1

testold_var_ = pd.read_csv('../input/test_variants')

# load test text from stage 1

testold_txt_ = pd.read_csv('../input/test_text', sep='\|\|', engine='python', header=None, skiprows=1, names=["ID","Text"])

# merge text & variants

testold_ = pd.merge(testold_var_, testold_txt_, how='left', on='ID').fillna('')

# clean up

del testold_var_

del testold_txt_



# load stage1 solutions

stage1sol_ = pd.read_csv('../input/stage1_solution_filtered.csv')

# get class

stage1sol_['Class'] = pd.to_numeric(stage1sol_.drop('ID', axis=1).idxmax(axis=1).str[5:]).fillna(0).astype(np.int64)

# drop records from testold_ if they are not in stage1sol_

testold_ = testold_[testold_.index.isin(stage1sol_['ID'])]

# merge class to testold_ from stage1sol_

newtraindata_ = testold_.merge(stage1sol_[['ID', 'Class']], on='ID', how='left')

# reindex columns

newtraindata_ = newtraindata_.reindex_axis(['ID','Gene','Variation','Class','Text'], axis=1)

# clean up

del stage1sol_

del testold_



# append new train data

train = train.append(newtraindata_)

# clean up

del newtraindata_



# print train data info

display(train.info())
print('Indexing word vectors.')

import os

from gensim.models import KeyedVectors

word2vec = None

# make sure you load this on your local env and uncomment the line

#word2vec = KeyedVectors.load_word2vec_format('PubMed-and-PMC-w2v.bin', binary=True)

if (word2vec == None):

    print("word2vec not loaded!")

else:

    print("Found {} word vectors of word2vec".format(len(word2vec.vocab)))
# due kaggle limit we will truncate train database, remove this block if running on local env

# debug msg

print('Split dataset.')

# set to max value of orig dataset

maxsize = len(train)

# check class distrubution and find min sample size

for c in range(1,10):

    _ = len(train[train['Class']==c])

    if (_ < maxsize):

        maxsize = _

# debug msg

print('max size', maxsize)

# create new dataframe

train_ = pd.DataFrame(columns=train.columns)

for c in range(1,10):

    # append samples from train of length maxsize

    train_ = train_.append(train[train['Class']==c][:maxsize], ignore_index=True)

# display truncated data

display(train_.head())

# debug msg

print('Train dataset old size {} new size {}'.format(len(train),len(train_)))

# overwrite train with truncated train data

train = train_

# debug msg

print('Split dataset done')
import nltk



# Create a function called "chunks" with two arguments, l and n:

def chunks(l, n):

    # For item i in a range that is a length of l,

    for i in range(0, len(l), n):

        # Create an index range for l of n items:

        yield l[i:i+n]



print('Expand records to sentences.')

# increase maxnumberofsentecs on local env to 400

maxnumberofsentences = 200

# increase splitbysenteces on local env to 10

splitbysentences = 2

# temp dict for new train set

tmpdf_ = {'Text': [], 'Class': [], 'ID': [], 'Gene': [], 'Variation': []}

for index, row in train.iterrows():

    # get sentences nltk

    sent_tokenize_list = nltk.sent_tokenize(row['Text'])

    # truncate sentences to last maxnumberofsentences (most important informations are at the end of text)

    if (len(sent_tokenize_list) > maxnumberofsentences):

        sent_tokenize_list = sent_tokenize_list[len(sent_tokenize_list)-maxnumberofsentences:]

    # split sentences to batch

    sent_chunk = list(chunks(sent_tokenize_list, splitbysentences))

    for chunk in sent_chunk:

        # join sentences in text

        tmpdf_['Text'].append(" ".join(chunk))

        # assign class

        tmpdf_['Class'].append(row['Class'])

        # assign ID

        tmpdf_['ID'].append(row['ID'])

        # assign Gene

        tmpdf_['Gene'].append(row['Gene'])

        # assign Variation

        tmpdf_['Variation'].append(row['Variation'])

# create new train set from temp dict

origtrainlen = len(train)

train = pd.DataFrame(tmpdf_)

# clean up

del tmpdf_

# display head

display(train.head())

# display 

print('expanded from {} to {}'.format(origtrainlen,len(train)))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# max top words, increase on local env to 100000

num_words = 5000

# max sequence length, increase on local env to 500

sequencelength = 200

# init tokenizer

tokenizer = Tokenizer(num_words=num_words)

# fit tokenizer

tokenizer.fit_on_texts(train['Text'])

# get sequences

X = tokenizer.texts_to_sequences(train['Text'])

# unique words in text

word_index = tokenizer.word_index

print("Found {} unique tokens.".format(len(word_index)))

# pad sequences

X = pad_sequences(X, maxlen=sequencelength)



embedding_matrix = None

if (word2vec != None):

    # out of vocabulary words > use this to do text analysis

    oov_words = []

    # prepare embedding matrix

    embedding_matrix = np.zeros((num_words+1, 200)) #200 = word2vec dim

    for word, i in word_index.items():

        if i >= num_words:

            continue

        if word in word2vec.vocab:

            # embedd from word2vec

            embedding_matrix[i] = word2vec.word_vec(word)

        else:

            # add to out of vocabulary

            oov_words.append(word)

    print('Preparing embedding matrix done. out-of-vocabulary rate (OOV): {} ({})'.format(len(oov_words)/float(len(word_index)),len(oov_words)))

    
import keras

from sklearn.utils import class_weight



embed_dim = 200 #same as word2vec dim



model_filename = 'model'



# prepare Y values

Y = train['Class'].values-1

# get weights for unevenly distributed dataset 

class_weight = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

# one hot

Y = keras.utils.to_categorical(Y)

# batch size increase on local env

batch_size = 20

# epochs increase on local env

epochs = 3

# Model saving callback

ckpt_callback = keras.callbacks.ModelCheckpoint(model_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')



# input layer

input1 = keras.layers.Input(shape=(sequencelength,))

# embedding layer

if (embedding_matrix == None):

    # word2vec was not loaded. use fallback method

    embedding = keras.layers.Embedding(num_words+1, embed_dim, trainable=True)(input1)

else:

    # word2vec was loaded, load weights and set to untrainable

    embedding = keras.layers.Embedding(num_words+1, embed_dim, weights=[embedding_matrix], trainable=False)(input1)

 

# conv layers

convs = []

filter_sizes = [2,3,4]

for fsz in filter_sizes:

    l_conv = keras.layers.Conv1D(filters=100,kernel_size=fsz,activation='relu')(embedding)

    l_pool = keras.layers.MaxPooling1D(sequencelength-100+1)(l_conv)

    l_pool = keras.layers.Flatten()(l_pool)

    convs.append(l_pool)

# merge conv layers

l_merge = keras.layers.concatenate(convs, axis=1)

# drop out regulation

l_out = keras.layers.Dropout(0.5)(l_merge)

# output layer

output = keras.layers.Dense(units=9, activation='softmax')(l_out)

# model

model = keras.models.Model(input1, output)

# compile model

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])

# train model

model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1, class_weight=class_weight, callbacks=[ckpt_callback])
# load test dataset

test = pd.read_csv('../input/stage2_test_variants.csv')

# load test text dataset

test_txt_ = pd.read_csv('../input/stage2_test_text.csv', sep='\|\|', engine='python', header=None, skiprows=1, names=["ID","Text"])

# merge text & variants

test = pd.merge(test, test_txt_, how='left', on='ID')

# clean up

del test_txt_
print('Expand records to sentences.')

# temp dict for new train set

tmpdf_ = {'Text': [], 'ID': [], 'Gene': [], 'Variation': []}

for index, row in test.iterrows():

    # get sentences nltk

    sent_tokenize_list = nltk.sent_tokenize(row['Text'])

    # truncate sentences to last maxnumberofsentences (most important informations are at the end of text)

    if (len(sent_tokenize_list) > maxnumberofsentences):

        sent_tokenize_list = sent_tokenize_list[len(sent_tokenize_list)-maxnumberofsentences:]

    # split sentences to batch

    sent_chunk = list(chunks(sent_tokenize_list, splitbysentences))

    for chunk in sent_chunk:

        # join sentences in text

        tmpdf_['Text'].append(" ".join(chunk))

        # assign ID

        tmpdf_['ID'].append(row['ID'])

        # assign Gene

        tmpdf_['Gene'].append(row['Gene'])

        # assign Variation

        tmpdf_['Variation'].append(row['Variation'])

# create new train set from temp dict

origtestlen = len(test)

test = pd.DataFrame(tmpdf_)

# clean up

del tmpdf_

# display head

display(test.head())

# display 

print('expanded from {} to {}'.format(origtestlen,len(test)))
# load best model

model = keras.models.load_model(model_filename)

# get sequences

Xtest = tokenizer.texts_to_sequences(test['Text'])

# pad sequences

Xtest = pad_sequences(Xtest, maxlen=sequencelength)

# predict

probas = model.predict(Xtest, verbose=1)

# prepare data for submission

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])

# insert IDs

submission_df.insert(loc=0, column='ID', value=test['ID'].values)

# average grouped data

submission_df = submission_df.groupby(['ID'], as_index=False).mean()

# save to csv

submission_df.to_csv('submission.csv', index=False)

# debug

print("\n----------------------\n")

print("Done")
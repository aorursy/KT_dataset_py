from lxml import etree

from nltk.corpus import wordnet as wn

from nltk.tokenize import RegexpTokenizer

from typing import Dict, List, Tuple

import os

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from gensim.models import KeyedVectors

from nltk.corpus import stopwords

from string import punctuation as punct

import numpy as np

from nltk.corpus import wordnet as wn
import re



def anglicise(matchobj): 

    if matchobj.group(0) == '&amp;':

        return matchobj.group(0)

    else:

        return matchobj.group(0)[1]



# data = ""

with open('../input/omstidata/semcoromsti.data.xml') as inXML, open('semcoromsti.data.xml', 'w') as outXML:

#     data = inXML.readline()

    outXML.write(inXML.readline())

#     data = data + '<root>'

    outXML.write('<root>\n')

    for line in inXML.readlines():

        outXML.write(re.sub('&[a-zA-Z]+;',anglicise,line))

#         data = data + re.sub('&[a-zA-Z]+;',anglicise,line)

    outXML.write('</root>\n')

#     data = data + '</root>'
from io import BytesIO

some_file_like = BytesIO(data.encode('utf-8'))
def load_bn2wn_mapping(bn2wn_mapping_path: str, flip_dxn = False) -> Dict[str, str]:

    """

    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora

    :return bn2wn_mapping; the BabelNet to WordNet mapping file encoded as a dictionary with the BabelNet IDs as keys

    """

    first = 0

    second = 1

    if (flip_dxn):

        first = 1

        second = 0

    bn2wn_mapping = dict()

    with open(bn2wn_mapping_path, 'r') as handle:

        for line in handle:

            line = line.strip().split("\t")

            if (line):

                bn2wn_mapping[line[first]] = line[second]



    return bn2wn_mapping





def load_gold_data(gold_data_path: str) -> Dict[str, str]:

    gold_mapping = dict()

    with open(gold_data_path, 'r') as handle:

        for line in handle:

            line = line.strip().split(" ")

            if (line):

                gold_mapping[line[0]] = line[1]



    return gold_mapping
import re

from IPython.core.debugger import set_trace

sentences = []

sequences = []

# xpath = "trainX.txt"

# ypath = "trainY.txt"

sentence_count = 0



def check_punc(word):

    """ Checks if word does not contain only punctuations

    """

    if not re.match(r'^[_\W]+$', word):

        return True

    else:

        return False



def preprocess_semcor(file_path:str,gold_mapping_path: str, bn2wn_mapping_path: str) -> None:

    """ Parse xml file  provided in file_path and returns a sentence,

    babelnetId,lemma and anchors respectively for each sentences

    """

    

    print("Loading gold data...")

    gold_mapping = load_gold_data(gold_mapping_path)

    print("Loading WordNet mapping data...")

    bn2wn_mapping = load_bn2wn_mapping(bn2wn_mapping_path, True)

    

    print("Parsing Data...")

    context = etree.iterparse(file_path, events=('end',), tag='sentence')

    for event, elem in context:

        sentence_x = ""

        sentence_y = ""

        

        try:

            for e in elem.iter():

               

                if e.tag == "wf" and e.attrib.get('lemma') != False and e.text != None and (check_punc(e.text) == True) and (check_punc(e.attrib['lemma'].strip()) == True):

                    if (len(e.text.split()) > 1):

                        sentence_x = sentence_x + " " + e.text.replace(" ","_")

                    else:

                        sentence_x = sentence_x + " " + e.text

                    sentence_y = sentence_y + " " + e.attrib['lemma'].strip()



                elif e.tag == "instance" and e.attrib.get('id') != False and e.text != None and (check_punc(e.text) == True) and (check_punc(e.attrib['lemma'].strip()) == True):

                        if (len(e.text.split()) > 1):

                            sentence_x = sentence_x + " " + e.text.replace(" ","_")

                        else:

                            sentence_x = sentence_x + " " + e.text



                        sense_key = gold_mapping[e.attrib['id']]

                        wn_synset = wn.lemma_from_key(sense_key).synset()

                        wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

                        sentence_y = sentence_y + " " + bn2wn_mapping[wn_synset_id]

                

                

#                   sentence_y = sentence_y + " " + e.attrib['id'].strip()

#             sentence_x = Punctuation(sentence_x)

#             sentence_y = Punctuation(sentence_y)

        

            if(len(sentence_x.split()) == len(sentence_y.split())):     

                sentences.append(sentence_x)

                sequences.append(sentence_y)

                sentence_count += 1



            else:

                print("Mismatch")



        except Exception:

            continue 

            

        print("{:,d} sentences extracted...".format(sentence_count), end="\r")

        # It's safe to call clear() here because no descendants will be accessed

        elem.clear()

        # Also eliminate now-empty references from the root node to <Title> 

        while elem.getprevious() is not None:

            del elem.getparent()[0]

    print ("\nDone!")
preprocess_semcor(some_file_like,"../input/omstidata/semcoromsti.gold.key.txt", "../input/babelnetdata/babelnet2wordnet.tsv")

print(len(sequences))

print(len(sentences))
def Punctuation(string): 

  

    # punctuation marks 

    punctuations = '''!()[]{}|;'"\,<>./?@#$%^&*~'''

  

    # traverse the given string and if any punctuation 

    # marks occur replace it with null 

    for x in string.lower(): 

        if x in punctuations: 

            string = string.replace(x, "") 

    

    return string

            

clean_sentences = [Punctuation(s) for s in sentences]

print(clean_sentences[0:2])



clean_sequences = [Punctuation(s) for s in sequences]

print(clean_sequences[0:2])
from typing import Tuple, List, Dict

import tensorflow as tf

import numpy as np

from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping

from collections import Counter

from tensorflow.keras.layers import Dense, Input,Masking,LSTM, Embedding,Reshape, Dropout, Activation,TimeDistributed,Bidirectional,concatenate, GlobalMaxPool1D

from tensorflow.keras.models import Model,Sequential,load_model

from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

import pickle

from tensorflow.keras.optimizers import SGD

from  tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical 

from sklearn.model_selection import GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import collections

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
def tokenize_dataset(sentences,sequences):

    """

    Converts each sentences to words



    :param sentences: sentences

    :param sequences: sequences

    :return: encoded X  training set

    """

    all_sentences,all_sentence_tags = [],[]

    for s,t in zip(sentences,sequences):

        sentence,tags = s.split(),t.split()

        if (len(sentence) == len(tags)):

            all_sentences.append(np.array(sentence))

            all_sentence_tags.append(np.array(tags))

        else:

            print("Mismatch!")

    return all_sentences,all_sentence_tags
all_sentences,all_sentence_tags = tokenize_dataset(clean_sentences,clean_sequences)

print(all_sentences[0])

print(all_sentence_tags[0])
from sklearn.model_selection import train_test_split



train_sentences,train_tags = all_sentences, all_sentence_tags

train_sentences_X,train_tags_y = [], []
def vocabulary():  

    """

    This is the function to build the vocabulary of the dataset.

    

    :param unigram_path: The path to the file that contains the unigrams

    :return: None

    """

    words, tags = set([]), set([])

 

    for s in train_sentences:

        for w in s:

            words.add(w.lower())



    for ts in train_tags:

        for t in ts:

            tags.add(t)



    word2index = {w: i + 2 for i, w in enumerate(list(words))}

    word2index['-PAD-'] = 0  # The special value used for padding

    word2index['-OOV-'] = 1  # The special value used for OOVs



    tag2index = {t: i + 2 for i, t in enumerate(list(tags))}

    tag2index['-PAD-'] = 0  # The special value used to paddin

    tag2index['-OOV-'] = 1  # The special value used to paddin

    

 

    for s in train_sentences:

        s_int = []

        for w in s:

            try:

                s_int.append(word2index[w.lower()])

            except KeyError:

                s_int.append(word2index['-OOV-'])



        train_sentences_X.append(s_int)

 

#     for s in test_sentences:

#         s_int = []

#         for w in s:

#             try:

#                 s_int.append(word2index[w.lower()])

#             except KeyError:

#                 s_int.append(word2index['-OOV-'])



#         test_sentences_X.append(s_int)



    for s in train_tags:

        t_int = []

        for t in s:

            try:

                t_int.append(tag2index[t])

            except KeyError:

                t_int.append(tag2index['-OOV-'])

                

        train_tags_y.append(t_int)

                                 

#     for s in test_tags:

#         t_int = []

#         for t in s:

#             try:

#                 t_int.append(tag2index[t])

#             except KeyError:

#                 t_int.append(tag2index['-OOV-'])

                

#         test_tags_y.append(t_int)

    print("Train: ",train_sentences_X[0])

    print("Train tags: ",train_tags_y[0])

#     print("Test: ",test_sentences_X[0])

#     print("Test tags: ",test_tags_y[0])

 

    return word2index,tag2index

#     vocab = dict()

#     for line in file_path_x:

#         words = line.split()

#         for word in words:

#             word = word.lower()

#             if word not in vocab:

#                 vocab[word] = 1

#             else:

#                 vocab[word] += 1

#     return vocab
word_len , tag_len = vocabulary()

print(len(word_len))

print(len(tag_len))

MAX_LENGTH = len(max(train_sentences_X, key=len))

print(MAX_LENGTH)  # 27





# a = np.array(train_sentences_X)

# print(a.shape)

from keras.preprocessing.sequence import pad_sequences

 

train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')

# test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')

train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')

# test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

 

print(train_sentences_X[0])

# print(test_sentences_X[0])

print(train_tags_y[0])

# print(test_tags_y[0])
# a = np.array(train_sentences_X)

# print(a.shape)



# b = np.array(train_tags_y)

# print(b.shape)



from numpy import newaxis

reshaped_train_tags_y = train_tags_y[:, :, newaxis]

reshaped_train_tags_y.shape
for a,b in zip(train_sentences_X,train_tags_y):

    if a.shape != b.shape:

        print("Error")
from keras.models import Sequential

from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation

from keras.optimizers import Adadelta

from keras.callbacks import ModelCheckpoint

 



model = Sequential()

model.add(InputLayer(input_shape=(None, )))

model.add(Embedding(len(word_len), 128))

model.add(Bidirectional(LSTM(256, return_sequences=True,dropout=0.6,recurrent_dropout=0.4),merge_mode='sum'))

model.add(TimeDistributed(Dense(len(tag_len))))

model.add(Activation('softmax'))

 

model.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adadelta(1))



filepath="delta_weights-improvement.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

 

model.summary()
history = model.fit(train_sentences_X, reshaped_train_tags_y, batch_size=64, epochs=5,callbacks=callbacks_list,validation_split=0.01)
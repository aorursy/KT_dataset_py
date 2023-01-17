# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import print_function, division
from builtins import range
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('Loading comments ....')
data = pd.read_csv('../input/tunics.csv', error_bad_lines=False,
                    names = ["productId","title","description","imageUrlStr","mrp","sellingPrice","specialPrice","productUrl","categories","productBrand","productFamily","inStock","codAvailable","offers","discount","shippingCharges","deliveryTime","size","color","sizeUnit","storage","displaySize","keySpecsStr","detailedSpecsStr","specificationList","sellerName","sellerAverageRating","sellerNoOfRatings","sellerNoOfReviews","sleeve","neck","idealFor"])
data.head()
# Creating vocab. 
# as a 1st step, just giving unique no. to every word
def createVocab(field):
    not_found = 'nan'
    vocab = []
    for i in field:
        if (i not in vocab) and (str(i) != not_found):
            vocab.append(i)
    return vocab
# Selective fields which has info
idealFor_vocab = createVocab(data['idealFor'].values)
neck_vocab = createVocab(data['neck'].values)
sleeve_vocab = createVocab(data['sleeve'].values)
title_vocab = createVocab(data['title'].values)
productBrand_vocab = createVocab(data['productBrand'].values)
size_vocab = createVocab(data['size'].values)
color_vocab = createVocab(data['color'].values)
sizeUnit_vocab = createVocab(data['sizeUnit'].values)
displaySize_vocab = createVocab(data['displaySize'].values)
keySpecsStr_vocab = createVocab(data['keySpecsStr'].values)
detailedSpecsStr_vocab = createVocab(data['detailedSpecsStr'].values)
specificationList_vocab = createVocab(data['specificationList'].values)
sellerName_vocab = createVocab(data['sellerName'].values)

#summing up all the vocabs
t_vocab = title_vocab + productBrand_vocab + size_vocab + color_vocab + sizeUnit_vocab + displaySize_vocab +\
keySpecsStr_vocab + detailedSpecsStr_vocab + specificationList_vocab +\
sellerName_vocab + sleeve_vocab + neck_vocab +  sellerName_vocab + idealFor_vocab
# generating seq IDs for words in vocabulary
def genSeqNo(vocabulary):
    i=1.0
    vocab_seq = {}
    for word in vocabulary:
        if word not in vocab_seq:
            vocab_seq[word] = i
            i = i+1
    return vocab_seq
t_vocab_seq = genSeqNo(t_vocab)
'''
[("Vea Kupia Printed Women's Tunic", 1.0),
 ("U&F Solid Women's Tunic", 2.0),
 ("Taurus Printed Women's Tunic", 3.0),
 '''
# generating inverse relation of seq to words
inv_t_vocab_seq = {i:words for words, i in t_vocab_seq.items()}

''' Sample
[(1.0, "Vea Kupia Printed Women's Tunic"),
 (2.0, "U&F Solid Women's Tunic"),
 (3.0, "Taurus Printed Women's Tunic"),
'''
# function to replace words with its uniq seq ID
def createSeq(field):
    seq = []
    for word in field.values:
        if word in t_vocab_seq:
            seq.append(t_vocab_seq[word])
        else:
            seq.append(0)
    return seq

# replacing words with numeric values
neck_seq = createSeq(data['neck'])
title_seq = createSeq(data['title'])
productBrand_seq = createSeq(data['productBrand'])
size_seq = createSeq(data['size'])
color_seq = createSeq(data['color'])
sizeUnit_seq = createSeq(data['sizeUnit'])
displaySize_seq = createSeq(data['displaySize'])
keySpecsStr_seq = createSeq(data['keySpecsStr'])
detailedSpecsStr_seq = createSeq(data['detailedSpecsStr'])
specificationList_seq = createSeq(data['specificationList'])
sellerName_seq = createSeq(data['sellerName'])
sleeve_seq = createSeq(data['sleeve'])
idealFor_seq = createSeq(data['idealFor'])
# Preparing final input data having only numeric data
final_data = np.column_stack((title_seq, data['mrp'].values, data['sellingPrice'].values, productBrand_seq, 
                              data['shippingCharges'].values, size_seq, color_seq, sizeUnit_seq, displaySize_seq,
                              keySpecsStr_seq, detailedSpecsStr_seq, specificationList_seq, 
                              sellerName_seq, sleeve_seq, neck_seq, idealFor_seq))
#replacing NAN values
final_data = np.nan_to_num(final_data)

# checking if there is any NAN value left
np.argwhere(np.isnan(final_data))
#extracting and saving Seqn IDs
seq_data = np.array(data['productId']).astype(str)
# Importing keras libraries
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, UpSampling1D
from keras.models import Model
from keras.models import model_from_json
# Designing Auto-encoder Neural Net 
MAX_SEQUENCE_LENGTH = 16
num_words = len(final_data)
EMBEDDING_DIM = 1
embedding_matrix = t_vocab_seq

print('Building model...')

# train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
print('input', input_.shape)
x = Dense(16, activation='relu')(input_)
print('dense1', x.shape)
x = Dense(8, activation='relu')(x)
print('dense2', x.shape)

### Compressed version
x = Dense(4, activation='relu')(x)
print('dense3', x.shape)

'''
x = Dense(1, activation='sigmoid')(x)
print('dense4', x.shape)

x = Dense(4, activation='relu')(x)
print('dense5', x.shape)
'''
x = Dense(8, activation='relu')(x)
print('dense6', x.shape)
x = Dense(16, activation='relu')(x)
print('dense7', x.shape)

output = x
print('final', output.shape)

model = Model(input_, output)
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
#model.fit(final_data, final_data, epochs=3, validation_data=(x_test, x_test))

BATCH_SIZE = 200
epochs = 500
VALIDATION_SPLIT=0.2
print('Training model...')
r = model.fit(final_data,
              final_data,
              batch_size=BATCH_SIZE,
              epochs=epochs,
              validation_split=VALIDATION_SPLIT)
from keras import backend as K
# extracting third layer to extract the trained weights
compressed_layer = 3
get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[compressed_layer].output])
compressed = get_3rd_layer_output([final_data])[0]
print(compressed.shape)
print(final_data.shape)
print(final_data[:10])
print(compressed[:10])
from scipy.spatial import distance
print(distance.euclidean(compressed[2], compressed[1]))
# function to find and print duplicate product IDs

def find_dup(prodID):
    id = seq_data.tolist().index(prodID)
    y=0
    i = compressed[id]
    for j in compressed:
        if (id != y):
            #print(i, j, x, y)
            dist = distance.euclidean(i,j)
            #print(dist)
            if dist <10:
                #sentence = (i for i in sqn_data[y])
                #print(sentence)
                dup_list.append([seq_data[y], dist])
                #dup_dict[i].append(print(sqn_data[y]))
            #diff_mat[x][y] = dist
        y=y+1
    return dup_list
print(find_dup('TUNE9CG3CDTCKVHK'))
print(data.loc[data['productId'].isin(['LJGDWBRGTN25SSKG','LJGDWBRG6HMM9AAK','LJGDUUFVHDPDKN7H'])])

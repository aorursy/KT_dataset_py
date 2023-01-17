# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from keras.preprocessing.text import Tokenizer

from keras.engine.topology import Layer

from keras.preprocessing.sequence import pad_sequences

import numpy as np # linear algebra

from keras.optimizers import Adam, Nadam

import fasttext

import re

from keras import initializers

from nltk.tokenize import word_tokenize

from keras.layers import Embedding

from keras import backend as K

from keras.layers import Dense, Input, Flatten

from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed

from keras.models import Model

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")

test=pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")

sub=pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
train.head(2)
test.head(2)
print('[%s]' % ', '.join(map(str, train.columns.values.tolist())))

print("\n")

print('[%s]' % ', '.join(map(str, test.columns.values.tolist())))

print("\n")

print('[%s]' % ', '.join(map(str, sub.columns.values.tolist())))
target_columns = sub.columns.values[1:].tolist()

target_columns
xtrain=train[["question_title", "question_body","answer"]]

xtest=test[["question_title", "question_body","answer"]]

ytrain=train[target_columns]
xtrain
ytrain
xtest
def preprocess_text(sen):

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)

    

    sentence=sentence.lower()



    return sentence
for i in range(len(xtrain)):

    xtrain["question_title"][i]= preprocess_text(xtrain["question_title"][i])

    xtrain["question_body"][i]= preprocess_text(xtrain["question_body"][i])

    xtrain["answer"][i]= preprocess_text(xtrain["answer"][i])
for i in range(len(xtest)):

    xtest["question_title"][i]= preprocess_text(xtest["question_title"][i])

    xtest["question_body"][i]= preprocess_text(xtest["question_body"][i])

    xtest["answer"][i]= preprocess_text(xtest["answer"][i])
xtest
emb_file = fasttext.load_model('../input/fasttext-pretrained-word-vectors-english/wiki.en.bin')
trainl=len(xtrain)

testl=len(xtest)

fulld=xtrain.append(xtest,ignore_index=True)

fulld=fulld.stack().reset_index()[0]

import nltk

for i in range(len(fulld)):

    k=len(nltk.word_tokenize(fulld[i]))

    #lens.append(len(data[0][i].split(" ")))

    if(k<500):

        pass

    else:

        te=nltk.word_tokenize(fulld[i])[:500]

        fulld[i]=' '.join(te)

max_features = 50000 #number of words to keep. 1200 is the number of unique words in the corpus.

tokenizer = Tokenizer(nb_words=max_features, split=' ')

tokenizer.fit_on_texts(fulld.values)

X = tokenizer.texts_to_sequences(fulld.values)

X = pad_sequences(X, padding = 'post') #Zero padding at the end of the sequence

word_index = tokenizer.word_index
nb_words = len(word_index)+1

embedding_dimension = 300



embedding_matrix = np.zeros((nb_words, embedding_dimension))

for word, i in word_index.items():

    embedding_matrix[i,:] = emb_file.get_word_vector(word).astype(np.float32)
X=np.split(X,trainl+testl)
X=np.array(X)
xtrain1=X[0:trainl]

xtest1=X[trainl:trainl+testl]
ytrain=train[target_columns]
class AttentionLayer(Layer):

    """

    Hierarchial Attention Layer as described by Hierarchical Attention Networks for Document Classification(2016)

    - Yang et. al.

    Source: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

    Theano backend

    """

    def __init__(self,attention_dim=100,return_coefficients=False,**kwargs):

        # Initializer 

        self.supports_masking = True

        self.return_coefficients = return_coefficients

        self.init = initializers.get('glorot_uniform') # initializes values with uniform distribution

        self.attention_dim = attention_dim

        super(AttentionLayer, self).__init__(**kwargs)



    def build(self, input_shape):

        # Builds all weights

        # W = Weight matrix, b = bias vector, u = context vector

        assert len(input_shape) == 3

        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)),name='W')

        self.b = K.variable(self.init((self.attention_dim, )),name='b')

        self.u = K.variable(self.init((self.attention_dim, 1)),name='u')

        self.trainable_weights = [self.W, self.b, self.u]



        super(AttentionLayer, self).build(input_shape)



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, hit, mask=None):

        # Here, the actual calculation is done

        uit = K.bias_add(K.dot(hit, self.W),self.b)

        uit = K.tanh(uit)

        

        ait = K.dot(uit, self.u)

        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        

        if mask is not None:

            ait *= K.cast(mask, K.floatx())



        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        ait = K.expand_dims(ait)

        weighted_input = hit * ait

        

        if self.return_coefficients:

            return [K.sum(weighted_input, axis=1), ait]

        else:

            return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        if self.return_coefficients:

            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]

        else:

            return input_shape[0], input_shape[-1]
xtrain1.shape
embedding_layer = Embedding(len(word_index) + 1,300,weights=[embedding_matrix],

                            input_length=500,

                            trainable=True,

                            mask_zero=True)
# Words level attention model

word_input = Input(shape=(500,), dtype='int32',name='word_input')

word_sequences = embedding_layer(word_input)

word_gru = Bidirectional(LSTM(50, return_sequences=True),name='word_gru')(word_sequences)

word_dense = Dense(100, activation='relu', name='word_dense')(word_gru) 

word_att,word_coeffs = AttentionLayer(300,True,name='word_attention')(word_dense)

wordEncoder = Model(inputs = word_input,outputs = word_att)



# Sentence level attention model

sent_input = Input(shape=(3,500), dtype='int32',name='sent_input')

sent_encoder = TimeDistributed(wordEncoder,name='sent_linking')(sent_input)

sent_gru = Bidirectional(LSTM(50, return_sequences=True),name='sent_gru')(sent_encoder)

sent_dense = Dense(100, activation='relu', name='sent_dense')(sent_gru) 

sent_att,sent_coeffs = AttentionLayer(300,return_coefficients=True,name='sent_attention')(sent_dense)

sent_drop = Dropout(0.5,name='sent_dropout')(sent_att)

preds = Dense(30, activation='sigmoid',name='output')(sent_drop)



# Model compile

model1 = Model(sent_input, preds)

model1.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, clipnorm=4), metrics=['accuracy'])

print(wordEncoder.summary())

print(model1.summary())
model1.fit(xtrain1, ytrain,batch_size=50, epochs=3, verbose=1)
model1.predict(xtest1).shape
test_pred = model1.predict(xtest1)

print(test_pred.shape)
ids=test["qa_id"]
tem=pd.DataFrame(test_pred)
outs = tem.set_axis(target_columns, axis=1, inplace=False)
outs
final=pd.concat([ids,outs], axis=1)
final.to_csv('submission.csv', index=False)
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
import pandas as pd

import numpy as np

import re

from nltk import word_tokenize

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, Embedding

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords



from keras.layers import SimpleRNN,LSTM,CuDNNGRU,CuDNNLSTM,Conv1D,MaxPooling1D,Dropout

from keras import regularizers

from keras.layers import BatchNormalization

from keras import optimizers

from keras import initializers



from keras.callbacks import *

from keras import backend as K

import keras 







from keras.callbacks import *

from keras.optimizers import Adam









cols = ['sentiment','id','date','query_string','user','text']

df = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv', encoding='latin1', names=cols)
df.head()

df.info()

df['sentiment'].value_counts()
df.drop(['id','date','query_string','user'],axis = 1)
#Define a pattern



pat1= '#[^ ]+'

pat2 = 'www.[^ ]+'

pat3 = '@[^ ]+'

pat4 = '[0-9]+'

pat5 = 'http[^ ]+'

negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",   

                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",

                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",

                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",

                "mustn't":"must not"}





pattern = '|'.join((pat1,pat2,pat3,pat4,pat5))

neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
#Cleaning Data and removing Stop Words

stop_words = stopwords.words('english')

clean_tweets = []





for t in df['text']:

    t.lower()

    t = re.sub(pattern,'',t)

    t = neg_pattern.sub(lambda x: negations_dic[x.group()], t)

    t = word_tokenize(t)

    t = [x for x in t if len(x) >1]

    t = [x for x in t if x not in stop_words]

    t = [x for x in t if x.isalpha()]

    t = " ".join(t)

    t = re.sub("n't","not",t)

    t = re.sub("'s","is",t)

    clean_tweets.append(t)
df_clean = pd.DataFrame(clean_tweets,columns = ['text'])

df_clean['sentiment']=df['sentiment'].replace(4,1)
print(df_clean['text'].head(20))

df_clean['sentiment'].value_counts()    
length = []

for t in df_clean['text'] :

    l = len(re.findall(r'\w+', t))

    length.append(l)

np.percentile(length, [50,75,90,95,98])
x = df_clean['text']

y = df_clean['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
#Tokenization

tk = Tokenizer()

tk.fit_on_texts(x_train)



x_train_tok = tk.texts_to_sequences(x_train)

x_test_tok = tk.texts_to_sequences(x_test)



max_len = 20

x_train_pad = pad_sequences(x_train_tok, maxlen=max_len)

x_test_pad = pad_sequences(x_test_tok, maxlen=max_len)
unique_vocab = len(tk.word_index)

print(unique_vocab)
#Word Embedding

from gensim.models import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)

print('Found %s word vectors of word2vec' % len(word2vec.vocab))
# Testing Some Simliarities 

w2 = 'love'

word2vec.most_similar_cosmul(positive=w2)

w1 ='hate'

word2vec.most_similar_cosmul(positive=w1)



word2vec.similarity('woman', 'girl')




##https://github.com/bckenstler/CLR





class CyclicLR(keras.callbacks.Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())

embedding_dim = 300



embedding_matrix = np.zeros((unique_vocab+1, embedding_dim)) # intial embedding matrix with zeros \

                                                             #with dim (# of token word , # of features)

# Now get the feature for token words



for word, i in tk.word_index.items():      

    if word in word2vec.vocab:

        embedding_matrix[i] = word2vec[word]      #emb mat from 0 to 11549   if unique_vocab not +1



        

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))     #axis=1 --> row
from keras.layers import SpatialDropout1D

from keras.layers import advanced_activations

from keras.layers import TimeDistributed

from keras.layers import Conv2D

from keras.layers import Bidirectional

opt = optimizers.adam(lr=0.001)



model = Sequential()

model.add(Embedding(input_dim=unique_vocab+1, output_dim=embedding_dim, input_length=max_len,

                    weights=[embedding_matrix],trainable=False))







model.add(SpatialDropout1D((0.25)))



model.add(Conv1D(filters=300, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2,stride=1))

model.add(BatchNormalization()) 

model.add(SpatialDropout1D((0.5)))



model.add(Conv1D(filters=300, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2,stride=1))

model.add(BatchNormalization()) 

model.add(SpatialDropout1D((0.5)))







model.add(Conv1D(300,4,padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))

model.add(advanced_activations.LeakyReLU(alpha=0.3))

model.add(MaxPooling1D(pool_size=2,stride=1))

model.add(BatchNormalization()) 

model.add(SpatialDropout1D((0.5)))



model.add(Conv1D(300,4,padding='same',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))

model.add(advanced_activations.LeakyReLU(alpha=0.3))

model.add(MaxPooling1D(pool_size=2,stride=1))

model.add(BatchNormalization()) 

model.add(SpatialDropout1D((0.5)))







model.add(Bidirectional(CuDNNLSTM(256,kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),return_sequences=False)))





model.add(Dense(1,activation='sigmoid'))









model.compile(optimizer = opt,loss = 'binary_crossentropy',metrics =['accuracy'])

print(model.summary())
clr_triangular = CyclicLR(mode='triangular2',step_size = 8400)

history= model.fit(x=x_train_pad,y=y_train,validation_data=(x_test_pad,y_test),epochs =50,batch_size = 256,callbacks=[clr_triangular])
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

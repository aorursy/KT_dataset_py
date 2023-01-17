def f1():
    # Just to initialize Global Variables
    return
# The idea of set_all_parameters is to have one place 
# that you can select models,embeddings, evaluation methods and parameters etc
# in one place.
import time
def set_all_parameters():
    
    global epoc_used
    global mn             # Model Number - For essembling
    global batch_size
    global patience
    global nsplits
    global stop_split
    
    global num_lstm
    global num_dense
    global rate_drop_lstm
    global rate_drop_dense
    global rate_drop_spatial
    global loss
    global act
    global opt
    global met
    global es_mon
    global es_mode
    global Trainable
    global lr
    global method
    global model_list
    global maxlen
    global max_features
    global pretext_proc
    global bstart


    #
    # Choose Evaluate Method - Results change using different methods
    #
    #Each Weight will be applied and results will be divided by the sum of weigths**
    #method="0" # Standard Fit 
    #method="1" # EarlyStop F1**
    #method="2" # Manual Epochs with EarlyStop
    #method="3" # Manual Epochs with EarlyStop + CLR
    #method="4" # EarlyStop F1 + CLR*
    #method="5" # Manual Epochs
    #method="5" # EarlyStop + 
    sm ="6" # EarlyStop F1
    #
    # Possible embeddings ["Glove","Paragram","Google","Wiki","Combined","Concatenated"]
    # I am using Glove an Paragram only. 
    # To use Google and Wiki - search and delete comments in #embedding_matrix_2 = load_embedding("Wiki",word_index)
    #                           and #embedding_matrix_4 = load_embedding("Google",word_index)
    # Define Standard Models Parameters to be used by models.     
    #
    ssh = 50    #Standard Size of Hidden Layers - (Lstm or Gru)
    ssd = 50   #Standard Size of Hidden Layers Dense 
    sdh = 0.1  #Standard drop-out rate (Lstm or Gru) 
    sdd = 0.1  #Standard drop-out rate Dense 
    sds =0.1   #Standard drop-out rate Spatial
    ##
    ## Model Loss
    #  Optimizer Options loss = 'binary_crossentropy'      
    loss_bin='binary_crossentropy'      
    ##
    ## Model Optimizer
    #  Optimizer Options opt='adam' or 'rmsprop'
    opt_adam="adam"
    opt_rmsprop="rmsprop"
    ## Model Metric
    ## Metric Options met= ["accuracy"] or [f1] or ["accuracy",f1]  
    #
    met_acc=["accuracy"]
    met_f1=[f1]
    ## Early Stop Monitor and Early Stop Mode
    ## Early Stop Monitor Options es_mon = "val_f1" es_mode="max"  for met=[f1]
    #                             es_mon = "val_acc" es_mode="max" for met=["accuracy"]   
    #                             es_mon = "val_loss"
    mon_acc = "val_acc"
    mode_acc = "max"
    mon_f1 = "val_f1"
    mode_f1 = "max"
    mon_loss ="val_loss"
    mode_loss="min"
    ###  4 Splits = Validation Split=0.25%
    ###  5 Splits = Validation Split=0.20 %
    ### 10 Splits = Validation Split=0.10 %
    ### 12 Splits = Validation Split=0.0833 %
    ns=4  # Standar Number of Splits
    ### For Stratified k-fold - Validation -> nsplits=stop_splits
    ss=4   # Stop at split stop_split ina stratified k-fold

    ###
    ### Models Parameters
    ###
    bs=1024    #Standard Batch Size
    ne=8       #Standard number of epochs
    pl=3       #Standard Patience - How many time insist running epochs after not improving metric

    #Possible models ["0","1","2","3","4","5","6","7","8","9","14","15","16","17","18","19"]
    # model list parameters - does not use bellow
    # model_list.append([model_number, weight, model_eval, embeddings,epochs,patience,batch_size, size hidden, size dense, drop hidden, drop dense, drop spatial, loss,optimizer,metric])
   

    bstart = time.time()
    #
    # Create the list of models that will run , at the end it essembles them with the defined weights.
    #
    #
    model_list=[]
    #model_list.append(["18",1,sm,"Combine",ne,pl,bs,ns,ss,ssh,ssd,sdh,sdd,sds,loss_bin,opt_rmsprop,met_acc,mon_acc,mode_acc])
    # or 
    model_list.append(["18",1,"6","Concatenated",10,1,1024,4,4,50,50,0.1,0.1,0.15,'binary_crossentropy','rmsprop',["accuracy"],'val_acc','max'])
    # or
    #model_list.append(["18",1,"5","Combine",ne,pl,bs,ns,ss,ssh,ssd,sdh,sdd,sds,loss_bin,opt_rmsprop,met_acc,mon_acc,mode_acc])
    #model_list.append(["11",1,"6","Combine",ne,pl,bs,ns,ss,ssh,ssd,sdh,sdd,sds,loss_bin,opt_rmsprop,met_acc,mon_loss,mode_loss])

    ###
    ###  Load Train,Val and Test 
    ###
    maxlen = 75 # Maximum Sequence Size 
    max_features = None # Maximum Number of Words in Dictionary
    pretext_proc=True
    
    ###
    ###  Embeddings
    ###
    Trainable=False  # Embedding Layers trainable(True) or not(False)

set_all_parameters()
from IPython.display import display, HTML
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split,StratifiedKFold
from tqdm import tqdm
import math
from datetime import timedelta
import time
from datetime import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import colorama
from colorama import Fore

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten, Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, Reshape, Concatenate
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Bidirectional, GlobalMaxPool1D,GlobalMaxPooling1D,GlobalAveragePooling1D ,Conv1D, MaxPooling1D, GRU,CuDNNLSTM,CuDNNGRU, Reshape, MaxPooling1D,AveragePooling1D
from keras.optimizers import RMSprop, SGD, Nadam, Adamax, Adam
from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints
from keras.layers import Conv2D, MaxPool2D
import keras.backend as K
%matplotlib inline
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, merge, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
class CyclicLR(Callback):
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
    
import operator 
#for ele in t.word_counts:
#  print(ele,t.word_counts[ele])
def check_coverage(vocab,embeddings_index):
    
    try:
        print ("Len - embeddings_index:",len(embeddings_index))
    except :
        print ( "Len - embeddings_index:",len(embeddings_index.index2word))
    print ("Len -",len(vocab))
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            
            oov[word] = vocab[word]
            i += vocab[word]
            pass
    print ("Words Found:",k)
    print ("Words Not Found:",i)
    print ("Total Words:",i+k)
    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
########################################
## process texts in datasets
########################################
import re


def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

def ReplaceThreeOrMore(s):
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
    return pattern.sub(r"\1", s)

def splitstring(s):
    # searching the number of characters to split on
    proposed_pattern = s[0]
    for i, c in enumerate(s[1:], 1):
        if c != " ":
            if proposed_pattern == s[i:(i+len(proposed_pattern))]:
                # found it
                break
            else:
                proposed_pattern += c
    else:
        exit(1)

    return proposed_pattern

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
#Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)

#regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


def clean_text_NOTUSED(x):

    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x
def text_to_wordlist(text,to_lower=True, rem_urls=False, rem_3plus=False,
                     clean_t=True, clean_num=True,mispelling=True,rem_specwords= False,
                     split_repeated=True, rem_special=False, rep_num=False, 
                     man_adj=True, rem_stopwords=False, stem_snowball=False,
                     stem_porter=False, lemmatize=False):

    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    if rem_urls:
        text = remove_urls(text)
    if to_lower:    
        text = text.lower()
    if rem_3plus:    
        text = ReplaceThreeOrMore(text)
        
    if clean_t:
        text= clean_text(text)
        
    if clean_num:
        text= clean_numbers(text)    
        
    if mispelling:
        text= replace_typical_misspell(text)

    if man_adj: 
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

    # split them into a list
    text = text.split()
    
    if split_repeated:
        for i, c in enumerate(text):
            text[i]=splitstring(c)
    
    if rem_specwords:    
        to_remove = ['a','to','of','and']
        text = [w for w in text if not w in to_remove]
        
    # Optionally, remove stop words
    if rem_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    
    #Remove Special Characters
    if rem_special: 
        text=special_character_removal.sub('',text)
    
    #Replace Numbers
    if rep_num:     
        text=replace_numbers.sub('n',text)

    # Optionally, shorten words to their stems
    if stem_snowball:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    if stem_porter:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in text.split()])
        
    if lemmatize:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w) for w in text.split()])   
 
    # Return a list of words
    return(text)
import pickle
def save_tokenizer( file, tokenizer):
    # saving
    with open(file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer( file, tokenizer):
    # loading
    with open(file, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
def load_and_prec(PreProcess=False):
    global max_features 
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## split to train and val
    #train_fit, train_val = train_test_split(train_df, test_size=0.08, random_state=2018)
    train_X=train_df["question_text"].values
    test_X=test_df["question_text"].values
    
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    if PreProcess:
        train_questions = []
        for text in train_X:
            train_questions.append(text_to_wordlist(text))  
        test_questions=[]
        for text in test_X:
            test_questions.append(text_to_wordlist(text)) 
        tokenizer.fit_on_texts(train_questions+test_questions)   
        train_X = tokenizer.texts_to_sequences(train_questions)
        test_X = tokenizer.texts_to_sequences(test_questions) 
    else:
        tokenizer.fit_on_texts(list(train_X)+list(test_X))
        train_X = tokenizer.texts_to_sequences(train_X)
        test_X = tokenizer.texts_to_sequences(test_X)

    print(len(train_X), 'train sequences')
    print(len(test_X), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, train_X)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, test_X)), dtype=int)))
    print('Max train sequence length: {}'.format(np.max(list(map(len, train_X)))))
    print('Max test sequence length: {}'.format(np.max(list(map(len, test_X)))))  
    
    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
  
    print(len(tokenizer.word_index))
    print(len(tokenizer.word_counts))
    if (max_features==None):
        max_features = len(tokenizer.word_index)+1
    #save_tokenizer('tokenizer.pickle',tokenizer)
    return train_X, test_X, train_y, test_df, tokenizer.word_index
########################################
## BASE Model
########################################
def create_model0(embedding_matrix):

    mdln="0-Bidirec(CuDNNGRU)-GlobalMax-Dense-Dropout"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    
    x = LSTM(num_lstm, return_sequences=True)(embedding_layer)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=met)
       
    return model, mdln
def create_model1(embedding_matrix):
    
    mdln="1-LSTM-Dropout-Attention-Dense-Dropout-BatchNormalization"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

    x = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True)(embedding_layer)
    x = Dropout(rate_drop_lstm)(x)
    x = Attention(maxlen)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = BatchNormalization()(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)
    
    return model, mdln
########################################
## BASE Model
########################################
def create_model2(embedding_matrix):

    mdln="2-Bidirect(CuDNNLSTM)-GlobalMax-Dense-Dropout-Dense-Dropout"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

    #x = Bidirectional(LSTM(num_dense, return_sequences=True, dropout=rate_drop_dense, recurrent_dropout=rate_drop_lstm))(embedding_layer)
    x = Bidirectional(CuDNNLSTM(num_lstm, return_sequences=True))(embedding_layer)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)
    
    return model, mdln
def create_model3(embedding_matrix):
    
    mdln="3-Bidirect(CuDNNLSTM)-Bidirect(CuDNNLSTM)-Attention-Dense"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    num_lstm2=int(num_lstm/2)
    x = Bidirectional(CuDNNLSTM(num_lstm, return_sequences=True))(embedding_layer)
    x = Bidirectional(CuDNNLSTM(num_lstm2, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(num_dense, activation="relu")(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3),metrics=met)

    return model, mdln
def create_model4( embedding_matrix):
    
    mdln="4-Reshape-Concat(Conv2D(Filters)+MaxPool2d)-Flatten-Dropout"
    
    filter_sizes = [1,2,3,5]
    num_filters = 36

    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    x = Reshape((maxlen, embedding_matrix.shape[1], 1))(embedding_layer)
    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embedding_matrix.shape[1]),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    preds = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)
    return model, mdln
from keras.models import Sequential

def create_model5(embedding_matrix):

    mdln="05-conv-max-conv-max-bulstm-max-dd"
    
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

    #x = Dropout(0.2)(embedded_sequences)
    #x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
    x = Dropout(0.2)(embedding_layer)
    x = Conv1D(filters=embedding_matrix.shape[1], kernel_size=4, padding='same', activation='relu')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=embedding_matrix.shape[1], kernel_size=4, padding='same', activation='relu')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    #x = GRU(32)(main)
    #x = Dense(32, activation="relu")(x)
    #x = BatchNormalization()(x)
    #x = MaxPooling1D(pool_size=4)(x)
    #x = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True)(x)
    x = Bidirectional(CuDNNLSTM(num_lstm, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)

    return model, mdln
def create_model6( embedding_matrix):

    mdln="6-Bidirect(CuDNNGRU)-Cocat(GlobalMax+Global Avarage)- Dense - Dropout"

    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)


    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
  

    tower_1 = GlobalMaxPool1D()(x)
    tower_2 = GlobalAveragePooling1D()(x)
    
    output = concatenate([  tower_1, tower_2])

    x = Dense(num_dense, activation="relu")(output)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(1, activation="sigmoid")(x)                         

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=met)

   
    return model, mdln
def create_model7( embedding_matrix):
    
    mdln="7-Bidirect(CuDNNGRU)-Attention-Dense-Dropout"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
    x = Attention(maxlen)(x) # New
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=met)
    
    return model , mdln   
def create_model8( embedding_matrix):
    
    mdln="8-Bidirect(CuDNNGRU)-Concat(GlobalMax+GlobalAverage)-Dense-Dropout"    
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(num_dense, activation="relu")(conc)
    conc = Dropout(rate_drop_dense)(conc)
    preds = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=met)
    
    return model, mdln
def create_model9(embedding_matrix):
    mdln="9-Bidirect(CuDNNGRU)-Bidirect(CuDNNGRU)-Bidirect(CuDNNGRU)-Attention-Dense"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    num_lstm2=int(num_lstm)
    num_lstm12=num_lstm2+int(num_lstm2/2)
    x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)
    x = Bidirectional(CuDNNGRU(num_lstm12, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(num_lstm2, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=met)
    
    return model, mdln
def create_model10(embedding_matrix):

        mdln="10-Spatial-Bidirect(CuDNNGRU)-conv2-maxpool-attavr-average-concatenate-drop-dense"      
        input_layer = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

    
        recurrent_units = 64
        filter1_nums = 128

        embedding_layer = SpatialDropout1D(rate_drop_spatial)(embedding_layer)
        rnn_1 = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(embedding_layer)

        #conv_1 = Conv1D(filter1_nums, 1, kernel_initializer="uniform", padding="valid", activation="relu", strides=1)(rnn_1)
        #maxpool = GlobalMaxPooling1D()(conv_1)
        #attn = AttentionWeightedAverage()(conv_1)
        #average = GlobalAveragePooling1D()(conv_1)

        conv_2 = Conv1D(filter1_nums, 2, kernel_initializer="normal", padding="valid", activation="relu", strides=1)(rnn_1)
        maxpool = GlobalMaxPooling1D()(conv_2)
        attn = AttentionWeightedAverage()(conv_2)
        average = GlobalAveragePooling1D()(conv_2)

        concatenated = concatenate([maxpool, attn, average], axis=1)
        x = Dropout(rate_drop_dense)(concatenated)
        x = Dense(num_dense, activation="relu")(x)
        output_layer = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=5, decay=1e-5)
        model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=met)
        return model, mdln
from keras.initializers import *
def create_model11(embedding_matrix):

        mdln="11-Capsule"      
        input_layer = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

 
        x = SpatialDropout1D(rate=rate_drop_spatial)(embedding_layer)
        x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True, 
                                    kernel_initializer=glorot_normal(seed=123000), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

        x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)

        x = Flatten()(x)

        x = Dense(num_dense, activation='relu', kernel_initializer=glorot_normal(seed=123000))(x)
        x = Dropout(rate_drop_dense)(x)
        x = BatchNormalization()(x)

        preds = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer=Adam(),)
        return model,mdln
def create_model12(embedding_matrix):
    
    mdln="12-SpatialDropout+Bidirect(CudNNGRU)+Bidirect(CudNLSTM)+Attention+Attention+Dense+Dropout"
       
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    x = SpatialDropout1D(rate_drop_spatial)(embedding_layer)
    x = Bidirectional(CuDNNLSTM(num_lstm, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(x)
    
    atten_1 = AttentionWeightedAverage()(x) # skip connect
    atten_2 = AttentionWeightedAverage()(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(num_dense, activation="relu")(conc)
    conc = Dropout(rate_drop_dense)(conc)
    preds = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=met)
    
    return model,mdln
from keras.initializers import he_normal, he_uniform,  glorot_normal,  glorot_uniform

def create_model13(embedding_matrix):
    
    mdln="13-SpatialDropout+Bidirect(CudNNGRU)+Bidirect(CudNLSTM)+Attention+Attention+Dense+Dropout"
       
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    x = SpatialDropout1D(rate_drop_spatial)(embedding_layer)
    x = Bidirectional(CuDNNLSTM(num_lstm, kernel_initializer=glorot_uniform(seed = 2018), return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(num_lstm,  kernel_initializer=glorot_uniform(seed = 2018),return_sequences=True))(x)
    
    atten_1 = Attention(maxlen)(x) # skip connect
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(num_dense, kernel_initializer=he_uniform(seed=2018), activation="relu")(conc)
    conc = Dropout(rate_drop_dense)(conc)
    preds = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=met)
    
    return model,mdln
def create_model14(embedding_matrix):

        mdln="14-Bidirect(CuDNNLSTM)-SpatialDropout-GlobalMax-BatchNorm-Dense-Dropout"      
        input_layer = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

        # we add a GlobalMaxPool1D, which will extract information from the embeddings
        # of all words in the document
        x = CuDNNLSTM(num_lstm, return_sequences=True)(embedding_layer)
        x = SpatialDropout1D(rate_drop_spatial)(x)
        x = GlobalMaxPool1D()(x)

        # normalized dense layer followed by dropout
        x = BatchNormalization()(x)
        x = Dense(num_dense)(x)
        x = Dropout(rate_drop_dense)(x)

        # We project onto a six-unit output layer, and squash it with sigmoids:
        preds = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=preds)
        model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)
                                         
        
        return model, mdln
def create_model15(embedding_matrix):
### https://www.kaggle.com/yekenot/pooled-gru-fasttext
        mdln="15-SpatialDropout-Bidirect(CudNNGRU)+Concat(GlobalMax+GlobalAverage)+Dense+Dropout"
        input_layer = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
   
        x = SpatialDropout1D(rate_drop_spatial)(embedding_layer)
        x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)        
        conc = concatenate([avg_pool, max_pool])
        x = Dense(num_dense, activation="relu")(conc)
        x = Dropout(rate_drop_dense)(x)
        preds = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=input_layer, outputs=preds)
        model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)
     
        return model, mdln
def create_model16(embedding_matrix):

    mdln="16-Bidirect(CudNNGRU)+GlobalMax+Dense+Dropout"
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)

    x = Bidirectional(CuDNNLSTM(num_lstm, return_sequences=True))(embedding_layer)
    x = GlobalMaxPool1D()(x)
    x = Dense(num_dense, activation="relu")(x)
    x = Dropout(rate_drop_dense)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)
    
    return model, mdln
def create_model17(embedding_matrix ):
    
        mdln="17-Bidirec(CuDNNLSTM)-GlobalMax-Dense-Dropout-Dense-Dropout"
        input_layer = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
        x = Bidirectional(CuDNNLSTM(num_lstm, return_sequences=True))(embedding_layer)
        x = GlobalMaxPool1D()(x)
        x = Dense(num_dense, activation="relu")(x)
        x = Dropout(rate_drop_dense)(x)
        x = Dense(num_dense, activation="relu")(x)
        x = Dropout(rate_drop_dense)(x)
        preds = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=preds)
        model.compile(loss='binary_crossentropy',optimizer=opt, metrics=met)

        return model, mdln  
def create_model18(embedding_matrix):
    
    mdln="18-SpatialDropout+Bidirect(CudNNGRU)+Bidirect(CudNLSTM)+Attention+Attention+Dense+Dropout"
       
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=Trainable)(input_layer)
    x = SpatialDropout1D(rate_drop_spatial)(embedding_layer)
    x = Bidirectional(CuDNNLSTM(num_lstm, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(x)
    
    atten_1 = Attention(maxlen)(x) # skip connect
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(num_dense, activation="relu")(conc)
    conc = Dropout(rate_drop_dense)(conc)
    preds = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=met)
    
    return model,mdln
def create_model19( embedding_matrix):
    
    
        mdln="19-SpatialDropout+Bidirect(CudNNGRU)+Conv1D+Concat(GlobalMax+GlobalAverage)+Dense+Dropout"
        
        input_layer = Input(shape=(maxlen,))
        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                    weights=[embedding_matrix], trainable=Trainable)(input_layer)

        x = SpatialDropout1D(rate_drop_spatial)(embedding_layer)
        x = Bidirectional(CuDNNGRU(num_lstm, return_sequences=True))(x)

        #x = Conv1D(filters=num_lstm, kernel_size=2, padding='same', activation='relu')(x)
        x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
        tower_1 = GlobalMaxPool1D()(x)
        tower_2 = GlobalAveragePooling1D()(x)

        output = concatenate([  tower_1, tower_2])

        out = Dense(num_dense, activation="relu")(output)
        out = Dropout(rate_drop_dense)(out)
        preds = Dense(1, activation="sigmoid")(out)                         

        model = Model(inputs=input_layer, outputs=preds)
        #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=met)               


        return model, mdln    
        
       
def create_model (label,embedding_matrix):
    
            if (label=="0"):
                return create_model0(embedding_matrix)
            if (label=="1"):
                return create_model1(embedding_matrix)
            if (label=="2"):
                return create_model2(embedding_matrix)
            if (label=="3"):
                return create_model3(embedding_matrix)            
            if (label=="4"):
                return create_model4(embedding_matrix)
            if (label=="5"):
                return create_model5(embedding_matrix)
            if (label=="6"):
                return create_model6(embedding_matrix)
            if (label=="7"):
                return create_model7(embedding_matrix)  
            if (label=="8"):
                return create_model8(embedding_matrix)
            if (label=="9"):
                return create_model9(embedding_matrix) 
            if (label=="10"):
                return create_model10(embedding_matrix)            
            if (label=="11"):
                return create_model11(embedding_matrix)                
            if (label=="12"):
                return create_model12(embedding_matrix)                
            if (label=="13"):
                return create_model13(embedding_matrix)                
            if (label=="14"):
                return create_model14(embedding_matrix)                
            if (label=="15"):
                return create_model15(embedding_matrix)                
            if (label=="16"):
                return create_model16(embedding_matrix)   
            if (label=="17"):
                return create_model17(embedding_matrix) 
            if (label=="18"):
                return create_model18(embedding_matrix) 
            if (label=="19"):
                return create_model19(embedding_matrix) 
            return None,"None"
            
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
def best_F1 (v_y ,t_y,show_plot=False):
    bs=0
    bt=0
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        score = metrics.f1_score(v_y, (t_y>=thresh).astype(int))
        if score >= bs:
            bt = thresh
            bs = score  
    if (show_plot):        
        print("\n Best F1 score at threshold %2.4f is %2.4f \n" % (bt, bs))
        plot_confusion_matrix(v_y, np.array(pd.Series(t_y.reshape(-1,)).map(lambda x:1 if x>=bt else 0)))
        d=confusion_matrix(v_y, (t_y>=bt).astype(int))
        print ("Total:",d.sum(),"/Insincere:",d[1,:].sum(),"/Sincere:",d[0,:].sum())
        print ("Total Erros:",(d[0,1]+d[1,0]),"/Insincere: %2f"%((d[0,1]/(d[0,1]+d[1,1]))),"/Sincere: %2f" % ((d[1,0]/(d[1,0]+d[0,0]))))
        
    return bt, bs
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def rec_results (Model,Embedding,Thresh,Score,StartTime):
    global resultsdf
    resultsdf = resultsdf.append({'Model':Model,
                                  'Method':method,
                                  'Embedding':Embedding,
                                  'Pretext':pretext_proc,
                                  'Weigth':wgt,
                                  'F1':Score,
                                  "Threshold":Thresh,
                                  "Duration":str(round((time.time()-StartTime)/60,0)),
                                  'Epochs':epochs,
                                  'EpochsUsed':epoc_used,
                                  'Trainable':Trainable,
                                  'Opt':opt,
                                  'MaxLength':maxlen,
                                  'MaxFeatures':max_features,                                  
                                  'batch_size':batch_size,
                                  'patience':patience,
                                  'num_lstm':num_lstm,
                                  'rate_drop_lstm':rate_drop_lstm,
                                  'num_dense':num_dense,
                                  'rate_drop_dense':rate_drop_dense,
                                  'rate_drop_spatial':rate_drop_spatial,
                                  'opt':opt,
                                  'met':met,
                                  'es_mon':es_mon,
                                  'es_mode':es_mode,
                                  'Date':datetime.now().strftime("%d-%m-%Y %H:%M")}, ignore_index=True)
def load_embedding (emb,word_index) :
        estart = time.time()
        print('Indexing '+emb+' vectors')
        if (emb=="Glove"):
            EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
            def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
            embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
        if (emb=="Google"):
            EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
            embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
        if (emb=="Paragram"):
            EMBEDDING_FILE =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt' 
            def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
            embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)
        if (emb=="Wiki"):
            EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'    
            def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
            embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
        
        print("Vector",EMBEDDING_FILE )
        print("End Indexing:",(str(timedelta(seconds=(time.time()-estart)))) )
        estart = time.time()
        print('Preparing embedding matrix')
        
        out_of_features=0
        out_of_embedding=0
        in_embedding=0
        if (emb=="Wiki" or emb=="Glove" or emb=="Paragram"):
            all_embs = np.stack(embeddings_index.values())
            emb_mean,emb_std = all_embs.mean(), all_embs.std()
            embed_size = all_embs.shape[1]

            #word_index = tokenizer.word_index
            nb_words = min(max_features, len(word_index))+1
            np.random.seed(2018)
            embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
            for word, i in word_index.items():
                if i >= max_features: 
                    out_of_features+=1
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None: 
                    embedding_matrix[i] = embedding_vector
                    in_embedding+=1
                else:
                    out_of_embedding+=1
        else: #"Google"
            embed_size=300
            #word_index = tokenizer.word_index
            nb_words = min(max_features, len(word_index))+1
            np.random.seed(2018)
            embedding_matrix= (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
            #embedding_matrix= (np.random.normal(nb_words, embed_size) - 0.5) / 5.0
            for word, i in word_index.items():
                if i >= max_features:
                    out_of_features+=1
                    continue
                if word in embeddings_index:
                    embedding_vector = embeddings_index.get_vector(word)
                    embedding_matrix[i] = embedding_vector
                    in_embedding+=1
                else:
                    out_of_embedding+=1
        print("Total Vocabulary:",len(word_index))
        print("Out of Features:", out_of_features)
        print ("Out of Embeddings:",out_of_embedding)
        print ("Mapped Embeddings:",in_embedding)
        print("End Preparing embedding matrix:",(str(timedelta(seconds=(time.time()-estart)))) )
        del embeddings_index
        return embedding_matrix
def evaluate_model (label,emb, embedding_matrix,train_X,train_y,t_X, t_y, v_X, v_y, epochs, batch_size,patience):

        global resultsdf 
        global pred_val_ytrain_y
        global prev_test_y
        global met

        best_thresh = 0.0
        best_score = 0.0
        
        mstart = time.time()
        model=None
        model,mdln = create_model(label, embedding_matrix)
        print ("")
        print (mdln+'-'+emb)
        #print(modelx.summary())
        print("Model Fitting:")
        model.fit(t_X, t_y, batch_size=batch_size, epochs=epochs, validation_data=(v_X, v_y))
        print("Predicting Values:")
        pvy = model.predict([v_X], batch_size=1024, verbose=2)
        best_thresh,best_score=best_F1(v_y,  pvy[:,0] , False)
        print("\n")
        pty = model.predict([test_X], batch_size=1024, verbose=2)
        rec_results (mdln,emb,best_thresh,best_score,mstart)
        import gc;           
        del model
        gc.collect()
        time.sleep(10)  
        return pvy[:,0],pty[:,0],best_score,best_thresh
def evaluate_model1 (label,emb, embedding_matrix,train_X,train_y,t_X,t_y,v_X,v_y, epochs, batch_size, patience=0):

        global resultsdf 
        global pred_val_y
        global prev_test_y
        global met 
        global epoc_used
        best_thresh = 0.0
        best_score = 0.0
        mstart = time.time()

        
        model=None
        model,mdln = create_model(label,embedding_matrix)
        print ("")
        print (mdln+'-'+emb)

        model_checkpoint = ModelCheckpoint('./model.hdf5', monitor=es_mon, mode=es_mode,
                                      verbose=True, save_best_only=True, 
                                      save_weights_only=False)
    
        early_stopping = EarlyStopping(monitor=es_mon, mode=es_mode, patience=patience,verbose=True)
        callbacks = [ early_stopping, model_checkpoint]
        print("Model Fitting:")
        hist= model.fit(t_X, t_y, batch_size=batch_size, epochs=epochs,
              shuffle=True, verbose=True, validation_data=(v_X, v_y),
              callbacks=callbacks)
        #print("History\n",hist.history.keys()) 
        #print("F1",hist.history['f1'])
        epoc_used=len(hist.history[es_mon])
        ### Getting the Best Model
        model.load_weights('./model.hdf5')   

        print("Predicting Values:")
        #predictions_valid = bestmodel.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        pvy = model.predict([v_X], batch_size=batch_size, verbose=2)
        best_thresh,best_score=best_F1(v_y,pvy[:,0] , False)             
        pty = model.predict([test_X], batch_size=1024, verbose=2)
        rec_results (mdln,emb,best_thresh,best_score,mstart)
        import gc;           
        del model
        gc.collect()
        time.sleep(10)
        return  pvy[:,0],pty[:,0],best_score,best_thresh
def evaluate_model2 (label,emb, embedding_matrix,train_X,train_y, t_X, t_y, v_X, v_y, epochs, batch_size, patience=0):
    
    global resultsdf 
    global pred_val_y
    global prev_test_y
    global met

    best_thresh = 0.0
    best_score = 0.0

    mstart = time.time()
    model=None
    model,mdln = create_model(label, embedding_matrix)
    print ("")
    print (mdln+'-'+emb)
    bs=0
    pt=0
    print("Model Fitting:")
    for e in range(epochs):
        model.fit(t_X, t_y, batch_size=batch_size, epochs=1, validation_data=(v_X, v_y))
        print("Predicting Values:")
        pvy = model.predict([v_X], batch_size=1024, verbose=0)
        best_thresh,best_score=best_F1(v_y,pvy[:,0], False )
        print("epoch ",e+1,"/",epochs)
        if (best_score > bs):          
            print("F1 Score Improved from %2.4f to %2.4f" % (bs,best_score))
            bs = best_score
            bt = best_thresh
            epoc_used=e+1
            model.save_weights('./model.hdf5') 
        else:
            print("F1 Score not improved")
            pt=pt+1
            if (pt>patience):
                break
    model.load_weights('./model.hdf5')             
    pty = model.predict([test_X], batch_size=1024, verbose=0)
    rec_results (mdln,emb,bt,bs,mstart) 
    import gc;           
    del model
    gc.collect()
    time.sleep(10)
    return pvy[:,0],pty[:,0],bs,bt
def evaluate_model3 (label,emb, embedding_matrix,train_X,train_y, t_X, t_y, v_X, v_y, epochs, batch_size, patience=0):
    
    global resultsdf 
    global pred_val_y
    global prev_test_y
    global met
    global clr
    global epoc_used 
    clr = CyclicLR(base_lr=0.001, max_lr=0.01,
               step_size=300., mode='exp_range',
               gamma=0.99994)
    callback = [clr,]

    best_thresh = 0.0
    best_score = 0.0

    mstart = time.time()
    model=None
    model,mdln = create_model(label, embedding_matrix)
    print ("")
    print (mdln+'-'+emb)
    bs=0
    pt=0
    print("Model Fitting:")
    for e in range(epochs):
        model.fit(t_X, t_y, batch_size=batch_size, epochs=1, validation_data=(v_X, v_y),callbacks = callback,verbose=2)
        print("Predicting Values:")
        pvy = model.predict([v_X], batch_size=1024, verbose=0)
        print("F1 - Score:")
        best_thresh,best_score=best_F1(v_y,pvy[:,0] )
        print("epoch ",e+1,"/",epochs)
        if (best_score > bs):          
            print("F1 Score Improved from %2.4f to %2.4f" % (bs,best_score))
            bs = best_score
            bt = best_thresh
            epoc_used=e+1
            model.save_weights('./model.hdf5') 
        else:
            print("F1 Score not improved")
            pt=pt+1
            if (pt>patience):
                break
    model.load_weights('./model.hdf5')             
    pty = model.predict([test_X], batch_size=1024, verbose=0)
    rec_results (mdln,emb,bt,bs,mstart) 
    import gc;           
    del model
    gc.collect()
    time.sleep(10)
    return pvy[:,0],pty[:,0],bs,bt
def evaluate_model4 (label,emb, embedding_matrix,train_X,train_y,t_X,t_y,v_X,v_y, epochs, batch_size, patience=0):

        global resultsdf 
        global pred_val_y
        global prev_test_y
        global met 
        global clr
        global epoc_used 
        clr = CyclicLR(base_lr=0.001, max_lr=0.01,
                   step_size=300., mode='exp_range',
                   gamma=0.99994)
        best_thresh = 0.0
        best_score = 0.0
        
        mstart = time.time()
        model=None
        model,mdln = create_model(label,embedding_matrix)
        print ("")
        print (mdln+'-'+emb)
        model_checkpoint = ModelCheckpoint('./model.hdf5', monitor=es_mon, mode=es_mode,
                                      verbose=True, save_best_only=True, 
                                      save_weights_only=False)
    
        early_stopping = EarlyStopping(monitor=es_mon, mode=es_mode, patience=patience,verbose=True)
        callbacks = [ early_stopping, model_checkpoint,clr,]
        print("Model Fitting:")
        hist= model.fit(t_X, t_y, batch_size=batch_size, epochs=epochs,
              shuffle=True, verbose=True, validation_data=(v_X, v_y),
              callbacks=callbacks)
        epoc_used=len(hist.history[es_mon])  
        ### Getting the Best Model
        print ("Getting the Best Model")  
        model.load_weights('./model.hdf5')   

        print("Predicting Values:")
        #predictions_valid = bestmodel.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        pvy = model.predict([v_X], batch_size=batch_size, verbose=2)
        best_thresh,best_score=best_F1(v_y,pvy[:,0], False )             
        pty = model.predict([test_X], batch_size=1024, verbose=2)
        rec_results (mdln,emb,best_thresh,best_score,mstart)
        import gc;           
        del model
        gc.collect()
        time.sleep(10)
        return  pvy[:,0],pty[:,0],best_score,best_thresh
def evaluate_model5 (label,emb, embedding_matrix,train_X,train_y, t_X, t_y, v_X, v_y, epochs, batch_size, patience=0):
    
    global resultsdf 
    global pred_val_y
    global prev_test_y
    global met
    global clr
    global epoc_used 
    clr = CyclicLR(base_lr=0.001, max_lr=0.01,
               step_size=300., mode='exp_range',
               gamma=0.99994)
    callback = [clr,]
    best_thresh = 0.0
    best_score = 0.0

    mstart = time.time()
    model=None
    model,mdln = create_model(label, embedding_matrix)
    print ("")
    print (mdln+'-'+emb)
    bs=0
    pt=0
    print("Model Fitting:")
    for e in range(epochs):
        model.fit(t_X, t_y, batch_size=batch_size, epochs=1, validation_data=(v_X, v_y),verbose=2)
        print("Predicting Values:")
        pvy = model.predict([v_X], batch_size=1024, verbose=0)
        best_thresh = 0.0
        best_score = 0.0
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            score = metrics.f1_score(v_y, (pvy[:,0]  > thresh).astype(int))
            if score > best_score:
                print("F1COnf Score Improved from %2.4f to %2.4f" % (best_score,score))
                best_thresh = thresh
                best_score = score
        best_thresh = 0.0
        best_score = 0.0
        print("F1 - Score:")
        best_thresh,best_score=best_F1(v_y,pvy[:,0] )
        print("epoch ",e+1,"/",epochs)
        epoc_used=e+1
        if (best_score > bs):          
            print("F1 Score Improved from %2.4f to %2.4f" % (bs,best_score))
            bs = best_score
            bt = best_thresh
            
    pty = model.predict([test_X], batch_size=1024, verbose=0)
    rec_results (mdln,emb,bt,bs,mstart) 
    import gc;           
    del model
    gc.collect()
    time.sleep(10)
    return pvy[:,0],pty[:,0],bs,bt
from keras.callbacks import *
def evaluate_model6 (label,emb, embedding_matrix,train_X,train_y,t_X,t_y,v_X,v_y, epochs, batch_size, patience=0):

        global resultsdf 
        global pred_val_y
        global prev_test_y
        global met 
        global epoc_used
        best_thresh = 0.0
        best_score = 0.0
        mstart = time.time()

        
        model=None
        model,mdln = create_model(label,embedding_matrix)
        print ("")
        print (mdln+'-'+emb)
        
        model_checkpoint = ModelCheckpoint('./model.hdf5', monitor=es_mon, mode=es_mode,
                                      verbose=True, save_best_only=True, 
                                      save_weights_only=False)
        reduce_lr = ReduceLROnPlateau(monitor=es_mon, factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        early_stopping = EarlyStopping(monitor=es_mon,min_delta=0.0001, mode='auto', patience=patience,verbose=True)
        
        #callbacks = [ early_stopping, model_checkpoint]
        callbacks = [ early_stopping,model_checkpoint,reduce_lr]    
    
        print("Model Fitting:")
        hist= model.fit(t_X, t_y, batch_size=batch_size, epochs=epochs,
              shuffle=True, verbose=True, validation_data=(v_X, v_y),
              callbacks=callbacks)
        #print("History\n",hist.history.keys()) 
        #print("F1",hist.history['f1'])
        epoc_used=len(hist.history[es_mon])
        ### Getting the Best Model
        model.load_weights('./model.hdf5')   

        print("Predicting Values:")
        #predictions_valid = bestmodel.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        pvy = model.predict([v_X], batch_size=batch_size, verbose=2)
        best_thresh,best_score=best_F1(v_y,pvy[:,0] , False)             
        pty = model.predict([test_X], batch_size=1024, verbose=2)
        rec_results (mdln,emb,best_thresh,best_score,mstart)
        import gc;           
        del model
        gc.collect()
        time.sleep(10)
        return  pvy[:,0],pty[:,0],best_score,best_thresh
def evaluate_stratifiedkfold (label,mt,emb,train_X,train_y, test_X, n_splits, stop_split, epochs, batch_size,patience):
    
    global resultsdf 
    global allpred_val_y
    global allprev_test_y
    global mn
    global method
    method=mt
    random_seed = 2018
    train_stratified=np.zeros([len(train_X),(stop_split)])
    test_stratified = np.zeros([len(test_X),(stop_split)])
    msstart = time.time()
    embedding_matrix=embedding_matrix_combined
    if  (emb=="Glove"):
        embedding_matrix=embedding_matrix_1
    if  (emb=="Wiki"):
        embedding_matrix=embedding_matrix_2
    if  (emb=="Paragram"):
        embedding_matrix=embedding_matrix_3
    if  (emb=="Google"):
        embedding_matrix=embedding_matrix_4
    if  (emb=="Concatenated"):
        embedding_matrix=embedding_matrix_concatenated 

    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed).split(train_X, train_y))
    mn=mn+1
    bs=np.zeros([stop_split])
    bt=np.zeros([stop_split])
    for idx, (train_idx, valid_idx) in enumerate(splits):
            K.clear_session()
            X_split = train_X[train_idx]
            y_split = train_y[train_idx]
            print ("Split:",idx)
            print ("Total X-train:",len(y_split)," Insincere:",y_split.sum(),y_split.sum()/ len(y_split) )
            X_val = train_X[valid_idx]
            y_val = train_y[valid_idx]
            print ("Total X-val:",len(y_val)," Insincere:",y_val.sum(), y_val.sum()/len(y_val))
            if (mt=="0"):
                pred_train_y, pred_test_y, best_score, best_thresh=evaluate_model(md,emb,embedding_matrix,train_X,train_y, X_split, y_split, X_val, y_val, epochs, batch_size ,patience)
            if (mt=="1"):
                pred_train_y, pred_test_y, best_score, best_thresh=evaluate_model1(md,emb,embedding_matrix,train_X,train_y, X_split, y_split, X_val, y_val,epochs, batch_size ,patience)
            if (mt=="2"):
                pred_train_y, pred_test_y, best_score, best_thresh=evaluate_model2(md,emb,embedding_matrix,train_X,train_y, X_split, y_split, X_val, y_val,epochs, batch_size ,patience) 
            if (mt=="3"):
                pred_train_y, pred_test_y, best_score, best_thresh=evaluate_model3(md,emb,embedding_matrix,train_X,train_y, X_split, y_split, X_val, y_val,epochs, batch_size ,patience) 
            if (mt=="4"):
                pred_train_y, pred_test_y, best_score, best_thresh=evaluate_model4(md,emb,embedding_matrix,train_X,train_y, X_split, y_split, X_val, y_val,epochs, batch_size ,patience) 
            if (mt=="5"):
                pred_train_y, pred_test_y, best_score, best_thresh=evaluate_model5(md,emb,embedding_matrix,train_X,train_y, X_split, y_split, X_val, y_val,epochs, batch_size ,patience) 
            if (mt=="6"):
                pred_train_y, pred_test_y, best_score, best_thresh=evaluate_model6(md,emb,embedding_matrix,train_X,train_y, X_split, y_split, X_val, y_val,epochs, batch_size ,patience) 
            if (best_score > bs[idx]):          
                print("F1 Score Improved from %2.4f to %2.4f" % (bs[idx],best_score))
                bs[idx] = best_score
                bt[idx] = best_thresh
            #train_stratified[:,idx] =  pred_train_y
            test_stratified[:,idx] = pred_test_y
            if ((idx+1)==stop_split):
                break
    
    train_pred =  train_stratified.mean(axis=1)
    test_pred  = test_stratified.mean(axis=1)
    #train_pred =  train_stratified.max(axis=1)
    #test_pred  = test_stratified.max(axis=1)
    #print("F1 - Score - K-Fold:")
    #best_thresh,best_score=best_F1(train_y,train_pred,False )
    if (stop_split > 1):
        rec_results ("Model Mean:"+label+"-skfold- "+str(n_splits)+"Stop: "+str(stop_split),emb,bt.min(),bs.mean(),msstart)
    
    return train_pred,test_pred,bs.max(),bt.min()
###
###  Load Train,Val and Test 
###

train_X, test_X, train_y, test_df, word_index = load_and_prec(True)
###
### Load Embedding Matrix
###
embedding_matrix_1 = load_embedding("Glove",word_index)
#embedding_matrix_2 = load_embedding("Wiki",word_index)
embedding_matrix_3 = load_embedding("Paragram",word_index)
#embedding_matrix_4 = load_embedding("Google",word_index)
## Simple average: http://aclweb.org/anthology/N18-2031

# We have presented an argument for averaging as
# a valid meta-embedding technique, and found experimental
# performance to be close to, or in some cases 
# better than that of concatenation, with the
# additional benefit of reduced dimensionality  


## Unweighted DME in https://arxiv.org/pdf/1804.07983.pdf

# “The downside of concatenating embeddings and 
#  giving that as input to an RNN encoder, however,
#  is that the network then quickly becomes inefficient
#  as we combine more and more embeddings.”
  
# embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2, embedding_matrix_3], axis = 0)
embedding_matrix_combined = np.mean([embedding_matrix_1, embedding_matrix_3], axis = 0)
np.shape(embedding_matrix_combined)

embedding_matrix_concatenated = np.concatenate((embedding_matrix_1, embedding_matrix_3), axis=1)  
#del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3, embedding_matrix_4
#gc.collect()
np.shape(embedding_matrix_concatenated)
###
### Run All Models
###
set_all_parameters()

resultsdf = pd.DataFrame(columns=['Model','F1','Embedding','Pretext','Weigth','Duration','Method',
                                  'Epochs','EpochsUsed','patience','batch_size'
                                  ,'MaxLength','MaxFeatures' ])


### Matrix of Predictions - Validation and Test
### Stores All Predicted Values

allpred_train_y=np.zeros([len(train_X),(len(model_list))])
allpred_test_y=np.zeros([len(test_X),(len(model_list))])
allbest=np.zeros([(len(model_list))])
allthresh=np.zeros([(len(model_list))])

epoc_used=0
mn=-1 

#model_list.append(["18",1,"5","Combine",8,pat,bs,splits,stop,nh,nd,ndh,ndd,nds,loss_bin,opt_rmsprop,met_acc,mon_acc,mode_acc])
for ne,item in  enumerate(model_list): 
    md=item[0]  # Model to Run
    wgt=item[1]    # Model weigth for essembling
    mt=item[2]   # Model Evaluation Method
    emb=item[3] # Embeddings
    epochs=item[4] # Number of Epochs
    patience=item[5] # Patience - Wait patience epochs if score does not improve
    batch_size=item[6] # Batch Size to Fit Model
    nsplits=item[7]   # Number of Splits for Stratified k-fold
    stop_split=item[8]    # Number of Splits to run in Stratified k-fold
    num_lstm=item[9]  # Size of hidden layers in models (lstm and gru)
    num_dense=item[10] # Size of hidden layers in model - dense
    rate_drop_lstm = item[11] # lstm and gru - drop-out rates
    rate_drop_dense = item[12] # dense drop-out rates
    rate_drop_spatial=item[13]  # spatial drop-out rate
    loss=item[14] # Model Loss
    opt=item[15]  #Models Optmizer
    met=item[16]  #Models Metric
    es_mon=item[17] # Early stop monitor 
    es_mode=item[18]  # Early stop mode
    print (mt)
    print ("md-"+md)
    trainp,testp,bs,bt=evaluate_stratifiedkfold (md,mt,emb,train_X,train_y, test_X, nsplits, stop_split, epochs, batch_size,patience)
    allbest[mn]=bs
    allthresh[mn]=bt
    #allpred_train_y[:,mn] = trainp
    allpred_test_y[:,mn] = testp
    print ("FIM-"+md)


#### Aplicando Pesos nos Modelos
#### Aplicando Pesos nos Modelos
twgt=0
wgt_test_y=0
wgt_train_y=0
essemblet="Essemble: "
for ne,item in  enumerate(model_list):
    wg=item[1]
    twgt=twgt+wg
    #wgt_train_y=wgt_train_y+(allpred_train_y[:,ne]*wg)
    wgt_test_y =wgt_test_y +(allpred_test_y[:,ne]*wg)
    essemblet=essemblet+item[0]+" "
#wgt_train_y=wgt_train_y/twgt
wgt_test_y=wgt_test_y/twgt


#best_thresh,best_score=best_F1(train_y,wgt_train_y,True)
#
print (allthresh)
print (allthresh.min())
print (allthresh.mean())

# Write the output
y_te = (np.array(wgt_test_y) >= allthresh.min()).astype(np.int)
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("./submission.csv",index=False)
#display(submit_df)
rec_results ('Time Elapsed','',0,0,bstart)
display (resultsdf)

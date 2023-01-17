import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from sklearn.metrics import f1_score

import graphviz

from sklearn import tree

import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

import random

import warnings

warnings.filterwarnings("ignore")
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



seed_everything(1)
# from hmmlearn import hmm

# model = hmm.GaussianHMM(n_components=3, covariance_type="full")

# model.startprob_ = np.array([0.6, 0.3, 0.1])

# model.transmat_ = np.array([[0.7, 0.1,0.2],[0.3, 0.5, 0.2],[0.3, 0.3, 0.4]])

# model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])

# model.covars_ = np.tile(np.identity(2), (3, 1, 1))

# X, Z = model.sample(100)

# X.shape, Z.shape
%%time

test = pd.read_csv('../input/data-without-drift/test_clean.csv')[['time', 'signal']]

train = pd.read_csv('../input/data-without-drift/train_clean.csv')[['time', 'signal', 'open_channels']]

# train['signal'] = train['signal'].apply(lambda x:np.round(x,4))

# test['signal']  = test['signal'].apply(lambda x:np.round(x,4))



#Normalizing

train_input_mean = train.signal.mean()

train_input_sigma = train.signal.std()

train['signal'] = (train.signal-train_input_mean)/train_input_sigma

test['signal'] = (test.signal-train_input_mean)/train_input_sigma



train.shape,test.shape
# def f(row):

#     if row['open_channels'] == 0:

#         val = np.inf

#     else:

#         val = row['signal']

#     return val

# train['vol'] = train.apply(f, axis=1)
#Adding Previous signal values to make markovian

train['prev'] = 0

train['prev'][0+1:500000] = train['signal'][0:500000-1]

train['prev'][500000+1:1000000] = train['signal'][500000:1000000-1]

train['prev'][1000000+1:1500000] = train['signal'][1000000:1500000-1]

train['prev'][1500000+1:2000000] = train['signal'][1500000:2000000-1]

train['prev'][2000000+1:2500000] = train['signal'][2000000:2500000-1]

train['prev'][2500000+1:3000000] = train['signal'][2500000:3000000-1]

train['prev'][3000000+1:3500000] = train['signal'][3000000:3500000-1]

train['prev'][3500000+1:4000000] = train['signal'][3500000:4000000-1]

train['prev'][4000000+1:4500000] = train['signal'][4000000:4500000-1]

train['prev'][4500000+1:5000000] = train['signal'][4500000:5000000-1]



#Adding Previous signal values to make markovian



test['prev'] = 0

test['prev'][0+1:500000] = test['signal'][0:500000-1]

test['prev'][500000+1:1000000] = test['signal'][500000:1000000-1]

test['prev'][1000000+1:1500000] = test['signal'][1000000:1500000-1]

test['prev'][1500000+1:2000000] = test['signal'][1500000:2000000-1]
# 3d plot of data------without rounding off

# res = 100

# import matplotlib.pyplot as plt

# from mpl_toolkits import mplot3d

# fig = plt.figure(figsize=(30,25))

# ax = plt.axes(projection='3d')

# ax.set_xlabel('prev_signal',size=16)

# ax.set_ylabel('curr_signal',size=16)

# ax.set_zlabel('#channels',size=16);

# zdata = train.open_channels[0::res]

# xdata = train.prev[0::res]

# ydata = train.signal[0::res]

# ax.scatter3D(xdata, ydata, zdata, c=zdata,cmap='plasma');

# from tqdm.notebook import tqdm

# train1 = np.asarray(train[['signal','open_channels']])

# train_dict = {}#sig_val:list_of_open_channels rendered

# for sig,chan in tqdm(train1):

#     temp = []

#     try:

#         temp = train_dict[sig]

#         temp.append(chan)

#     except KeyError:

#         #if signal value occurs for the 1st time

#         temp.append(chan)

#     finally:

#         train_dict[sig] = temp

# print("unique signal values in the training data:",len(train_dict))

# #-------------------------------------------------

# sns.distplot(list(train_dict.keys()))

# plt.show()

# #----------------------------------------------------

# train_info_dict={}#signal:Counter of open_channels

# from collections import Counter

# for key,value in tqdm(train_dict.items()):

#     train_info_dict[key] = Counter(value)
# counter = 0

# overlapping_signals={}#A signal value corresponding to more than 1 open_channels in the data

# for key,value in tqdm(train_info_dict.items()):

#     if len(value)>1:

#         overlapping_signals[key] = value

#         counter+=1

#         #print(key,value)

# print("No of signal values that overlap:",counter)

# #--------------------------------------------------------------------------

# channel_probs = []#probabilities for overlapping signals ONLY.

# for sig,chan_dict in tqdm(overlapping_signals.items()):

#     for i,j in chan_dict.items():

#         #print(sig,i,np.round(j/sum(chan_dict.values()),4))

#         channel_probs.append((sig,i,np.round(j/sum(chan_dict.values()),4)))
# df_channel_probs = pd.DataFrame(channel_probs)

# df_channel_probs.columns=['signal','open_channels','prob']#renaming columns

# df_channel_probs.shape
# df_channel_probs.head()
# result = pd.merge(train, df_channel_probs, how='left', on=['signal','open_channels'])

# result = result.fillna(1)

# result.shape
# train_clean = result

# train_clean.head()
# test['prob'] = 0.5

# test_clean = test

# test_clean.head()
# REMOVING OUTLIERS

#credits :- https://www.kaggle.com/miklgr500/ghost-drift-and-outliers

FIRST_EMISSION = (47.857, 47.863)

SECOND_EMISSION = (364.229, 382.343)



train_clean=train

test_clean=test



train_cwe = train_clean_without_emission = train_clean.loc[(train_clean.time < FIRST_EMISSION[0]) | (train_clean.time > FIRST_EMISSION[1]), :]

train_cwe = train_clean_without_emission = train_cwe.loc[(train_cwe.time < SECOND_EMISSION[0]) | (train_cwe.time > SECOND_EMISSION[1]), :]



SGNAL_SHIFT_CONSTANT = np.exp(1)



#removing the "Ghost drift"



train_cwe.loc[2000000:2500000, 'signal'] += SGNAL_SHIFT_CONSTANT

train_cwe.loc[4500000:, 'signal'] += SGNAL_SHIFT_CONSTANT



test_clean.loc[500000:600000, 'signal'] += SGNAL_SHIFT_CONSTANT

test_clean.loc[700000:800000, 'signal'] += SGNAL_SHIFT_CONSTANT



train = train_cwe

test  = test_clean





train.shape,test.shape
sns.distplot(train['signal'])
sns.distplot(test['signal'])
train.head()
from sklearn.mixture import GaussianMixture

X = np.array(train[['prev','signal']])

gmm = GaussianMixture(n_components=11,random_state=1).fit(X)

# labels = gmm.predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='plasma');

probs = pd.DataFrame(gmm.predict_proba(X)).round(4).mul(100)#------------------------CHANGE HERE

temp = pd.concat([train.reset_index(),probs.reset_index()],axis=1)

train = temp

import gc

gc.collect()

train.shape
X = np.array(test[['prev','signal']])

probs = pd.DataFrame(gmm.predict_proba(X)).round(4).mul(100)#------------------------CHANGE HERE

temp = pd.concat([test.reset_index(),probs.reset_index()],axis=1)

test = temp

import gc

gc.collect()

test.shape
train=train.drop(columns=['index'],axis=1)

test=test.drop(columns=['index'],axis=1)

train.shape,test.shape
# #Prints the overlap between signals w.r.t no of open channels

# from tqdm.notebook import tqdm

# train1 = np.asarray(train[['signal','open_channels']])

# train_dict = {}

# for sig,chan in tqdm(train1):

#     temp = []

#     try:

#         temp = train_dict[chan]

#         temp.append(sig)

#     except KeyError:

#         temp.append(sig)

#     finally:

#         train_dict[chan] = temp

# for no_of_channels in range(11):

#     print('no_of_channels:',no_of_channels,'| percent of overlapped signal',100*np.round(len(np.unique(train_dict[no_of_channels]))/len(train_dict[no_of_channels]),4),'%')
plt.figure(figsize=(30,25)); res = 10

plt.xticks(np.arange(2000000, 5500000, step=500000))

plt.yticks(np.arange(-5, 12.5, step=1))



plt.plot(range(0,train.shape[0],res),train.signal[0::res])

#plt.plot(range(0,train.shape[0],res),train.open_channels[0::res],'magenta') 

plt.plot(range(0,test.shape[0],res),test.signal[0::res],'brown')





for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'g--')

for i in range(10): plt.plot([i*500000,i*500000],[-5,12.5],'black')

    

# plt.plot([0,5000000],[0,0],'r--')

# plt.plot([0,5000000],[1,1],'r--')



for j in range(10): plt.text(j*500000+200000,10,str(j+1),size=20)

plt.xlabel('Row',size=16); plt.ylabel('Signal & open channels',size=16); 

plt.title('Training Data Signal and open channels in 10 batches',size=20)

plt.show()
# plt.plot(range(0,1000000,1),test.signal[1000000:],'brown')

# plt.plot([0,1000000],[0.5,0.5],'r--')

# plt.plot([0,1000000],[1,1],'r--')

# plt.plot([0,1000000],[-0.65,-0.65],'r--')
# plt.plot(range(0,600000,1),train.signal[1000000:1600000],'brown')

# plt.plot([0,250000],[0.5,0.5],'r--')

# plt.plot([0,250000],[1,1],'r--')
# sns.distplot(train.signal.values[1500000:2000000])

# plt.show()



# sns.distplot(test.signal.values[1000000:])

# plt.show()
# train_orig = pd.read_csv('../input/data-without-drift/train_clean.csv')[['time', 'signal', 'open_channels']]

# condition0 = (train_orig.open_channels.values==0) 

# condition1 = (train_orig.open_channels.values==1) 

# condition2 = (train_orig.open_channels.values==2) 

# condition3 = (train_orig.open_channels.values==3) 

# condition4 = (train_orig.open_channels.values==4) 

# condition5 = (train_orig.open_channels.values==5) 

# condition6 = (train_orig.open_channels.values==6) 

# condition7 = (train_orig.open_channels.values==7) 

# condition8 = (train_orig.open_channels.values==8) 

# condition9 = (train_orig.open_channels.values==9) 

# condition10 = (train_orig.open_channels.values==10) 



# plt.figure(figsize=(20,10))



# k=sns.distplot(train_orig[condition0].signal.values,color='magenta',bins=2000)

# a=sns.distplot(train_orig[condition1].signal.values,color='black',bins=2000)

# b=sns.distplot(train_orig[condition2].signal.values,color='red',bins=2000)

# c=sns.distplot(train_orig[condition3].signal.values,color='blue',bins=2000)

# d=sns.distplot(train_orig[condition4].signal.values,color='green',bins=2000)

# e=sns.distplot(train_orig[condition5].signal.values,color='yellow',bins=2000)

# f=sns.distplot(train_orig[condition6].signal.values,color='pink',bins=2000)

# g=sns.distplot(train_orig[condition7].signal.values,color='grey',bins=2000)

# h=sns.distplot(train_orig[condition8].signal.values,color='purple',bins=2000)

# i=sns.distplot(train_orig[condition9].signal.values,color='cyan',bins=2000)

# j=sns.distplot(train_orig[condition10].signal.values,color='brown',bins=2000)

# plt.title("Original dataset")

# plt.show()

# train_orig = []


# # test = pd.read_csv('../input/data-without-drift/test_clean.csv')[['time', 'signal']]

# # train = pd.read_csv('../input/data-without-drift/train_clean.csv')[['time', 'signal', 'open_channels']]

# # #Normalizing

# # train_input_mean = train.signal.mean()

# # train_input_sigma = train.signal.std()

# # train['signal'] = (train.signal-train_input_mean)/train_input_sigma

# # test['signal'] = (test.signal-train_input_mean)/train_input_sigma

# # plt.figure(figsize=(30,10))

# # plt.xticks(np.arange(-2.0, 1.40, step=0.1))

# # test1=sns.distplot(test.signal.values[0:1000000,],color='red',bins=2000)

# # test2=sns.distplot(test.signal.values[1000000:,],color='black',bins=2000)



# condition0 = (train.open_channels.values==0) 

# condition1 = (train.open_channels.values==1) 

# condition2 = (train.open_channels.values==2) 

# condition3 = (train.open_channels.values==3) 

# condition4 = (train.open_channels.values==4) 

# condition5 = (train.open_channels.values==5) 

# condition6 = (train.open_channels.values==6) 

# condition7 = (train.open_channels.values==7) 

# condition8 = (train.open_channels.values==8) 

# condition9 = (train.open_channels.values==9) 

# condition10 = (train.open_channels.values==10) 





# k=sns.distplot(train[condition0].signal.values,color='magenta',bins=2000)

# a=sns.distplot(train[condition1].signal.values,color='black',bins=2000)

# b=sns.distplot(train[condition2].signal.values,color='red',bins=2000)

# c=sns.distplot(train[condition3].signal.values,color='blue',bins=2000)

# d=sns.distplot(train[condition4].signal.values,color='green',bins=2000)

# e=sns.distplot(train[condition5].signal.values,color='yellow',bins=2000)

# f=sns.distplot(train[condition6].signal.values,color='pink',bins=2000)

# g=sns.distplot(train[condition7].signal.values,color='grey',bins=2000)

# h=sns.distplot(train[condition8].signal.values,color='purple',bins=2000)

# i=sns.distplot(train[condition9].signal.values,color='cyan',bins=2000)

# j=sns.distplot(train[condition10].signal.values,color='brown',bins=2000)

# plt.title("After cleaning dataset")

# plt.show()
# trainvals = np.array(train[['prev','signal']].apply(lambda x:np.round(x,4)))

# testvals  = np.array(test[['prev','signal']].apply(lambda x:np.round(x,4)))

# train_set = set(map(lambda x: frozenset(tuple(x)), trainvals))

# test_set = set(map(lambda x: frozenset(tuple(x)), testvals))

# len(train_set),len(test_set),len(train_set.intersection(test_set))
#2:(132707, 91066, 86313)

#4:(4623885, 1881919, 219496)

plt.figure(figsize=(25,15))

plt.yticks(np.arange(-3, 8, step=1))

plt.xticks(np.arange(0, 2500000, step=100000));res = 1; 

let = ['1f', '3', '5', '1f','1f','10','5','10','1f','3']

plt.plot(range(0,test.shape[0],res),test.signal[0::res])

plt.plot([0,2000000],[0,0],'r--')

plt.plot([0,2000000],[1,1],'r--')



for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

for j in range(21): plt.plot([j*100000,j*100000],[-5,12.5],'y:')

for k in range(4): plt.text(k*500000+200000,10,str(k+1),size=20)

for k in range(10): plt.text(k*100000+40000,7,let[k],size=16)

plt.xlabel('Row',size=16); plt.ylabel('Signal',size=16); 

plt.title('Test Data Signal - 4 batches - 10 subsamples',size=20)

plt.show()
'channel-1:',min(train[train['open_channels']==2]['signal']),max(train[train['open_channels']==1]['signal'])
# train2 = train.copy()
# X_train = np.asarray(train2[['signal','prev']][0:1000000]).reshape((-1,2))

# y_train = np.asarray(train2.open_channels.values[0:1000000]).reshape((-1,1))



# X_train, y_train = rus.fit_resample(X_train, y_train)

# print('X_train.shape,y_train.shape:',X_train.shape,y_train.shape)



# plt.hist(y_train)

# clf1s = tree.DecisionTreeClassifier(max_depth=1,criterion='entropy')

# clf1s = clf1s.fit(X_train,y_train)

# print('Training model low-probability channel')

# preds = clf1s.predict(X_train)

# print('f1 validation score =',f1_score(y_train,preds,average='macro'))

# tree_graph = tree.export_graphviz(clf1s, out_file=None, max_depth = 10,

#     impurity = False, feature_names = ['signal','prev'], class_names = ['0', '1'],

#     rounded = True, filled= True )

# graphviz.Source(tree_graph)  
# #UNDERSAMPLING

# s=pd.concat([train2[1000000:2000000],train2[2500000:4500000]])

# from imblearn.under_sampling import RandomUnderSampler

# rus = RandomUnderSampler(random_state=0)

# X_resampled, y_resampled = rus.fit_resample(np.asarray(s[['signal','prev']]).reshape(-1,2), np.asarray(s['open_channels']))

# sns.countplot(y_resampleda

# plt.show()

# X_resampled.shape


# %%time

# import pandas

# import xgboost

# from sklearn import model_selection

# from sklearn.metrics import accuracy_score

# from sklearn.preprocessing import LabelEncoder



# X = X_sm#np.asarray(train2[['signal','prev']][1000000:]).reshape((-1,2))

# Y = y_sm#np.asarray(train2.open_channels.values[1000000:])

# # encode string class values as integers

# label_encoder = LabelEncoder()

# label_encoder = label_encoder.fit(Y)

# label_encoded_y = label_encoder.transform(Y)

# seed = 1

# test_size = 0.30

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, label_encoded_y, test_size=0.25, random_state=seed)

# # X_train, X_test, y_train, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

# # X_train, X_test, y_train, y_test = model_selection.train_test_split(X_test, y_test, test_size=test_size, random_state=seed)

# # fit model no training data

# print('X_train.shape,y_train.shape:',X_train.shape,y_train.shape)



# model2 = xgboost.XGBClassifier(objective='multi:softmax',num_classes=10)

# model2.fit(X_train, y_train)

# print(model2)

# # make predictions for test data

# y_pred = model2.predict(X_test)

# predictions = [round(value) for value in y_pred]

# # evaluate predictions

# accuracy = accuracy_score(y_test, predictions)

# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# from numpy import argmax

# from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import LabelEncoder

# from imblearn.over_sampling import SMOTE, ADASYN



# X, y = SMOTE().fit_resample(np.asarray(train[['prev','signal',0,1,2,3,4,5,6,7,8,9,10]]), np.asarray(train['open_channels']))

# sns.countplot(y)

# plt.show()



# print('X.shape,y.shape:',X.shape,y.shape)
# seed = 1

# #,0,1,2,3,4,5,6,7,8,9,10

# #y = LabelEncoder().fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,random_state=seed,shuffle=True)

# X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.50,random_state=seed,shuffle=True)

# X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.3,random_state=seed,shuffle=True)



# print("After splitting:",X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# sns.countplot(y_train)

# plt.show()

# x = X.shape[1]#no of features in data matrix

# X_train = X_train.reshape((-1,1,1,x))

# X_test = X_test.reshape((-1,1,1,x))

# y_train=y_train.reshape(-1,1)

# y_test=y_test.reshape(-1,1)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import tensorflow as tf

import keras.backend as K

from sklearn.metrics import f1_score

import time 

#,0,1,2,3,4,5,6,7,8,9,10

class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, training_data, validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]

        self.start=0.0

        self.end=0.0

        

    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        self.start = time.time()



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred_val = self.model.predict(self.x_val)

        pred = np.array([*map(np.argmax,y_pred_val)]).reshape(-1)

        target = self.y_val.reshape(-1)

        score = f1_score(target, pred, average="macro")

        print(f' F1Macro: {score:.5f}')

        self.end = time.time()

        print((self.end-self.start))
import tensorflow as tf

from tensorflow.keras.layers import Layer

from tensorflow.keras import initializers

from tensorflow.keras import regularizers

from tensorflow.keras import constraints



class Attention1(Layer):

    """

    Multi-headed attention layer.

    """

    

    def __init__(self, hidden_size, 

                 num_heads = 8, 

                 attention_dropout=.1,

                 trainable=True,

                 name='Attention1'):

        

        if hidden_size % num_heads != 0:

            raise ValueError("Hidden size must be evenly divisible by the number of heads.")

            

        self.hidden_size = hidden_size

        self.num_heads = num_heads

        self.trainable = trainable

        self.attention_dropout = attention_dropout

        self.dense = tf.keras.layers.Dense(self.hidden_size, use_bias=False)

        super(Attention1, self).__init__(name=name)



    def split_heads(self, x):

        """

        Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct

        values during the matrix multiplication.

        Args:

          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:

          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]

        """

        with tf.name_scope("split_heads"):

            batch_size = tf.shape(x)[0]

            length = tf.shape(x)[1]



            # Calculate depth of last dimension after it has been split.

            depth = (self.hidden_size // self.num_heads)



            # Split the last dimension

            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])



            # Transpose the result

            return tf.transpose(x, [0, 2, 1, 3])

    

    def combine_heads(self, x):

        """

        Combine tensor that has been split.

        Args:

          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:

          A tensor with shape [batch_size, length, hidden_size]

        """

        with tf.name_scope("combine_heads"):

            batch_size = tf.shape(x)[0]

            length = tf.shape(x)[2]

            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]

            return tf.reshape(x, [batch_size, length, self.hidden_size])        

        

    def call(self, inputs):

        """

        Apply attention mechanism to inputs.

        Args:

          inputs: a tensor with shape [batch_size, length_x, hidden_size]

        Returns:

          Attention layer output with shape [batch_size, length_x, hidden_size]

        """

    

        q = self.dense(inputs)

        k = self.dense(inputs)

        v = self.dense(inputs)



        q = self.split_heads(q)

        k = self.split_heads(k)

        v = self.split_heads(v)

        

        # Scale q to prevent the dot product between q and k from growing too large.

        depth = (self.hidden_size // self.num_heads)

        q *= depth ** -0.5

        

        logits = tf.matmul(q, k, transpose_b=True)

        # logits += self.bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        

        if self.trainable:

            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)

        

        attention_output = tf.matmul(weights, v)

        attention_output = self.combine_heads(attention_output)

        attention_output = self.dense(attention_output)

        return attention_output

        

    def compute_output_shape(self, input_shape):

        return tf.TensorShape(input_shape)
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Conv1D

from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.layers import LSTM, GRU

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras.layers import TimeDistributed

# instantiating the model in the strategy scope creates the model on the TPU

x= 13

def create_model():

    model = Sequential([

    TimeDistributed(Conv1D(filters=128, kernel_size=1,activation='relu'), input_shape=(None,1, 13)),

    TimeDistributed(MaxPooling1D(pool_size=1)),

    TimeDistributed(Flatten()),

    Bidirectional(LSTM(128, return_sequences=True)),

    #Attention1(512),

    BatchNormalization(),

    Dropout(0.20),

    Bidirectional(LSTM(128, return_sequences=True)),

    BatchNormalization(),

    Dropout(0.30),

    Bidirectional(LSTM(128, return_sequences=False)),

    BatchNormalization(),

    Dropout(0.20),

    Dense(16,activation="relu"),

    Dropout(0.20),

    Dense(11, activation='softmax')

    ])

    return model
create_model().summary()
# from IPython.display import SVG

# SVG(tf.keras.utils.model_to_dot(create_model(), dpi=70).create(prog='dot', format='svg'))
# from tensorflow.keras.callbacks import ModelCheckpoint

# sv = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='auto', save_freq='epoch')
test.head()
test2=test.copy()

test2=np.asarray(test2[['prev','signal',0,1,2,3,4,5,6,7,8,9,10]])

X_test = test2.reshape((-1,1,1,x))

X_test.shape


# print("Tensorflow version " + tf.__version__)

# AUTO = tf.data.experimental.AUTOTUNE

# from kaggle_datasets import KaggleDatasets

# gcs_path = KaggleDatasets().get_gcs_path('data-without-drift') 

#************************************************************************

# import tensorflow as tf

# try:

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

#     print('Running on TPU ', tpu.master())

# except ValueError:

#     tpu = None



# if tpu:

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else:

#     strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



# print("REPLICAS: ", strategy.num_replicas_in_sync)

#************************************************************************

#with strategy.scope():

#     model = create_model()

#     optimizer=Adam(lr=0.001)

#     model.compile(optimizer=optimizer,loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# model.summary()

# model.fit(X_train, y_train, epochs=20,batch_size=128,shuffle=False,validation_data=(X_test, y_test),

#               callbacks=[CustomCallback((X_train, y_train),(X_test, y_test)),lr_schedule])
# sys.path.insert(, "../input/multikfold/")

# from ml_stratifiers import MultilabelStratifiedKFold

#kf = MultilabelStratifiedKFold(n_splits = 5, random_state = 1)

import pickle

with open('../input/iondata/X_train.pickle', 'rb') as handle1:

    X_train = pickle.load(handle1)

with open('../input/iondata/y_train.pickle', 'rb') as handle2:

    y_train = pickle.load(handle2)

# %%time



# from hyperopt import tpe

# from hyperopt import STATUS_OK

# from hyperopt import Trials

# from hyperopt import hp

# from hyperopt import fmin

# import warnings

# warnings.filterwarnings("ignore")  

# MAX_EVALS = 10

# def objective(params):

#     model = create_model(**params)

#     optimizer=Adam(lr=0.001)

#     model.compile(optimizer=optimizer,loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#     model.fit(X_train_opt, y_train_opt, epochs=20, batch_size=256, verbose=False)

#     loss,accuracy = model.evaluate(X_eval_opt,y_eval_opt, steps=2, verbose=2)

#     return {'loss': loss, 'params': params, 'status': STATUS_OK}

# space = {

#     'a': hp.choice('a', range(128,800)),

#     'b': hp.choice('b', range(128,800)),

#     'c': hp.choice('c', range(64,800)),

#     #'dropout2': hp.uniform('dropout2', 0.2,0.4),

# }

# tpe_algorithm = tpe.suggest

# bayes_trials = Trials()

# best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
# train

all_predictions = []



import math

from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import KFold



kf = KFold(n_splits=5, random_state=1, shuffle=True)

lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * math.pow(0.001, math.floor((1+epoch)/3.0)))





for ind, (tr, val) in enumerate(kf.split(X_train)):

    X_tr = X_train[tr]

    y_tr = y_train[tr]

    X_vl = X_train[val]

    y_vl = y_train[val]

    model = create_model()

    optimizer=Adam(lr=0.001)

    model.compile(optimizer=optimizer,loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])



    print( X_tr.shape,y_tr.shape,X_vl.shape,y_vl.shape)

    model.fit(X_tr, y_tr, epochs=30,batch_size=128,shuffle=True,validation_data=(X_vl, y_vl),

              callbacks=[CustomCallback((X_tr, y_tr),(X_vl, y_vl)),lr_schedule])

    print("Done training! Now predicting")

    all_predictions.append(model.predict(X_test))
# %%time

# from bayes_opt import BayesianOptimization

# import numpy as np

# def fit_with(a,b,c):

#     model = create_model(a,b,c)

#     optimizer=Adam(lr=0.001)

#     model.compile(optimizer=optimizer,loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#     model.fit(X_train_opt, y_train_opt, epochs=33, batch_size=256, verbose=False)

#     loss,accuracy = model.evaluate(X_eval_opt,y_eval_opt, steps=12, verbose=0)

#     print('loss:', np.round(loss,5))

#     return np.round(accuracy,5)#rounding so that the optimizer converges for 5 decimal places 

# pbounds = {

#         'a':(256,512),

#         'b': (256,512),

#         'c':(256,512)

#           }

# optimizer = BayesianOptimization(f=fit_with,pbounds=pbounds,verbose=2,random_state=1)

# optimizer.maximize()

# for i, res in enumerate(optimizer.res):

#     print("Iteration {}: \n\t{}".format(i, res))
all_predictions[0].shape
## Predictions for k-fold TRAINING

sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')

avged = sum(all_predictions)/5.0

predictions =[*map(np.argmax,avged)]

sub.iloc[:,1] = np.asarray(predictions)

sub.to_csv('submission.csv',index=False,float_format='%.4f')

sub['open_channels'].hist()
# ##FOR SIMPLE TRAINING

# ans=model.predict(X_test)

# sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')

# predictions =[*map(np.argmax,ans)]

# sub.iloc[:,1] = np.asarray(predictions)

# sub.to_csv('submission.csv',index=False,float_format='%.4f')

# sub['open_channels'].hist()
# loss, acc, f1 = model.evaluate(X_test, y_test, verbose=1)

# print("loss={}, acc={}, f1={}".format(loss, acc,f1))


# import os

# import numpy

# import time

# import random

# import math



# import numpy as np

# import pandas as pd

# import matplotlib.pyplot as plt

# import tensorflow as tf

# from imblearn.over_sampling import SMOTE

# from sklearn.metrics import f1_score



# from sklearn.preprocessing import MinMaxScaler

# from sklearn.metrics import roc_curve, auc, confusion_matrix

# from sklearn.utils import shuffle



# from tensorflow.keras.models import Sequential, load_model

# from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation, LSTM, BatchNormalization, TimeDistributed, Conv1D, MaxPooling1D

# from tensorflow.keras.metrics import Precision, Recall

# from tensorflow.keras.callbacks import LearningRateScheduler

# from tensorflow.keras import optimizers

# from tensorflow.keras import backend as K

# from tensorflow.keras.utils import to_categorical

# from tensorflow.keras.metrics import Precision, Recall

# from tensorflow_addons.metrics import F1Score





# def mcor(y_true, y_pred):

#     # Matthews correlation

#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))

#     y_pred_neg = 1 - y_pred_pos



#     y_pos = K.round(K.clip(y_true, 0, 1))

#     y_neg = 1 - y_pos



#     tp = K.sum(y_pos * y_pred_pos)

#     tn = K.sum(y_neg * y_pred_neg)



#     fp = K.sum(y_neg * y_pred_pos)

#     fn = K.sum(y_pos * y_pred_neg)



#     numerator = (tp * tn - fp * fn)

#     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



#     return numerator / (denominator + K.epsilon())





# def make_roc(true, predicted):



#     # roc curve plotting for multiple



#     n_classesi = predicted.shape[1]



#     fpr = {}

#     tpr = {}

#     roc_auc = {}



#     for i in range(n_classesi):

#         fpr[i], tpr[i], _ = roc_curve(true[:, i], predicted[:, i])

#         roc_auc[i] = auc(fpr[i], tpr[i])



#     plt.figure()

#     lw = 2

#     plt.plot(fpr[2], tpr[2], color='darkorange',

#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

#     plt.xlim([0.0, 1.0])

#     plt.ylim([0.0, 1.0])

#     plt.xlabel('False Positive Rate')

#     plt.ylabel('True Positive Rate')

#     plt.title('Receiver operating characteristic example')

#     plt.legend(loc="lower right")

#     plt.show()



#     plt.figure(2)

#     plt.xlim(0, 1)

#     plt.ylim(0, 1)

#     colors = ['aqua', 'darkorange', 'cornflowerblue',

#                     'red', 'black', 'yellow']

#     for i in range(n_classesi):

#         plt.plot(fpr[i], tpr[i], color=color[i], lw=lw,

#                  label='ROC curve of class {0} (area = {1:0.2f})'

#                  ''.format(i, roc_auc[i]))



#     plt.xlabel('False Positive Rate (1 - Specificity)')

#     plt.ylabel('True Positive Rate (Sensitivity)')

#     plt.title('Zooom in View: Some extension of ROC to multi-class')

#     plt.legend(loc="lower right")

#     plt.show()





# def step_decay(epoch):

#     # Learning rate scheduler object

#     initial_lrate = 0.001

#     drop = 0.001

#     epochs_drop = 3.0

#     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

#     return lrate





# '''

# ############# SET UP RUN HERE ####################

# '''



# batch_size = 256







# df = pd.read_csv('outfinaltest161.csv', header=None)

# dataset = df.values.astype('float64')

# timep = dataset[:, 0]

# maxer = np.amax(dataset[:, 2])

# maxeri = maxer.astype('int')

# maxchannels = maxeri

# idataset = dataset[:, 2].astype(int)

# scaler = MinMaxScaler(feature_range=(0, 1))

# dataset = scaler.fit_transform(dataset)



# # train and test set split and reshape:

# train_size = int(len(dataset) * 0.80)

# modder = math.floor(train_size/batch_size)

# train_size = int(modder*batch_size)

# test_size = int(len(dataset) - train_size)

# modder = math.floor(test_size/batch_size)

# test_size = int(modder*batch_size)



# print(f'training set = {train_size}')

# print(f'test set = {test_size}')

# print(f'total length = {test_size + train_size}')





# x_train = dataset[:, 1]

# y_train = idataset[:]

# x_train = x_train.reshape((len(x_train), 1))

# y_train = y_train.reshape((len(y_train), 1))





# sm = SMOTE(sampling_strategy='auto', random_state=42)

# X_res, Y_res = sm.fit_sample(x_train, y_train)



# yy_res = Y_res.reshape((len(Y_res), 1))

# yy_res = to_categorical(yy_res, num_classes=maxchannels+1)

# xx_res, yy_res = shuffle(X_res, yy_res)





# trainy_size = int(len(xx_res) * 0.80)

# modder = math.floor(trainy_size/batch_size)

# trainy_size = int(modder*batch_size)

# testy_size = int(len(xx_res) - trainy_size)

# modder = math.floor(testy_size/batch_size)

# testy_size = int(modder*batch_size)



# print('training set= ', trainy_size)

# print('test set =', testy_size)

# print('total length', testy_size+trainy_size)





# in_train, in_test = xx_res[0:trainy_size,

#                            0], xx_res[trainy_size:trainy_size+testy_size, 0]

# target_train, target_test = yy_res[0:trainy_size,

#                                    :], yy_res[trainy_size:trainy_size+testy_size, :]

# in_train = in_train.reshape(len(in_train), 1, 1, 1)

# in_test = in_test.reshape(len(in_test), 1, 1, 1)





# # validation set!!

# df_val = pd.read_csv('outfinaltest78.csv', header=None)

# data_val = df_val.values.astype('float64')



# idataset2 = data_val[:, 2].astype(int)



# val_set = data_val[:, 1]

# scaler = MinMaxScaler(feature_range=(0, 1))

# val_set = scaler.fit_transform(val_set.reshape(-1,1))

# val_set = val_set.reshape(len(val_set), 1, 1, 1)

# val_target = data_val[:, 2]

# val_target = to_categorical(val_target, num_classes=maxchannels+1)





# # model starts..



# newmodel = Sequential()

# timestep = 1

# input_dim = 1

# newmodel.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,

#                                     activation='relu'), input_shape=(None, timestep, input_dim)))

# newmodel.add(TimeDistributed(MaxPooling1D(pool_size=1)))

# newmodel.add(TimeDistributed(Flatten()))



# newmodel.add(LSTM(256, activation='relu', return_sequences=True))

# newmodel.add(BatchNormalization())

# newmodel.add(Dropout(0.2))



# newmodel.add(LSTM(256, activation='relu', return_sequences=True))

# newmodel.add(BatchNormalization())

# newmodel.add(Dropout(0.2))



# newmodel.add(LSTM(256, activation='relu'))

# newmodel.add(BatchNormalization())

# newmodel.add(Dropout(0.2))



# newmodel.add(Dense(maxchannels+1))

# newmodel.add(Activation('softmax'))





# newmodel.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False), metrics=[

#                  'accuracy', Precision(), Recall(), F1Score(num_classes=maxchannels+1, average='micro')])





# lrate = LearningRateScheduler(step_decay)





# epochers = 2

# history = newmodel.fit(x=in_train, y=target_train, initial_epoch=0, epochs=epochers, batch_size=batch_size, callbacks=[

#                        lrate], verbose=1, shuffle=False, validation_data=(in_test, target_test))





# # prediction for test set

# predict = newmodel.predict(in_test, batch_size=batch_size)



# # prediction for val set

# predict_val = newmodel.predict(val_set, batch_size=batch_size)





# class_predict = np.argmax(predict, axis=-1)

# class_predict_val = np.argmax(predict_val, axis=-1)

# class_target = np.argmax(target_test, axis=-1)

# class_target_val = np.argmax(val_target, axis=-1)





# cm_test = confusion_matrix(class_target, class_predict)

# cm_val = confusion_matrix(idataset2, class_predict_val)



# rnd = 1

# # summarize history for accuracy

# plt.plot(history.history['accuracy'])

# plt.plot(history.history['val_accuracy'])

# plt.title('model accuracy')

# plt.ylabel('accuracy')

# plt.xlabel('epoch')

# plt.legend(['train', 'test'], loc='lower right')

# plt.savefig(str(rnd)+'acc.png')

# plt.show()



# # summarize history for loss

# plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

# plt.title('model loss')

# plt.ylabel('loss')

# plt.xlabel('epoch')

# plt.legend(['train', 'test'], loc='upper right')

# plt.savefig(str(rnd)+'loss.png')

# plt.show()





# plotlen = test_size

# lenny = 2000



# plt.figure(figsize=(30, 6))

# plt.subplot(2, 1, 1)

# # temp=scaler.inverse_transform(dataset)

# plt.plot(xx_res[trainy_size:trainy_size+lenny, 0],

#          color='blue', label="some raw data")

# plt.title("The raw test")



# plt.subplot(2, 1, 2)

# plt.plot(class_target[:lenny], color='black', label="the actual idealisation", drawstyle='steps-mid')



# line, = plt.plot(class_predict[:lenny], color='red',

#                  label="predicted idealisation", drawstyle='steps-mid')

# plt.setp(line, linestyle='--')

# plt.xlabel('timepoint')

# plt.ylabel('current')

# # plt.savefig(str(rnd)+'data.png')

# plt.legend()

# plt.show()





# # newmodel.save('nmn_oversampled_deepchanel6_5.h5')



# make_roc(val_target, predicted_val)
# train2 = train.copy()
# test2.tail()
# test2=test.copy()

# test2['prev']=0

# test2['prev'][0+1:500000] = test2['signal'][0:500000-1]

# test2['prev'][500000+1:1000000] = test2['signal'][500000:1000000-1]

# test2['prev'][1000000+1:1500000] = test2['signal'][1000000:1500000-1]

# test2['prev'][1500000+1:2000000] = test2['signal'][1500000:2000000-1]

# test2['prob']=0.5

# #For neural-network

# sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')

# test2=np.asarray(test2[['signal']])

# test2 = test2.reshape((-1,1,1))

# test2.shape

# predictions =[*map(np.argmax,newmodel.predict(test2))]

# sub.iloc[:,1] = np.asarray(predictions)
# sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')



# a = 0 # SUBSAMPLE A, Model 1f

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



# a = 1 # SUBSAMPLE B, Model 3

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))





# a = 2 # SUBSAMPLE C, Model 5

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



# a = 3 # SUBSAMPLE D, Model 1f

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



# a = 4 # SUBSAMPLE E, Model 1f

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



# a = 5 # SUBSAMPLE F, Model 10

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))





# a = 6 # SUBSAMPLE G, Model 5

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



# a = 7 # SUBSAMPLE H, Model 10

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



# a = 8 # SUBSAMPLE I, Model 1s

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



# a = 9 # SUBSAMPLE J, Model 3

# sub.iloc[100000*a:100000*(a+1),1] = newmodel.predict(np.asarray(test2[['signal']][100000*a:100000*(a+1)]))



#  # BATCHES 3 AND 4 seem to be generated from Model 1s

# #sub.iloc[1000000:2000000,1] = clf1s.predict(test2.signal.values[1000000:2000000].reshape((-1,1)))

# sub.iloc[1000000:2000000,1] = newmodel.predict(np.asarray(test2[['signal']][1000000:2000000]))
# sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')



# a = 0 # SUBSAMPLE A, Model 1s

# #sub.iloc[100000*a:100000*(a+1),1] = clf1s.predict(test2[['signal','prev']][100000*a:100000*(a+1)].reshape((-1,1)))

# sub.iloc[100000*a:100000*(a+1),1] =clf1s.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))



# a = 1 # SUBSAMPLE B, Model 3

# y_pred = model.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))

# predictions = [*map(np.argmax,y_pred)]

# sub.iloc[100000*a:100000*(a+1),1] = np.asarray(predictions)



# a = 2 # SUBSAMPLE C, Model 5

# y_pred = model.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))

# predictions =[*map(np.argmax,y_pred)]

# sub.iloc[100000*a:100000*(a+1),1] = np.asarray(predictions)



# a = 3 # SUBSAMPLE D, Model 1s

# #sub.iloc[100000*a:100000*(a+1),1] = clf1s.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

# sub.iloc[100000*a:100000*(a+1),1] =clf1s.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))



# a = 4 # SUBSAMPLE E, Model 1f

# y_pred = model.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))

# predictions = [*map(np.argmax,y_pred)]

# sub.iloc[100000*a:100000*(a+1),1] = np.asarray(predictions)



# a = 5 # SUBSAMPLE F, Model 10

# y_pred = model.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))

# predictions = [*map(np.argmax,y_pred)]

# sub.iloc[100000*a:100000*(a+1),1] = np.asarray(predictions)



# a = 6 # SUBSAMPLE G, Model 5

# y_pred = model.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))

# predictions = [*map(np.argmax,y_pred)]

# sub.iloc[100000*a:100000*(a+1),1] = np.asarray(predictions)



# a = 7 # SUBSAMPLE H, Model 10

# y_pred = model.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))

# predictions = [*map(np.argmax,y_pred)]

# sub.iloc[100000*a:100000*(a+1),1] = np.asarray(predictions)



# a = 8 # SUBSAMPLE I, Model 1s

# #sub.iloc[100000*a:100000*(a+1),1] = clf1s.predict(test2.signal.values[100000*a:100000*(a+1)].reshape((-1,1)))

# sub.iloc[100000*a:100000*(a+1),1] =clf1s.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))



# a = 9 # SUBSAMPLE J, Model 3

# y_pred = model.predict(np.asarray(test2[['signal','prev']][100000*a:100000*(a+1)]))

# predictions = [*map(np.argmax,y_pred)]

# sub.iloc[100000*a:100000*(a+1),1] = np.asarray(predictions)



#  # BATCHES 3 AND 4 seem to be generated from Model 1s

# #sub.iloc[1000000:2000000,1] = clf1s.predict(test2.signal.values[1000000:2000000].reshape((-1,1)))

# sub.iloc[1000000:2000000,1] =clf1s.predict(np.asarray(test2[['signal','prev']][1000000:2000000]))
# plt.figure(figsize=(20,5))

# plt.ylim(bottom=-1,top=12)

# plt.yticks(np.arange(-1, 12, step=1))

# plt.ylabel('Channels Open',size=16)

# res = 1000

# plt.plot(range(0,test.shape[0],res),sub.open_channels[0::res])

# for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')

# for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')

# for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)

# for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)

# plt.title('Test Data Predictions',size=16)

# plt.show()
# sub.to_csv('submission.csv',index=False,float_format='%.4f')
# sub['open_channels'].hist()

# sub['open_channels'].hist()
# import pickle

# # save model to file

# with open('y_test.pickle', 'wb') as handle:

#     pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #X_train.shape, X_test.shape, y_train.shape, y_test
# # # some time later...

# import pickle

# with open('X_test.pickle', 'rb') as handle:

#     b = pickle.load(handle)

# # # load model from file

# # loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
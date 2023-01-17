# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



from nltk.tokenize import sent_tokenize, word_tokenize 

import warnings 

  

warnings.filterwarnings(action = 'ignore') 

  

import gensim 

from gensim.models import Word2Vec 

print(os.listdir("../input/data 4"))

  



# Any results you write to the current directory are saved as output.
import pandas as pd







# HBP

data = pd.read_csv('../input/data 4/hbp.txt', sep=">",header=None)

sequences=data[0].dropna()

labels=data[1].dropna()

sequences.reset_index(drop=True, inplace=True)

labels.reset_index(drop=True, inplace=True)

list_of_series=[sequences.rename("sequences"),labels.rename("Name")]

df_hbp = pd.concat(list_of_series, axis=1)

df_hbp['label']='hbp'

df_hbp.head()









# not HBP

data = pd.read_csv('../input/data 4/non-hbp.txt', sep=">",header=None)

sequences=data[0].dropna()

labels=data[1].dropna()

sequences.reset_index(drop=True, inplace=True)

labels.reset_index(drop=True, inplace=True)

list_of_series=[sequences.rename("sequences"),labels.rename("Name")]

df_N_hbp = pd.concat(list_of_series, axis=1)

df_N_hbp['label']='non-hbp'

df_N_hbp.head()
frames = [df_hbp,df_N_hbp]

df=pd.concat(frames)

df.head()
from sklearn.preprocessing import LabelBinarizer

import keras

# Transform labels to one-hot

lb = LabelBinarizer()

Y = lb.fit_transform(df.label)

Cat_y=keras.utils.to_categorical(Y,num_classes=2)
arr=[]

for i in df.sequences:

    arr.append(len(i))

    

arr=np.asarray(arr)

print("Minimum length of string is = ",(arr.min()))

minlength=arr.min()

from keras.preprocessing import text, sequence

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split



# maximum length of sequence, everything afterwards is discarded!

max_length = minlength



#create and fit tokenizer

tokenizer = Tokenizer(char_level=True)

tokenizer.fit_on_texts(df.sequences)

#represent input data as word rank number sequences

X = tokenizer.texts_to_sequences(df.sequences)

X = sequence.pad_sequences(X, maxlen=max_length)
from keras.models import Sequential

from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,Dropout,Conv2D

from keras.layers import LSTM

from keras.layers.embeddings import Embedding



embedding_dim = 8



# create the model

model = Sequential()

model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))

model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'))

model.add(Dropout(0.5))



model.add(Conv1D(filters=8, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=1))



model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, Cat_y, test_size=.3)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=1)
from keras.utils import plot_model

plot_model(model, to_file='model.png')
from sklearn.metrics import classification_report

import numpy as np





print(classification_report(Y_test, y_pred))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split

import keras

from keras.callbacks import LambdaCallback

from keras.layers import Conv1D, Flatten

from keras.layers import Dense ,Dropout,BatchNormalization

from keras.models import Sequential 

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical 

from keras import regularizers

from sklearn import preprocessing

from sklearn.ensemble import  VotingClassifier

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics

from sklearn import ensemble

from sklearn import gaussian_process

from sklearn import linear_model

from sklearn import naive_bayes

from sklearn import neighbors

from sklearn import svm

from sklearn import tree

from sklearn import discriminant_analysis

from sklearn import model_selection

from xgboost.sklearn import XGBClassifier 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#Machine Learning Algorithm (MLA) Selection and Initialization



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]









#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy' ]

MLA_compare = pd.DataFrame(columns = MLA_columns)







#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

   # cv_results = model_selection.cross_validate(alg, X_train, y_train)

    alg.fit(X_train, y_train)

    y_pred=alg.predict(X_test)

    score=metrics.accuracy_score(y_test, y_pred)

    

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] =score



    

    

    row_index+=1



    



MLA_compare



MLA_compare.to_csv("classifier.csv")

#MLA_predict
X= df.sequences.astype(str).str[0:96]

X=X.values

print("Every String has length equal to =",len(X[1]))

# Every Sequence Length is now 96
def onehot(ltr):

     return [1 if i==ord(ltr) else 0 for i in range(97,123)]



def onehotvec(s):

     return [onehot(c) for c in list(s.lower())]



sequence_encode=[]



for i in range(0,len(X)):

    

    X[i]=X[i].lower()

    a=onehotvec(X[i])

    a=np.asarray(a)

    sequence_encode.append(a)

    

sequence_encode=np.asarray(sequence_encode)  

print("Shape of One Hot Encoded Sequence",sequence_encode.shape)
X=sequence_encode.reshape(-1,96,26,1)

X.shape
from keras.layers import Conv2D,LeakyReLU,MaxPooling2D

from keras.layers.core import Activation



# create the model

model = Sequential()

model.add(Conv2D(16,kernel_size = (2,2),input_shape=(96,26,1)))

model.add(Activation("relu"))

model.add(Conv2D(32,kernel_size = (2,2),input_shape=(96,26,1)))

model.add(Activation("relu"))



model.add(MaxPooling2D(pool_size=(2, 2),strides=2, padding='same', data_format=None))

model.add(Dropout(0.2))



model.add(Conv2D(64,kernel_size = (2,2),input_shape=(96,26,1)))

model.add(Activation("relu"))

model.add(Conv2D(82,kernel_size = (2,2),input_shape=(96,26,1)))

model.add(Activation("relu"))







model.add(Flatten())

model.add(Dense(64, activation='relu'))



model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, Cat_y, test_size=.8)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=1)
# Import Keras and other Deep Learning dependencies

from keras.models import Sequential

import time

from keras.optimizers import Adam

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model

import seaborn as sns

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.initializers import glorot_uniform

from sklearn.preprocessing import LabelBinarizer

from keras.optimizers import *

from keras.engine.topology import Layer

from keras import backend as K

from keras.regularizers import l2

K.set_image_data_format('channels_last')

import cv2

import os

from skimage import io

import numpy as np

from numpy import genfromtxt

import pandas as pd

import tensorflow as tf



import numpy.random as rng

from sklearn.utils import shuffle



%matplotlib inline

%load_ext autoreload

%reload_ext autoreload

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
X_train=X_train.reshape(X_train.shape[0],96,26,1)

X_test=X_test.reshape(X_test.shape[0],96,26,1)

train_groups = [X_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]

test_groups = [X_test[np.where(y_test==i)[0]] for i in np.unique(y_test)]

print('train groups:', [X.shape[0] for X in train_groups])

print('test groups:', [X.shape[0] for X in test_groups])
def gen_random_batch(in_groups, batch_halfsize = 8):

    out_img_a, out_img_b, out_score = [], [], []

    all_groups = list(range(len(in_groups)))

    for match_group in [True, False]:

        group_idx = np.random.choice(all_groups, size = batch_halfsize)

        out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]

        if match_group:

            b_group_idx = group_idx

            out_score += [1]*batch_halfsize

        else:

            # anything but the same group

            non_group_idx = [np.random.choice([i for i in all_groups if i!=c_idx]) for c_idx in group_idx] 

            b_group_idx = non_group_idx

            out_score += [0]*batch_halfsize

            

        out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]

            

    return np.stack(out_img_a,0), np.stack(out_img_b,0), np.stack(out_score,0)
from keras.models import Model

from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout

img_in = Input(shape = X_train.shape[1:], name = 'FeatureNet_ImageInput')

n_layer = img_in

for i in range(2):

    n_layer = Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)

    n_layer = BatchNormalization()(n_layer)

    n_layer = Activation('relu')(n_layer)

    n_layer = Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)

    n_layer = BatchNormalization()(n_layer)

    n_layer = Activation('relu')(n_layer)

    n_layer = MaxPool2D((2,2))(n_layer)

n_layer = Flatten()(n_layer)

n_layer = Dense(32, activation = 'linear')(n_layer)

n_layer = Dropout(0.5)(n_layer)

n_layer = BatchNormalization()(n_layer)

n_layer = Activation('relu')(n_layer)

feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')

feature_model.summary()
from keras.layers import concatenate

img_a_in = Input(shape = X_train.shape[1:], name = 'ImageA_Input')

img_b_in = Input(shape = X_train.shape[1:], name = 'ImageB_Input')

img_a_feat = feature_model(img_a_in)

img_b_feat = feature_model(img_b_in)

combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')

combined_features = Dense(16, activation = 'linear')(combined_features)

combined_features = BatchNormalization()(combined_features)

combined_features = Activation('relu')(combined_features)

combined_features = Dense(4, activation = 'linear')(combined_features)

combined_features = BatchNormalization()(combined_features)

combined_features = Activation('relu')(combined_features)

combined_features = Dense(1, activation = 'sigmoid')(combined_features)

similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = [combined_features], name = 'Similarity_Model')

similarity_model.summary()
similarity_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['mae'])
def siam_gen(in_groups, batch_size = 4):

    while True:

        pv_a, pv_b, pv_sim = gen_random_batch(train_groups, batch_size//2)

        yield [pv_a, pv_b], pv_sim

# we want a constant validation group to have a frame of reference for model performance

valid_a, valid_b, valid_sim = gen_random_batch(test_groups, 10)

loss_history = similarity_model.fit_generator(siam_gen(train_groups), 

                               steps_per_epoch = 100,

                               validation_data=([valid_a, valid_b], valid_sim),

                                              epochs = 100,

                                             verbose = True)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split Data

lb = LabelBinarizer()

Y = lb.fit_transform(df.label)



X_train, X_test,y_train,y_test = train_test_split(df.sequences, Y, test_size = 0.2, random_state = 1)













y_test_cat=keras.utils.to_categorical(y_test)

y_train_cat=keras.utils.to_categorical(y_train)

# Create a Count Vectorizer to gather the unique elements in sequence

vect = CountVectorizer(analyzer = 'char_wb', ngram_range = (4,4))



# Fit and Transform CountVectorizer

vect.fit(X_train)

X_train_df = vect.transform(X_train)

X_test_df = vect.transform(X_test)



#Print a few of the features

print(vect.get_feature_names()[-20:])
X_train_df.shape
#Machine Learning Algorithm (MLA) Selection and Initialization





MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    ]









#create table to compare MLA metrics

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy' ]

MLA_compare = pd.DataFrame(columns = MLA_columns)







#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

   # cv_results = model_selection.cross_validate(alg, X_train, y_train)

    alg.fit(X_train_df.toarray(), y_train)

    y_pred=alg.predict(X_test_df.toarray())

    score=metrics.accuracy_score(y_test, y_pred)

    

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] =score



    

    

    row_index+=1



    



MLA_compare

#MLA_predict




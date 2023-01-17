#################################################################################################

# MNIST - from Machine Learning to Neural Networks, from Neural Networks to Deep Learning

################################################################################################

# author Alexandre Gazagnes
# update 08/22/18
# commit v6

# Machine Learning : 
#   first dataset tour
#   dummy and naive approch
#   KNN
#   GridSearchCV and meta parametres
#   first 'pooling' method

# Neural Networks : 
#   previous features
#   MLP
#   learning rate
#   various data augmentation strategies (shift, flip, add nosise)
#   activation function
#   momuntum

# Deep Learning :
#   previous features
#   convolutional layes
#   dropout
#   batch normalization
#   loss function
#   optimization



####
# first we have import, plotting and logging instructions 
####

# import 
import os, sys, logging, random, time
from itertools import product
from collections import Iterable

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

import tensorflow as tf

import fastai

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# constants
TEST_SIZE = 0.3
CV        = 5 

# logging and warnings
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)
l = logging.INFO
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info
# import warnings
# warnings.filterwarnings('ignore')

# graph spec
%matplotlib inline
# get_ipython().magic('matplotlib inline')
sns.set()

# input data files are available in the "../input/" directory.
print(os.listdir("../input"))
####
# then we can init our dataframe objects
####

train   = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")
sample  = pd.read_csv("../input/sample_submission.csv")
# just for control
train.head()
# just for control
test.head()
# just for control
sample.head()
# let's see more about our df
print(train.ndim)
print(train.shape)
print(train.max().max())
print(train.min().min())
print(train.isna().any().any())
print(train.isna().any().describe())
print((785-1) **0.5)
# about our labels
print(train.label.value_counts(normalize=True))
####
# it is better to have a visual repr of numbers
####

def show_numbers(df, n=10, dim=28) : 
    fig, ax = plt.subplots(1, n,  figsize=(25,15))
    for i in range(n) : 
        df_img = np.array(df.iloc[i, :]).reshape(dim, dim)
        ax[i].imshow(df_img, cmap="gray")
        
####

show_numbers(train.iloc[:20, 1:])
# 10 samples of each
for i in range(10) : 
    _df = (train.loc[train.label == i, :]).iloc[:10, :].drop("label", axis=1)
    show_numbers(_df, 10)
####
# we now are going to test a dummy and a a naive model to define our worst performance level
####

# split X and y
X, y = train.drop("label", axis=1), train.label
print(X.shape)
print(y.shape)
# we will be able to deal with a 42000 * 800 matrix but it is a little bit messy
# it is better to sample this dataframe

def reduce_size(X,y=None,batch_size = 0.3) : 
    size = int(batch_size *len(X))
    if isinstance(y, Iterable) : 
        X,y = shuffle(X,y)
        X = X.iloc[:size, :]
        y = y.iloc[:size]
        return X,y
    else : 
        X = shuffle(X)
        X = X.iloc[:size, :]
        return X
    
####

X,y = reduce_size(X,y,0.05)
print(X.shape, y.shape)
####
# then we can separate test and train data
####

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])
####
# using DummyClassifier from sklearn
####

def dummy_score(X_train, X_test, y_train, y_test) : 
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return accuracy_score(y_test, y_pred)

####

np.array([dummy_score(X_train, X_test, y_train, y_test) for i in range(100)]).mean().round(2)
# ok, our dummy classifer do not give us a good representation of our basic performance...
######
# let's try a brutal GridSearchCV on KNN
######

# params_all = dict(  n_neighbors=[1,2,3,4,5,6,7,8,9,10], algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], 
#                     metric=['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis'],
#                     p = [1,2])

params_dict = dict(n_neighbors=[1,3,5], algorithm=['auto'])
grid = GridSearchCV(KNeighborsClassifier(), params_dict, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
# ok, and so on...
print(grid.best_score_, grid.best_params_)
# ok, we can now say that our basic performance is about 0.85-0.90
# let's try various algo
params_dict = dict(  n_neighbors=[1,3,5], algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'])
grid = GridSearchCV(KNeighborsClassifier(), params_dict, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
print(grid.best_params_)
print(grid.best_score_)
####
# we are able to reduce the number of examples, but what about feature's space ?
# is it possible to have a much smaller dimension space, without loosing a lot of information?
####

# define a pooling function
def pooling_vector(v) : 
    size = len(v)**0.5
    if size %2  : raise ValueError("not implemented")
    size = int(size)
    _size = int((size/2))    
    
    old_mat = np.array(v).reshape(size,size)
    new_mat = np.zeros(_size**2).reshape(_size, _size)

    idx = range(len(old_mat[0]))
    for i,j in product(idx, idx) : 
        if (i%2 or j%2): continue   
        n = np.array([  old_mat[i,j],old_mat[i,j+1],
                            old_mat[i+1,j],old_mat[i+1,j+1]]).max()
        new_mat[int(i/2), int(j/2)] = n
    return new_mat.flatten()
# just have a look, and try this new function

# create a test vector/matrix
test_vect = np.zeros(28**2).reshape(28,28)
test_vect[:,19] = test_vect[11,:] = 127

# test_vect is a matrix full of 0, with a cross of 127 in the middle 
test_vect = test_vect.flatten()
print(len(test_vect))

# apply this transformation 
new_vect = pooling_vector(test_vect)
print(len(new_vect))

# control that our cross of 888 is still there
print(new_vect.reshape(14,14))
print(784/4)
# we can control visualy the action of pooling
_test_vect = test_vect.reshape(28,28)
_new_vect = new_vect.reshape(14,14)
fig, ax = plt.subplots(1, 2,  figsize=(15,8))
ax[0].imshow(_test_vect)
ax[1].imshow(_new_vect)

# not bad but if we try with a real number, it is much more better

num_before = np.array(X_train.iloc[0, :]).reshape(28,28)
numb_after = pooling_vector(np.array(X_train.iloc[0, :])).reshape(14,14) 
fig, ax = plt.subplots(1, 2,  figsize=(15,8))
ax[0].imshow(num_before, cmap="gray")
ax[1].imshow(numb_after, cmap="gray")
# Ok for one vector of features, but for the entire dataset? 
new_X = pd.DataFrame(0, index = X.index, columns=range(int(len(X.columns)/4)))
new_X.head()
# control shape 
print(new_X.shape)
print(new_X.shape[1] * 4)
print(X.shape)
# awfull way to proceed, but easily understadable 
for i in X.index : new_X.loc[i, :] = pooling_vector(X.loc[i, :])
new_X.head()
# just have a look
print(new_X.head())
# are indexes still good?
(new_X.index == y.index).all()
# change our dataframe
X = new_X.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])
# compute loss of accuracy, dividing features space by 4 !  

params_dict = dict(n_neighbors=[1,3,5], algorithm=['auto'])
grid = GridSearchCV(KNeighborsClassifier(), params_dict, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
# ok, we can say that the performance is still good but the model is much more faster!
# function to pool a entire dataframe in one command
def pooling_dataframe(X) : 
    if "label" not in X.columns :
        new_X = pd.DataFrame(0, index = X.index, columns=range(int(len(X.columns)/4)))
        for i in X.index : new_X.loc[i, :] = pooling_vector(X.loc[i, :])
        # new_X = X.apply(pooling_vector, axis=1)
    else : 
        y = X.label
        X = X.drop("label", axis=1)
        new_X = pd.DataFrame(0, index = X.index, columns=range(int(len(X.columns)/4)))
        for i in X.index : new_X.loc[i, :] = pooling_vector(X.loc[i, :])
        # new_X = X.apply(pooling_vector, axis=1)
        new_X["label"] = y
    return new_X
####
# we now are able to compute much more faster, so we can increase our example numbers...
####

_train = train.copy()
print(_train.shape)
_train = reduce_size(_train, batch_size=0.3)
X, y = _train.drop("label", axis=1), _train.label
print(X.shape, y.shape)

X_before = X.copy()
X = pooling_dataframe(X)
X_after = X.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])
# control our lost information
show_numbers(X_before, dim=28)
show_numbers(X_after, dim=14)
# interlude
assert (1+1)==2
# and try to evaluate global performance.

params_dict = dict(n_neighbors=[1,3,5], algorithm=['auto'])
grid = GridSearchCV(KNeighborsClassifier(), params_dict, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
# and we are happy to observe a really big step in our performance quest!
print(grid.best_score_, grid.best_params_)
# and if we reduce drasitcly our train dataset vs our test_dataset

X, y = train.drop("label", axis=1), train.label
print(X.shape, y.shape)

X,y = reduce_size(X,y, 0.3)
print(X.shape, y.shape)

X = pooling_dataframe(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.7)
print([i.shape for i in (X_train, X_test, y_train, y_test)])

params_dict = dict(n_neighbors=[1,3,5], algorithm=['auto'])
grid = GridSearchCV(KNeighborsClassifier(), params_dict, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
print(grid.best_score_, grid.best_params_)
####
# do not forget to submit :/
####

# test.head()
# test.shape

# X = test
# X = pooling_dataframe(X)
# y_pred = grid.predict(X)

# subs_data = pd.DataFrame({'Label':y_pred})
# subs_data.index+=1
# subs_data.index.name='ImageId'
# subs_data.to_csv('../input/submission0.csv',index=True)
# we could try to find out the best KNN configuration, searchingg the best meta
# parametres and trying to upgrade our accuracy score 0.001 by 0.0001
# but let's go deeper and try to have a look at our first Neural Networks model : MLP
####
# We are now having a look at Neural Networks, specialy the easiest one : MLP multi layer perceptron
####

# let's draw our worst model

X, y = train.drop("label", axis=1), train.label
X,y = reduce_size(X,y, 0.3)

X_train = pooling_dataframe(X_train)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
accuracy_score(y_test, y_pred)
# not so good, let's add a GridSearchCV
grid = GridSearchCV(MLPClassifier(), {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
print(grid.best_score_, grid.best_params_)
# let's try one specific model
mlp = MLPClassifier((150, 150, 20))
grid = GridSearchCV(mlp, {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
print(grid.best_score_, grid.best_params_)
# second one 
mlp = MLPClassifier((200,200,200))
grid = GridSearchCV(mlp, {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
print(grid.best_score_, grid.best_params_)
# and just to try what about a MLP with full dataset, all featuress? 

X, y = train.drop("label", axis=1), train.label
print(X.shape, y.shape)

# X,y = reduce_size(X,y, 0.3)
# print(X.shape, y.shape)
# X = pooling_dataframe(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])

mlp = MLPClassifier((200, 200, 200, 100, 50))
grid = GridSearchCV(mlp, {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
print(grid.best_score_, grid.best_params_)
# same thing just with our pooling function

X, y = train.drop("label", axis=1), train.label
print(X.shape, y.shape)

X = pooling_dataframe(X)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])

mlp = MLPClassifier((200, 200, 200, 100, 50))
grid = GridSearchCV(mlp, {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
print(grid.best_score_, grid.best_params_)
####
# we are now focus on data augmentation, first thing first : data augmentation fo every number
####

def shift_left(v, n=2, dim=28) :
    old_matrix = np.array(v).reshape(dim,dim)
    new_matrix = np.zeros(dim**2).reshape(dim,dim)
    new_matrix[:, :-n]   = old_matrix[:,n:]
    new_matrix[:, -n:]     = left  = old_matrix[:, :n]
    return new_matrix.flatten()    

def shift_right(v, n=2, dim=28) : 
    old_matrix = np.array(v).reshape(dim,dim)
    new_matrix = np.zeros(dim**2).reshape(dim,dim)
    new_matrix[:, n:]   = old_matrix[:,:-n]
    new_matrix[:, :n] = left  = old_matrix[:, -n:]
    return new_matrix.flatten()

def shift_up(v, n=2, dim=28):
    old_matrix = np.array(v).reshape(dim,dim)
    new_matrix = np.zeros(dim**2).reshape(dim,dim)
    new_matrix[:-n, :]   = old_matrix[n:,:]
    new_matrix[-n:, :] = left  = old_matrix[:n,: ]
    return new_matrix.flatten()

def shift_down(v, n=2, dim=28) : 
    old_matrix = np.array(v).reshape(dim,dim)
    new_matrix = np.zeros(dim**2).reshape(dim,dim)
    new_matrix[n:, :]   = old_matrix[:-n,:]
    new_matrix[:n, :] = old_matrix[-n:,: ]
    return new_matrix.flatten()
# control shift action 
fig, ax = plt.subplots(1,5, figsize=(25,15))
v = train.iloc[0, 1:]
ax[0].imshow(np.array(v).reshape(28,28), cmap="gray")
ax[1].imshow(np.array(shift_left(v, n=3)).reshape(28,28), cmap="gray")
ax[2].imshow(np.array(shift_right(v, n=3)).reshape(28,28), cmap="gray")
ax[3].imshow(np.array(shift_up(v, n=3)).reshape(28,28), cmap="gray")
ax[4].imshow(np.array(shift_down(v, n=3)).reshape(28,28), cmap="gray")

####
# then 0,1 and 8 can be "flipped" horizontaly or verticaly
####

def flip_verticaly(v, dim=28) : 
    old_matrix = np.array(v).reshape(dim,dim)
    new_matrix = np.zeros(dim**2).reshape(dim,dim)
    for i in range(dim) : 
        new_matrix[i, :]   = old_matrix[(dim-1)-i,:]
    return new_matrix.flatten()

def flip_honrizontaly(v, dim=28) : 
    old_matrix = np.array(v).reshape(dim,dim)
    new_matrix = np.zeros(dim**2).reshape(dim,dim)
    for i in range(dim) : 
        new_matrix[:, i]   = old_matrix[:,(dim-1)-i]
    return new_matrix.flatten()

def flip_both(v, dim=28) :
    return flip_honrizontaly(flip_verticaly(v,dim=dim), dim=dim)

# control flip action 
for i in [0, 8, 1] : 
    fig, ax = plt.subplots(1,8, figsize=(20,10))
    v = (train.loc[train.label==i, :]).iloc[0, 1:]
    v2 = (train.loc[train.label==i, :]).iloc[1, 1:]
    ax[0].imshow(np.array(v).reshape(28,28), cmap="gray")
    ax[1].imshow(np.array(flip_verticaly(v)).reshape(28,28), cmap="gray")
    ax[2].imshow(np.array(flip_honritontaly(v)).reshape(28,28), cmap="gray")
    ax[3].imshow(np.array(flip_both(v)).reshape(28,28), cmap="gray")
    ax[4].imshow(np.array(v2).reshape(28,28), cmap="gray")
    ax[5].imshow(np.array(flip_verticaly(v2)).reshape(28,28), cmap="gray")
    ax[6].imshow(np.array(flip_honritontaly(v2)).reshape(28,28), cmap="gray")
    ax[7].imshow(np.array(flip_both(v2)).reshape(28,28), cmap="gray")
####
# we now combine shift and flip, not on one vect but on one dataframe
####

def shift_dataframe(df, y=None, dim=28) : 
    if ("label" in df.columns) and (not isinstance(y, Iterable)) : 
        label = df.label 
        _df = df.drop("label", axis=1)
        tdf = _df.copy()
    elif ("label" not in df.columns) and (isinstance(y, Iterable))  :
        _df = df.copy()
        tdf = _df.copy()         
    elif ("label" not in df.columns) and (not isinstance(y, Iterable)):
        raise ValueError("Error")
    
    for transf in [shift_right, shift_left, shift_up, shift_down] :     
        _df = _df.append(tdf.apply(transf, axis=1, dim=dim), ignore_index=True)
                      
    if ("label" in df.columns) and (not isinstance(y, Iterable)) : 
        df = _df.copy()
        df["label"] = 5 * list(label.values)
        return df
    elif ("label" not in df.columns) and (isinstance(y, Iterable))  :
        df = _df.copy()
        y = 5 * list(y.values)
        return df, y          
                                          
def flip_dataframe(df, y=None, dim=28) :     
    mask = ((df.label == 0) | (df.label ==1) | (df.label == 8))
    _df = df.loc[mask, :]
    df = df.loc[~mask, :]

    if ("label" in df.columns) and (not isinstance(y, Iterable)) : 
        _label = _df.label                            
        _df = df.drop("label", axis=1)
        tdf = _df.copy()
    elif ("label" not in df.columns) and (isinstance(y, Iterable))  :
        _df = df.copy()
        tdf = _df.copy()
        y = y.loc[~mask] 
        _y = y.loc[mask]
    elif ("label" not in df.columns) and (not isinstance(y, Iterable)):
        raise ValueError("Error")                                          
                                        
    for transf in [flip_verticaly, flip_honrizontaly, flip_both] :     
        _df = _df.append(tdf.apply(transf, axis=1, dim=dim), ignore_index=True)
               
    if ("label" in df.columns) and (not isinstance(y, Iterable)) : 
        lab = list(df.label.values)
        lab.extend(4 * list(_label.values))
        df = df.drop("label", axis=1).append(_df, ignore_index=True)
        df["label"] = lab
        return df
    elif ("label" not in df.columns) and (isinstance(y, Iterable))  :
        df = _df.copy()
        y = list(y.values)                                  
        y.extend(4 * list(_y.values))
        return df, y    
                                          
def augment_dataframe(df,y=None, dim=28) : 
    return flip_dataframe(shift_dataframe(df,y=y,dim=dim),y=y, dim=dim)
# control shift 
df = train.iloc[:2, :]
df= shift_dataframe(df,dim=28)
l = len(df)
df = df.drop("label", axis=1)

fig, ax = plt.subplots(1,l, figsize = (20,10))
for i,_ in enumerate(df.index) : 
    ax[i].imshow(df.iloc[i, :].values.reshape(28,28), cmap="gray")

# control flip
df = train.iloc[:2, :]
df= flip_dataframe(df,dim=28)
l = len(df)
print(l)
df = df.drop("label", axis=1)

fig, ax = plt.subplots(1,l, figsize = (20,10))
for i,_ in enumerate(df.index) : 
    ax[i].imshow(df.iloc[i, :].values.reshape(28,28), cmap="gray")
# control both
df = train.iloc[:2, :]
df= augment_dataframe(df)
l = len(df)
print(list(df.label))
df = df.drop("label", axis=1)
l == 2 * 5 * 4
fig, ax = plt.subplots(1,l, figsize = (40,20))
for i,_ in enumerate(df.index) : 
    ax[i].imshow(df.iloc[i, :].values.reshape(28,28), cmap="gray")
def perturb_vector(v, rate=0.02, hard_method=True):
    
    if hard_method  : method = lambda i : 0 if i >128 else 255
    else :            method = lambda i : np.random.randint(0,255)
        
    l, n= len(v), int(len(v) * rate)
    idxs = n * [1] ; idxs.extend((l-n )*[0])
    idxs = shuffle(idxs)
    v = np.array([(j if not i else method(j)) for i, j in zip(idxs,v)])
    return v
# control this fucntion
v = np.arange(128-50, 128+50)
v2 = perturb_vector(v)
v2
def perturb_dataframe(X, rate=0.02, hard_method=True) : 
    if "label" not in X.index :
        new_X = X.copy()
        new_X = X.apply(perturb_vector, axis=1, rate=rate, hard_method=hard_method)
        return new_X
    else : 
        y = X.label
        X = X.drop("label", axis=1)
        new_X = X.copy()
        new_X = X.apply(perturb_vector, axis=1, rate=rate, hard_method=hard_method)
        new_X["label"] = y
        return new_X
        
# FACTORIZATION !!!!
# we can show the "noise" created with or without 'hard_method'

df = train.iloc[:10].drop("label", axis=1)
show_numbers(df)
df = perturb_dataframe(df, hard_method=True)
show_numbers(df)

df = train.iloc[:10].drop("label", axis=1)
df = perturb_dataframe(df, hard_method=False)
show_numbers(df)
# we now are able to perturb before or after pooling

df = train.iloc[:10].drop("label", axis=1)
show_numbers(df,dim=28)
df = pooling_dataframe(df)
show_numbers(df,dim=14)
_df = perturb_dataframe(df, hard_method=True)
show_numbers(_df, dim=14)
_df = perturb_dataframe(df, hard_method=False)
show_numbers(_df, dim=14)
df = train.iloc[:10].drop("label", axis=1)
show_numbers(df,dim=28)
df = perturb_dataframe(df, hard_method=True)
show_numbers(df, dim=28)
df = pooling_dataframe(df)
show_numbers(df,dim=14)

df = train.iloc[:10].drop("label", axis=1)
show_numbers(df,dim=28)
df = perturb_dataframe(df, hard_method=False)
show_numbers(df, dim=28)
df = pooling_dataframe(df)
show_numbers(df,dim=14)
# we can that perturbing and then pooling is not good method, neither using "soft" method
# we can just pool and then use pertubating hard method
# lets now try this perturb without pooling

X, y = train.drop("label", axis=1), train.label
print(X.shape, y.shape)

X,y = reduce_size(X,y, 0.3)
print(X.shape, y.shape)
# X = pooling_dataframe(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])
X_train = perturb_dataframe(X_train, hard_method=True)

mlp = MLPClassifier((200, 200, 200, 100, 50))
grid = GridSearchCV(mlp, {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)


# lets now try this pooling and then perturb 

X, y = train.drop("label", axis=1), train.label
print(X.shape, y.shape)

X,y = reduce_size(X,y, 0.3)
print(X.shape, y.shape)
X = pooling_dataframe(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])
X_train = perturb_dataframe(X_train, hard_method=True)

mlp = MLPClassifier((200, 200, 200, 100, 50))
grid = GridSearchCV(mlp, {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
# finally let see the infulence of data augmentation 

_train = shift_dataframe(train)
X, y = _train.drop("label", axis=1), _train.label
print(X.shape, y.shape)

X,y = reduce_size(X,y, 0.3)
print(X.shape, y.shape)
X = pooling_dataframe(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
print([i.shape for i in (X_train, X_test, y_train, y_test)])
X_train = perturb_dataframe(X_train, hard_method=True)

mlp = MLPClassifier((200, 200, 200, 100, 50))
grid = GridSearchCV(mlp, {}, cv=CV)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy_score(y_test, y_pred)
# we could of course try to find out the best MLP configuration ever but again, but even if we could reach 0.96, 0.97 accuracy score
# it is better to go depper and to have look at Deep learning and scpecialy Convolutional Neural Networks
####
# we now are going to use our preprocessing tools in a 'real' deep learning networks
# we will strat with a very simple CNN model, of 6 hidden layers, build with tensor flow
# then we will build much more complex/performant model using keras
####

# classic
_train = train.copy()
print(_train.shape)
# _train = reduce_size(_train, batch_size=0.3)
print(_train.shape)
X, y = _train.drop("label", axis=1), _train.label
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15)
print([i.shape for i in (X_train, X_test, y_train, y_test)])

# preprocessing of our train set 
_train = X_train.copy()
print(_train.shape)
print(y_train.shape)
_train["label"] = y_train
_train = shift_dataframe(_train)
print(_train.shape)
# _train = flip_dataframe(_train)
X_train, y_train = _train.drop("label", axis=1), _train.label
print(X_train.shape, y_train.shape)
X_train = perturb_dataframe(X_train, hard_method=True)

X_test_len = len(X_test)
X_train_len = len(X_train)
# now we need to reshape 
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test  = X_test.values.reshape(-1, 28, 28, 1)
y_train = y_train.values.reshape(-1, 1)
y_test  = y_test.values.reshape(-1, 1)
# split and reshape X and y
# X = train.values[:,1:].reshape(-1, 28, 28, 1)
# y = train.values[:,:1].reshape(-1, 1)

# split test train

# test_size = 0.4
# L = len(train)
# X_test_len = int(L*test_size)
# X_train_len = L - X_test_len

# X_train, X_test = X[: N,:,:,:], X[N:,:,:,:]
# y_train, y_test = y[:N, :], y[N:, :]

# L = 40000
# N = 2000

# X_train = train.values[X_test_len:,1:].reshape(-1, 28, 28, 1)
# y_train = train.values[X_test_len:,:1].reshape(-1, 1)

# X_test = train.values[:X_test_len,1:].reshape(-1, 28, 28, 1)
# y_test = train.values[:X_test_len,:1].reshape(-1, 1)
# then we need to define our placesholders

x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
#####
# here we draw our layer model
####

_layer = x
_filters = [128, 128, 128, 256, 256, 512]

for f in _filters: 
    _layer = tf.layers.conv2d(_layer, filters=f, kernel_size=3, padding="SAME")
    _layer = tf.nn.relu(_layer)
    _layer = tf.nn.max_pool(_layer, ksize=(1, 2, 2, 1), strides=(1,2,2,1), padding="SAME")
    _layer = tf.nn.dropout(_layer, 0.5)
####
# we just chnage the last layer and the output layer
####

_layer = tf.layers.conv2d(_layer, filters=1024, kernel_size=3, padding="SAME")
_layer = tf.nn.relu(_layer)
_layer = tf.nn.max_pool(_layer, ksize=(1, 2, 2, 1), strides=(1,2,2,1), padding="SAME")
_layer = tf.reshape(_layer, [-1, 1024])

out = tf.layers.dense(_layer, units=10)
####
# our loss function
####

loss = tf.reduce_mean(0.5 * tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))
####
#our optimizer
####

optimizer = tf.train.AdamOptimizer().minimize(loss)

#####
# our batch parser
#####

def next_batch(batch_size, training):
    array_x = []
    array_y = []
    
    for i in range(batch_size):
        index = np.random.randint(0, X_train_len)
        array_x.append(X_train[index])
        array_y.append(np.eye(10)[y_train[index]].reshape(10))
    return { x: array_x, y: array_y}
####
# our global params
####

epochs  = 12000 # 100, 500, 1000, 3000, 8000, 12000
sess    = tf.Session()
sess.run(tf.global_variables_initializer())
####
# finally the training session
####

for i in range(epochs):
    train_batch = next_batch(128, True)
    _ = sess.run([optimizer], feed_dict=train_batch)
# we can compute accuracy

test_pred = np.array([np.argmax(item) for item in sess.run([out], feed_dict={x:X_test})[0]]).reshape(-1, 1)
print(np.sum(np.equal(y_test, test_pred))/len(test_pred))
test_pred
y_test
"""
data = train.copy()
del train
data = input_data.read_data_sets(DATA_DIR, one_hot=True)
data = input_data.read_data_sets("../input", one_hot=True)
"""
"""
y_true = tf.placeholder (tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
"""
"""
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
"""
"""
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
"""
"""
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
"""
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})
"""
"""
ans = sess.run(accuracy, feed_dict={x: data.test.images,y_true: data.test.labels})
"""
"""
print("Accuracy: {:.4}".format(ans*100))
"""
# CNN
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input_, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input_, W) + b)


def full_layer(input_, size):
    in_size = int(input_.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input_, W) + b
"""
DATA_DIR = '/tmp/data'
MINIBATCH_SIZE = 50
STEPS = 5000
"""
"""
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
"""
"""
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
"""

"""
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = full_layer(full1_drop, 10)
"""
"""
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = mnist.train.next_batch(MINIBATCH_SIZE)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1],
                                                           keep_prob: 1.0})
            print("step {}, training accuracy {}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean(
        [sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])
"""
"""
print("test accuracy: {}".format(test_accuracy))
"""



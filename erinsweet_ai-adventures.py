# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import random as rn



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

#%matplotlib inline



# plotly library

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold 

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.pipeline import Pipeline

from scipy.stats import uniform



import itertools



import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)



from keras.utils.np_utils import to_categorical

from keras.utils import np_utils



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, MaxPool2D

#from keras.layers import AvgPool2D, BatchNormalization, Reshape

from keras.optimizers import Adadelta, RMSprop, Adam

from keras.losses import categorical_crossentropy

from keras.wrappers.scikit_learn import KerasClassifier



import tensorflow as tf



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
img_rows, img_cols = 28, 28

np.random.seed(5)



def get_best_score(model):

    

    print(model.best_score_)    

    print(model.best_params_)

    print(model.best_estimator_)

    

    return model.best_score_



def print_validation_report(y_true, y_pred):

    print("Classification Report")

    print(classification_report(y_true, y_pred))

    acc_sc = accuracy_score(y_true, y_pred)

    print("Accuracy : "+ str(acc_sc))

    

    return acc_sc



def plot_confusion_matrix(y_true, y_pred):

    mtx = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8,8))

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)

    #  square=True,

    plt.ylabel('true label')

    plt.xlabel('predicted label')

    

def plot_history_loss_and_acc(history_keras_nn):



    fig, axs = plt.subplots(1,2, figsize=(12,4))



    axs[0].plot(history_keras_nn.history['loss'])

    axs[0].plot(history_keras_nn.history['val_loss'])

    axs[0].set_title('model loss')

    axs[0].set_ylabel('loss')

    axs[0].set_xlabel('epoch')

    axs[0].legend(['train', 'validation'], loc='upper left')



    axs[1].plot(history_keras_nn.history['acc'])

    axs[1].plot(history_keras_nn.history['val_acc'])

    axs[1].set_title('model accuracy')

    axs[1].set_ylabel('accuracy')

    axs[1].set_xlabel('epoch')

    axs[1].legend(['train', 'validation'], loc='upper left')



    plt.show()
# read the data

train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")



y = train["label"]

X = train.drop(["label"],axis = 1)

X_test = test
# Normalization

X = X/255.0

X_test = X_test/255.0
# for best performance, especially of the NN classfiers,

# set mode = "commit"

mode = "edit"

mode = "commit"

#



if mode == "edit" :

    nr_samples = 1200



if mode == "commit" :    

    nr_samples = 30000



y_train=y[:nr_samples]

X_train=X[:nr_samples]

start_ix_val = nr_samples 

end_ix_val = nr_samples + int(nr_samples/3)

y_val=y[start_ix_val:end_ix_val]

X_val=X[start_ix_val:end_ix_val]
print("nr_samples train data:", nr_samples)

print("start_ix_val:", start_ix_val)

print("end_ix_val:", end_ix_val)
#explore the data



print("X:")

print(X.info())

print("*"*50)

print("X_test:")

print(X_test.info())

print("*"*50)

print("y:")

print(y.shape)



print(X.iloc[0:5,:])
#  print first five samples

fig, axs = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(10,6))

axs = axs.flatten()

for i in range(0,5):

    im = X.iloc[i]

    im = im.values.reshape(-1,28,28,1)

    axs[i].imshow(im[0,:,:,0], cmap=plt.get_cmap('gray'))

    axs[i].set_title(y[i])

plt.tight_layout()    

fig, ax = plt.subplots(figsize=(8,5))

g = sns.countplot(y)
import matplotlib.pyplot as plt

%matplotlib inline



for i in range(500,505):

    sample = np.reshape(test[test.columns[1:]].iloc[i].values/255,(28,28))

    plt.figure()

    plt.title("labeled class {}".format(test["label"].iloc[i]))

    plt.imshow(sample, "gray")
# get indexes of first 10 occurences for each number

li_idxs = []

for i in range(10):

    for nr in range(10):

        ix = y[y==nr].index[i]

        li_idxs.append(ix) 

        

# and show them

fig, axs = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10,12))

axs = axs.flatten()

for n, i in enumerate(li_idxs):

    im = X.iloc[i]

    im = im.values.reshape(-1,28,28,1)

    axs[n].imshow(im[0,:,:,0], cmap=plt.get_cmap('gray'))

    axs[n].set_title(y[i])

plt.tight_layout()    
# sklearn classifiers

# Perceptron

from sklearn.linear_model import Perceptron

clf_Perceptron = Perceptron(random_state=0)

param_grid = { 'penalty': ['l1','l2'], 'tol': [0.05, 0.1] }

GridCV_Perceptron = GridSearchCV(clf_Perceptron, param_grid, verbose=1, cv=5)

GridCV_Perceptron.fit(X_train,y_train)

score_grid_Perceptron = get_best_score(GridCV_Perceptron)



pred_val_perc = GridCV_Perceptron.predict(X_val)

acc_perc = print_validation_report(y_val, pred_val_perc)

plot_confusion_matrix(y_val, pred_val_perc)

# prediction



sample_submission = pd.read_csv('../input/fashion-mnist_test.csv')

if mode == "edit" :

    X = X[:nr_samples//2]

    y = y[:nr_samples//2]

    X_test = X_test[:nr_samples//2]

    sample_submission = sample_submission[:nr_samples//2]

    

print(X.shape)

print(y.shape)

print(X_test.shape)
print(GridCV_Perceptron.best_params_)

GridCV_Perceptron.best_estimator_.fit(X,y)
t =X_test.iloc[:,:-1]

pred_test_perc = GridCV_Perceptron.best_estimator_.predict(t)

#'ImageId':sample_submission.ImageId,

result_perc = pd.DataFrame({ 'Label':pred_test_perc})

result_perc.to_csv("subm_perc.csv",index=False)
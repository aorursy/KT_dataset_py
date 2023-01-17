# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



# To plot pretty figures

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



import pandas as pd

import seaborn as sns #for better and easier plots



%matplotlib inline





import warnings

warnings.filterwarnings(action="ignore")
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head(3) #looking at the first 3 entries, just to have an overall picture of what we have at hand
train_labels = train['label'].copy()

train.drop('label', axis=1, inplace=True)
print(train.shape, test.shape) #checking the shape
train /= 255

test /= 255
fig, ax = plt.subplots(figsize=(8,5))

sns.countplot(train_labels)
list_indx = [] #creating a list of the indexes for each digit

for i in range(10):

    for nr in range(10):

        indx = train_labels[train_labels==nr].index[i]

        list_indx.append(indx) 
fig, axs = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10,12))

axs = axs.flatten() 

for n, i in enumerate(list_indx): #n for each different plot and i for each different index stored in list_index

    im = train.iloc[i]

    im = im.values.reshape(-1,28,28,1) #reshaping it to 28x28 pixels

    axs[n].imshow(im[0,:,:,0], cmap=plt.get_cmap('gray')) #making the background dark

    axs[n].set_title(train_labels[i])

plt.tight_layout()    
from sklearn.model_selection import train_test_split



train, train_val, labels_train, labels_val = train_test_split(train, train_labels, test_size=0.1, random_state=42)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



def report_and_confusion_matrix(label, prediction):

    print("Model Report")

    print(classification_report(label, prediction))

    score = accuracy_score(label, prediction)

    print("Accuracy : "+ str(score))

    

    ####################

    fig, ax = plt.subplots(figsize=(8,8)) #setting the figure size and ax

    mtx = confusion_matrix(label, prediction)

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=True, ax=ax) #create a heatmap with the values of our confusion matrix

    plt.ylabel('true label')

    plt.xlabel('predicted label')
from sklearn.linear_model import LogisticRegression

model_lgre = LogisticRegression(random_state=0)

param_grid = {'C': [0.014,0.012], 'multi_class': ['multinomial'],  

              'penalty': ['l1'],'solver': ['saga'], 'tol': [0.1] }

GridCV_LR = GridSearchCV(model_lgre, param_grid, verbose=1, cv=5)

GridCV_LR.fit(train,labels_train)

score_grid_LR = GridCV_LR.best_score_
print(score_grid_LR)
##using the logist regression parameters found in our grid search, let's predict using our validation set and see how well the model is doing

predict_lgr = GridCV_LR.predict(train_val)

report_and_confusion_matrix(labels_val, predict_lgr)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)

knn_clf.fit(train, labels_train)
predict_knn = knn_clf.predict(train_val)

report_and_confusion_matrix(labels_val, predict_knn)
from sklearn.ensemble import RandomForestClassifier



rand_forest_clf = RandomForestClassifier(random_state=42)

param_grid = {'max_depth': [15], 'max_features': [100],  

              'min_samples_split': [5],'n_estimators' : [50] }

GridCV_rd_clf = GridSearchCV(rand_forest_clf, param_grid, verbose=1, cv=5)

GridCV_rd_clf.fit(train, labels_train)

score_grid_rd = GridCV_rd_clf.best_score_
print(score_grid_rd)
predict_rand_forest = GridCV_rd_clf.predict(train_val)

report_and_confusion_matrix(labels_val, predict_rand_forest)
from sklearn.neural_network import MLPClassifier



mlp_clf = MLPClassifier(activation = "logistic", hidden_layer_sizes=(200,), random_state=42,batch_size=2000)

param_grid = { 'max_iter': [600], 'alpha': [1e-4], 

               'solver': ['sgd'], 'learning_rate_init': [0.05,0.06],'tol': [1e-4] }

    

GridCV_MLP = GridSearchCV(mlp_clf, param_grid, verbose=1, cv=3)

GridCV_MLP.fit(train,labels_train)

score_grid_MLP = GridCV_MLP.best_score_
print(score_grid_MLP)
predict_MLP = GridCV_MLP.predict(train_val)

report_and_confusion_matrix(labels_val, predict_MLP)
predict_test = GridCV_MLP.predict(test)
sub = pd.read_csv('../input/sample_submission.csv')

sub['Label'] = predict_test.astype('int64')

sub.to_csv("mlp.csv", index=False)
## loading relevant models for this part of the notebook

from keras.models import Sequential, Model

from keras.optimizers import Adam, SGD

from keras.layers import Dense, Activation, Dropout, Input, concatenate

from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping
def build_model(input_shape, n_hidden=1, n_neurons=30,optimizer = SGD(3e-3)):

    model = Sequential()

    options = {"input_shape": input_shape}

    for layer in range(n_hidden):

        model.add(Dense(n_neurons, activation="elu", kernel_initializer="he_normal", **options))

        model.add(Dropout(0.25))

        options = {}

    model.add(Dense(10, activation='softmax',**options))

    model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=['accuracy'])

    return model
y_train = to_categorical(labels_train, 10)

y_val = to_categorical(labels_val, 10)
model = build_model(input_shape=train.shape[1:], n_hidden=4, n_neurons=60, optimizer='Adam')

history = model.fit(x=train, y=y_train, epochs=60, validation_data=(train_val, y_val))
pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
prediction_keras_nn = model.predict_classes(train_val)

report_and_confusion_matrix(labels_val, prediction_keras_nn)
prediction_keras_nn_test_sub = model.predict_classes(test)

sub['Label'] = prediction_keras_nn_test_sub.astype('int64')

sub.to_csv("Keras_NN.csv", index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import the train & test sets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# create vectors with the images and the labels data



# train set

X_train = train.loc[:, train.columns != 'label']

y_train = train["label"]



# convert to numpy array

X_train = X_train.values

y_train = y_train.values



print("Dimensions of the train images' vector: {}".format(X_train.shape))

print("Dimensions of the train labels' vector: {}".format(y_train.shape))



# test set to numpy array

X_test = test.values



print("Dimensions of the test images' vector: {}".format(X_test.shape))



# number of classes & labels

n_classes = 10

labels_text = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]



# visualize some random numbers of each class

import matplotlib.pyplot as plt



fig, ax = plt.subplots(1, n_classes, figsize=(20,20))



idxs = [np.where(y_train == i)[0] for i in range(n_classes)]



for i in range(n_classes):

    k = np.random.choice(idxs[i])

    ax[i].imshow(X_train[k].reshape(28, 28), cmap="gray")

    ax[i].set_title("{}".format(labels_text[i]))
# train - val sets

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)



print("Dimensions of the train images' vector: {}".format(X_train.shape))

print("Dimensions of the train labels' vector: {}".format(y_train.shape))



print("Dimensions of the validation images' vector: {}".format(X_val.shape))

print("Dimensions of the validation labels' vector: {}".format(y_val.shape))

   
# apply PCA to reduce dimensionality



from sklearn.decomposition import PCA



pca = PCA(n_components=100, random_state=2017)

pca_fit = pca.fit(X_train)

X_train_pca = pca_fit.transform(X_train)

X_val_pca = pca_fit.transform(X_val)

X_test_pca = pca_fit.transform(X_test)



def label_proportion(y):

    '''Function that gives the proportion of each class (digit) in the dataset'''

    _, count = np.unique(y, return_counts=True)

    return np.true_divide(count, y.shape[0])

    



print("No. of available PCA images to train: {}".format(X_train_pca.shape[0]))

print("No. of available PCA images to validate: {}".format(X_val_pca.shape[0]))



print("Proporción of the classes in the train set:\n{}\n".format(label_proportion(y_train)))



# display the proportions for each class

proportions = pd.DataFrame({'Class':labels_text,'Proportion':label_proportion(y_train)})

print(proportions)
# hyperparameter tuning



from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from time import time





knn = KNeighborsClassifier()



# subset of the training set, to speed up the computations

n = 4000

images_subs = X_train_pca[0:n,]

label_subs = y_train[0:n]



# define the parameter values that should be searched 



# k (number of neighbourhoods): values from 1 to 10

k_range = list(range(1, 11))

# different weights values

weights = ['uniform', 'distance']



# create a parameter grid: map the parameter names to the values that should be searched

param_grid = dict(n_neighbors=k_range, weights = weights)



# instantiate the grid

grid = GridSearchCV(knn, param_grid, cv=4, scoring='accuracy')



# fit the grid with data & check elapsed time

start = time()

grid.fit(images_subs, label_subs)

end = time()



print("Elapsed time {} s.".format(end - start))



print('\n-----------------\n')



means = grid.cv_results_["mean_test_score"]

stds = grid.cv_results_["std_test_score"]

parameters = grid.cv_results_['params']

ranks = grid.cv_results_['rank_test_score']



# print the results for each of the different combinations. 

# The one which has rank 1 would be the most accurate.

for rank, mean, std, param in zip(ranks, means, stds, parameters):

    print("{}. Average accuracy {:.2f} with S.D {:.2f}. The parameters are: {}.".format(rank, mean*100, std*100, param))
from sklearn.metrics import accuracy_score



# Show the optimal hyperparameters 

print("Optimal value for k: {}".format(grid.best_params_["n_neighbors"]))

print("Optimal value for weights: {}".format(grid.best_params_["weights"]))



# We call the model with the best parameters

knn_model = grid.best_estimator_



# and fit it to the PCA train dataset

knn_model.fit(X_train_pca,y_train)



# predict with the model on the PCA validation data

y_pred = knn_model.predict(X_val_pca)



# Compute the accuracy

acc = accuracy_score(y_val, y_pred)



print('Accuracy of the model on the (PCA) val. data: {:.2f}% '.format(acc*100))
import itertools

from sklearn.metrics import confusion_matrix



conf_mat = confusion_matrix(y_val, y_pred)



# confusion matrix

def plot_confusion_matrix(cm, classes):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

   

    cmap=plt.cm.Blues

    fig= plt.figure(figsize=(8,5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() * 2 

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], ".2f"),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# print the matrix

plot_confusion_matrix(conf_mat, classes=labels_text)







# plot of 5 cases of wrong classification

fig, ax = plt.subplots(1, n_classes, figsize=(20,20))



idxs = [np.where((y_val == i) & (y_pred != i))[0] for i in range(n_classes)]



for i in range(n_classes):

    k = np.random.choice(idxs[i])

    ax[i].imshow(X_val[k].reshape(28, 28), cmap="gray")

    ax[i].set_title("True: {}\n Predicted: {}".format(labels_text[y_val[k]], labels_text[y_pred[k]]))

        

#import the modules

import keras

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from sklearn.model_selection import StratifiedKFold

from scipy.stats import uniform 

from scipy.stats import randint

from time import time

import random 



# set a seed for reproducibility

random.seed(2)



kf = StratifiedKFold(n_splits=4)



n_iter_search = 10

parameters = np.zeros((n_iter_search, 3))



n_neurons_dist = randint(low=20, high=1000)

n_epochs_dist = randint(low=10, high=150)

lr_dist = uniform(loc=0.0001, scale=0.2)



best_iter = -1

max_score = 0

    

start = time()

for i in range(n_iter_search):

    n_neurons = n_neurons_dist.rvs()

    n_epochs = n_epochs_dist.rvs()

    lr = lr_dist.rvs()

    

    parameters[i, 0] = n_neurons

    parameters[i, 1] = n_epochs

    parameters[i, 2] = lr

    

    print("\n Iteration number {}-----------------\n".format(i+1))

    print("Parameters: No. of neurons in the hidden layer: {}; No. of epochs: {}; Learning rate: {}".format(n_neurons, n_epochs, lr))



    scores = np.zeros(4)

    j = 0

    

    # loop through the 4-folds: create a train-test split and check the accuracy of the model on those test sets

    for train_index, test_index in kf.split(X_train_pca, y_train):

        # update train and test for this fold

        X_train_kf, X_test_kf = X_train_pca[train_index], X_train_pca[test_index]

        y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

        

        # one-hot encoding on the target data

        y_train_kf = keras.utils.to_categorical(y_train_kf, num_classes = n_classes)

        y_test_kf = keras.utils.to_categorical(y_test_kf, num_classes = n_classes)

        

        # create the neural network model -------------------

        model = Sequential()

        # hidden layer: 100: dimension of the input data: X_train_pca; n_neurons: number of neurons of the layer

        model.add(Dense(n_neurons, input_shape=(100,), activation="sigmoid"))

        # output layer: with softmax activation, as we have a classification problem

        model.add(Dense(n_classes, activation='softmax'))

        

        # compile the model with the optimizer: Stoch. gradient descent with learning rate = lr

        model.compile(optimizer=SGD(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        

        # fit the model with the number of epochs n_epochs

        model.fit(X_train_kf, y_train_kf, epochs = n_epochs, verbose=0)

        

        # check the accuracy on the test data

        score = model.evaluate(X_test_kf, y_test_kf, verbose=0)

        

        # store the score on the scores array

        scores[j] = score[1]

        j += 1

    

    # compute the mean and SD of the accuracies obtained in the previous loop

    score_mean = np.mean(scores)

    score_std = np.std(scores)

    

    print("Results: average accuracy: {:.2f}%, with {:.2f} S.D.".format(score_mean*100, score_std*100))

    

    # update the value of the maximum score achieved: to obtain the value of the parameters which gave the best acc.

    if (score_mean > max_score):

        max_score = score_mean

        best_iter = i



end = time()



print("Elapsed time: {} s.".format(end - start))

from sklearn.metrics import accuracy_score





print("Optimal number of neurons: {}".format(int(parameters[best_iter, 0])))

print("Optimal number of epochs: {}".format(int(parameters[best_iter, 1])))

print("Optimal learning rate: {:.4f}".format(parameters[best_iter, 2]))



# create the model with the optimal parameters

NN_model = Sequential()

NN_model.add(Dense(int(parameters[best_iter, 0]), input_shape=(100,), activation="sigmoid"))

NN_model.add(Dense(n_classes, activation='softmax'))



NN_model.compile(optimizer=SGD(lr=parameters[best_iter, 2]), loss='categorical_crossentropy', metrics=['accuracy'])



NN_model.fit(X_train_pca, keras.utils.to_categorical(y_train, num_classes=n_classes), epochs=int(parameters[best_iter, 1]), verbose=0)



# -------------brief explanatory note: why we need to use np.argmax----------------

# The following line returns the predicted classes for each sample: the argmax acts as the 'inverse' of to_categorical: 

# references: 1) https://forums.fast.ai/t/why-using-np-argmax-for-getting-predictions/14937/2

# 2)https://github.com/keras-team/keras/issues/4981



# The model.predict function outputs the values for the 5 neurons in the output layer. 

# We can see those values as probabilities that the given instance belongs to a given class (each class is represented by a neuron in the output layer)

# The predicted class will be the maximum of those values, i.e., the class to which the instance is most likely to belong (it is required to use the softmax activation!)

y_pred = np.argmax(NN_model.predict(X_val_pca, verbose=0), axis=1)



accuracy =  accuracy_score(y_val, y_pred)*100

print("Accuracy of the model on the (PCA) test data: {:.2f}%".format(accuracy))

print(X_test.shape)



X_val.shape
# Create output file from the first model



# predictions using the test file

predicted_classes = knn_model.predict(X_test_pca)

predicted_classes





# create output dataframe and file

output = pd.DataFrame()



output["ImageId"] = [i for i in range(1, predicted_classes.shape[0]+1)]

output["Label"] = predicted_classes



output.to_csv("predicted_classes.csv", index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Data preprocessing

from sklearn.model_selection import train_test_split, cross_val_score

from tensorflow.keras.utils import normalize



# Cross-validation for model hyperparameters search

from sklearn.model_selection import cross_val_score



# Accuracy estimation

from tensorflow.keras.metrics import SparseCategoricalAccuracy as Accuracy

from sklearn.metrics import accuracy_score



def accuracy_sklearn(model, X, y):

    y_pred = model.predict(X)

    return accuracy_score(y, y_pred)





# ANN blocks

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2



# Algorithms to compare with ANN

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv("/kaggle/input/ess9-preprocessed-data/ESS_final.csv")



# List of 10 Schwarz values 

values = ['conformity', 'tradition', 'benevolence', 'universalism', 'self_direction', 

          'stimulation', 'hedonism', 'achievement', 'power', 'security']



# Choose number of clusters you want to predict

y_3 = Data["cluster_3"]

y_4 = Data["cluster_4"]



X = Data.drop(values + ["dweight", "pweight", "total_weight", "mrat", "cluster_3", "cluster_4", "cluster_5"], axis = 1)

FACTORS_NUMBER = X.shape[1]


def ann_classifier(layers = [], dropout_rates = None, l2_lambda = 0, classes_num = 4, input_dim = FACTORS_NUMBER):

    """

    Create fully-connected ANN model

    Input:

    layers - list of units number for hidden layers starting from 1st hidden layer (Also defines the number of hidden layers)

    dropout_rates - droput rates which would be added to each hidden layer accordingly. List should be equal to layers list to apply accordingly

    C - a constant for L2 regularization which would be applied to all kernel parameters of all hidden layers

    classes_num - number of classes model would predict

    input_dim - number of features in the data based on which prediction is made

    

    Remarks: 

    Model uses ReLU activation function for hidden layers and sofrmax for output layer. 

    Batch normalization is applied to all hidden layers by default to improve learning speed.

    Input_dim is explicitly prescribed, but actually model uses it only in the first layer after output. Other input dimensions are figured out by tf automatically

    """

    model = Sequential()

    

    for i in range(0, len(layers)):

        model.add(Dense(layers[i], kernel_regularizer = l2(l2_lambda), activation = "relu",  input_dim = input_dim))



        if dropout_rates != None:

            model.add(Dropout(dropout_rates[i]))

            

        model.add(BatchNormalization())

    model.add(Dense(classes_num, activation = "softmax", input_dim = input_dim))

    return model





# Helps to train ann with learning rate decay for better optimization

def train_ann(model, X_train, y_train, X_val = None, y_val = None, batch_size = 64, epochs_per_lr = 1):

    learning_rates = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.000001]

    for lr in learning_rates:

        adam = Adam(learning_rate = lr)

        model.compile(loss = "sparse_categorical_crossentropy", optimizer = adam, metrics = ["accuracy"])

        if type(X_val) == pd.core.frame.DataFrame:

            model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = batch_size, epochs = epochs_per_lr)

        else:

            model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs_per_lr)



# Applying cross-validation to ANN model

def cross_validate_ann(X, y, layers = [], dropout_rates = None, l2_lambda = 0, classes_num = 4, folders = 3, train_size = 0.9):

    accuracy_scores = []

    for i in range(0,folders):

        model = ann_classifier(layers = layers, dropout_rates = dropout_rates, l2_lambda = l2_lambda, classes_num = classes_num)

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, test_size = 1-train_size)

        train_ann(model, X_train, y_train, X_val, y_val)

        accuracy_scores.append(ann_accuracy(model, X_val, y_val))

    return np.array(accuracy_scores)



# Returns the accuracy of the model on the test set.

def ann_accuracy(model, X, y):

    y_pred = model.predict(X)

    acc_measurer = Accuracy()

    acc_measurer.update_state(y, y_pred)

    return acc_measurer.result().numpy()

# Put in y_n where n is number of clusters you want to analyze

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_3, train_size=0.9, test_size=0.1, random_state = 0)



#Normalizing training and test data

X_train_val, X_test = normalize(X_train_val), normalize(X_test)

CLASSES_NUMBER = 3
# # ann_classifier(layers = [25])

# Train acc: ~60.1%

# [0.59438616 0.5934608  0.601789   0.60425663 0.58667487]

# 0.5961135



# # ann_classifier(layers = [50])

# Train acc: ~60.1%

# [0.60271436 0.6095003  0.5903763  0.5842073  0.5885256 ]

# 0.59506476



# # ann_classifier(layers = [75])

# Train acc: ~60.2%

# [0.58883405 0.5934608  0.60919183 0.5906848  0.6162862 ]

# 0.59969157



# # ann_classifier(layers = [100])

# Train acc: ~60.1%

# [0.6076496  0.5848242  0.59993833 0.5934608  0.5891425 ]

# 0.5950031





# # ann_classifier(layers = [125])

# Train acc: ~60.3%

# [0.59592843 0.6107341  0.60055524 0.59839606 0.6064158 ]

# 0.6024059





# # ann_classifier(layers = [20, 20])

# Train acc: ~60%

# [0.6036397  0.58328193 0.601789   0.59222704 0.5940777 ]

# 0.595003





# # ann_classifier(layers = [40, 20])

# Train acc: ~60.1%

# [0.59716225 0.5829735  0.5909932  0.59376925 0.5962369 ]

# 0.59222704





# # ann_classifier(layers = [50, 30])

# Train acc: ~60.1%

# [0.58328193 0.5897594  0.59376925 0.59993833 0.60549045]

# 0.5944479





# # ann_classifier(layers = [50, 50])

# Train acc: ~60%

# [0.58883405 0.5872918  0.60610735 0.58760023 0.6064158 ]

# 0.59524983



# # ann_classifier(layers = [60, 60])

# Train acc: ~60.1%

# [0.59161013 0.6076496  0.6079581  0.58574957 0.6070327 ]

# 0.6





# # ann_classifier(layers = [80, 40])

# Train acc: ~60.2%

# [0.5934608  0.6135102  0.57618755 0.58605796 0.59284395]

# 0.5924121



# # ann_classifier(layers = [80, 60])

# Train acc: ~60.2%

# [0.59006786 0.5946946  0.6002468  0.5962369  0.60980874]

# 0.59821093



# # ann_classifier(layers = [80, 80])

# Train acc: ~60.1%

# [0.6011721  0.5913017  0.59839606 0.58513266 0.6030228 ]

# 0.59580505



# # ann_classifier(layers = [100, 80])

# Train acc: ~60.2%

# [0.6085749  0.5783467  0.58945096 0.59592843 0.59777915]

# 0.5940161





# # ann_classifier(layers = [50, 25, 10])

# Train acc: ~60.3%

# [0.6122764  0.5940777  0.5909932  0.59777915 0.59161013]

# 0.5973473



# # ann_classifier(layers = [50, 50, 50])

# Train acc: ~60.2%

# [0.60333127 0.5848242  0.60672426 0.6101172  0.6101172 ]

# 0.6030228





# # ann_classifier(layers = [60, 60, 60])

# Train acc: ~60.3%

# [0.6196792  0.5891425  0.60148054 0.59376925 0.5909932 ]

# 0.599013



# # ann_classifier(layers = [80, 65, 20])

# Train acc: ~60.3%

# [0.6030228  0.6088834  0.5934608  0.599013   0.58605796]

# 0.59808755



# # ann_classifier(layers = [80, 60, 40])

# Train acc: ~60.2%

# [0.6107341  0.60209745 0.5987045  0.60148054 0.60333127]

# 0.6032696





# # ann_classifier(layers = [100, 50, 25])

# Train acc: ~60.2%

# [0.5835904  0.60333127 0.6057989  0.60672426 0.60055524]

# 0.6



# # ann_classifier(layers = [100, 80, 60])

# Train acc: ~60.4%

# [0.6057989  0.5980876  0.61042565 0.6122764  0.59592843]

# 0.6045034







# # ann_classifier(layers = [50, 30, 30, 15])

# Train acc: ~60.3%

# [0.59284395 0.5885256  0.5987045  0.59284395 0.59161013]

# 0.59290564



# # ann_classifier(layers = [50, 50, 50, 50])

# Train acc: ~60.2%

# [0.5863664  0.60086364 0.6045651  0.59777915 0.59376925]

# 0.5966688

    

# # ann_classifier(layers = [60, 40, 20, 10])

# Train acc: ~60.3%

# [0.5808143  0.59716225 0.6057989  0.5903763  0.5987045 ]

# 0.5945713





# # ann_classifier(layers = [60, 60, 40, 40])

# Train acc: ~60.4%

# [0.5934608  0.58667487 0.6011721  0.59222704 0.599013  ]

# 0.5945096





# # ann_classifier(layers = [75, 50, 50, 25])

# Train acc: ~60.4%

# [0.5974707  0.6064158  0.59592843 0.6011721  0.6162862 ]

# 0.6034547





# # ann_classifier(layers = [80, 60, 60, 40])

# Train acc: ~60.3%

# [0.59654534 0.5940777  0.6101172  0.6045651  0.59654534]

# 0.6003701



# # ann_classifier(layers = [80, 80, 80, 80])

# Train acc: ~60.3%

# [0.6101172 0.5885256 0.5872918 0.5919186 0.5987045]

# 0.5953115





# # ann_classifier(layers = [100, 80, 60, 40])

# Train acc: ~60.4%

# [0.59161013 0.61165947 0.6002468  0.60148054 0.582665  ]

# 0.5975324

    



# # ann_classifier(layers = [60, 60, 40, 40, 20])

# Train acc: ~60.2%

# [0.5931524  0.59993833 0.59993833 0.58821714 0.599013  ]

# 0.5960518



# # ann_classifier(layers = [80, 60, 60, 40, 40])

# Train acc: ~60.2%

# [0.59500307 0.6002468  0.5962369  0.59839606 0.5909932 ]

# 0.5961752



# #ann_classifier(layers = [100, 80, 80, 60, 60, 40, 40, 20, 20])

# Train acc: ~60.2% (Likely, depend on weigts initialization more than smaller networks)

# [0.61381865 0.58821714 0.6057989  0.6002468  0.5820481 ]

# 0.5980259
# Choose architectures with the highest difference between train and val acc



# [80, 40]



# l2_lambda = 0.001

# Train acc: ~59.8%

# [0.58945096 0.5980876  0.60333127 0.6122764  0.5863664 ]

# 59.79%



# l2_lambda = 0.01

# Train acc: ~59.4%

# [0.59222704 0.58173966 0.5823566  0.57988894 0.6024059 ]

# 58.77%



# l2_lambda = 0.1

# Train acc: ~51.5%

# [0.50030845 0.51172113 0.5231339  0.5209747  0.5070944 ]

# 51.26%



# l2_lambda = 1

# Train acc: ~50.4%

# [0.4222702  0.49876618 0.51233804 0.51758176 0.50092536]

# 49.04%



# dropout_rates = [0.05, 0.05, 0.05, 0.05]

# Train acc: ~54-58%

# [0.54781   0.5743368 0.5172733 0.5789636 0.5647748]

# Val acc: 51.7-57.8%

# 55.66%



# dropout_rates = [0.1, 0.1, 0.05, 0.05]

# Train acc: ~53-56%

# [0.490438   0.5357804  0.54565084 0.5783467  0.4833436 ]

# Val acc: 49-57.8%

# 52.67%



# dropout_rates = [0.25, 0.25, 0.1, 0.1]

# Train acc: ~45-50%

# [0.5027761  0.4299815  0.38895744 0.48704502 0.48827884]

# 45.94%

# Val acc: 38.9-50.3%





# [60, 60, 40, 40]



# l2_lambda = 0.001

# Train acc: ~59.5%

# [0.5879087 0.5968538 0.5962369 0.5940777 0.5974707]

# 59.45%



# l2_lambda = 0.01

# Train acc: ~52.6%

# [0.52899444 0.5271437  0.5234423  0.52590996 0.5243677 ]

# 52.6%





# l2_lambda = 0.1

# Train acc: ~52.3%

# [0.5064775  0.53670573 0.52004933 0.5197409  0.528686  ]

# 52.23%



# l2_lambda = 1

# Train acc: ~41.4%

# [0.41301665 0.42134485 0.41116595 0.4083899  0.41579273]

# 41.39%



# dropout_rates = [0.05, 0.05, 0.05, 0.05]

# Train acc: ~51.3 - 54.8%

# [0.51573104 0.5027761  0.53670573 0.5586058  0.5363973 ]

# Val acc: 50.3-55.9%

# 53.0%



# dropout_rates = [0.1, 0.1, 0.05, 0.05]

# Train acc: ~51.3 - 54.8%

# [0.47563234 0.5006169  0.4922887  0.4802591  0.46452808]

# Val acc: 46.5-50.1%

# 48.27%





# dropout_rates = [0.25, 0.25, 0.1, 0.1]

# Train acc: ~40-48%

# [0.39358422 0.38247994 0.4056138  0.41425046 0.3923504 ]

# 39.77%

    
# Example of model training and cross validation code

# cv_results = cross_validate_ann(X_train_val, y_train_val, layers = [20, 20], classes_num = CLASSES_NUMBER, folders = 5)

# print(cv_results)

# print(str(round(cv_results.mean() * 100, 2)) + "%")
# Based on hyperparameters search with cross-validation: the best model training and estimating final accuracy on test set

# model = ann_classifier(layers = [100, 80, 60], classes_num = CLASSES_NUMBER)

# train_ann(model, X_train_val, y_train_val)

# print("Best model result on test data:", ann_accuracy(model, X_test, y_test))

# print("Best model result on train data:", ann_accuracy(model, X_train_val, y_train_val))
# Example of model training and cross validation code

# model = LogisticRegression(C = 10 ** 15)

# cv_results = cross_val_score(model, X_train_val, y_train_val, cv = 5, scoring = "accuracy")

# print(cv_results)

# print(str(round(cv_results.mean() * 100, 2)) + "%")



# Based on hyperparameters search with cross-validation: the best model training and estimating final accuracy on test set 

# model = LogisticRegression(C = 10 ** 15)

# model.fit(X_train_val, y_train_val)

# print("Best model result on test data:", accuracy_sklearn(model, X_test, y_test))

# print("Best model result on training data:", accuracy_sklearn(model, X_train_val, y_train_val))
#RESULTS HISTORY







# DecisionTreeClassifier(min_samples_split = 300)

# [0.52953879 0.52059232 0.52938454 0.53385778 0.53649128]

# 53.0%



# DecisionTreeClassifier(min_samples_split = 500)

# [0.53894802 0.52953879 0.53015579 0.5279963  0.53664558]

# 53.27%





# DecisionTreeClassifier(min_samples_split = 800)

# [0.53817677 0.53509178 0.52923029 0.53863952 0.54034871]

# 53.4%





# DecisionTreeClassifier(min_samples_split = 1000)

# [0.53817677 0.53509178 0.52923029 0.53863952 0.54034871]

# 53.63%

    

# DecisionTreeClassifier(min_samples_split = 1500)

# [0.53817677 0.53509178 0.52923029 0.53863952 0.54034871]

# 53.31%

# Example of model training and cross validation code

# model = DecisionTreeClassifier()

# cv_results = cross_val_score(model, X_train_val, y_train_val, cv = 5, scoring = "accuracy")

# print(cv_results)

# print(str(round(cv_results.mean() * 100, 2)) + "%")



# Based on hyperparameters search with cross-validation: the best model training and estimating final accuracy on test set 

# model = DecisionTreeClassifier(min_samples_split = 1000)

# model.fit(X_train_val, y_train_val)

# print("Best model result on test data:", accuracy_sklearn(model, X_test, y_test))

# print("Best model result on training data:", accuracy_sklearn(model, X_train_val, y_train_val))
#RESULTS HISTORY



# RandomForestClassifier(n_estimators=25)

# [0.56794694 0.5622397  0.5622397  0.56517045 0.55546983]

# 56.26%



# RandomForestClassifier(n_estimators=50)

# [0.57535092 0.57041493 0.56995218 0.57751041 0.57244252]

# 57.31%





# RandomForestClassifier(n_estimators=100)

# [0.58105815 0.57658491 0.57781891 0.59154712 0.58355192]

# 58.21%



# RandomForestClassifier(n_estimators=200)

# [0.58707389 0.58599414 0.58676539 0.59200987 0.58185465]

# 58.67%
# Example of model training and cross validation code

# model = RandomForestClassifier(n_estimators=100)

# cv_results = cross_val_score(model, X_train_val, y_train_val, cv = 5, scoring = "accuracy")

# print(cv_results)

# print(str(round(cv_results.mean() * 100, 2)) + "%")



# Based on hyperparameters search with cross-validation: the best model training and estimating final accuracy on test set 

# model = RandomForestClassifier(n_estimators=200)

# model.fit(X_train_val, y_train_val)

# print("Best model result on test data:", accuracy_sklearn(model, X_test, y_test))

# print("Best model result on training data:", accuracy_sklearn(model, X_train_val, y_train_val))
# Put in y_n where n is number of clusters you want to analyze

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_4, train_size=0.9, test_size=0.1, random_state = 0)



#Normalizing training and test data

X_train_val, X_test = normalize(X_train_val), normalize(X_test)

CLASSES_NUMBER = 4
# Bonus: really deep neural network

# model = ann_classifier(layers = [130, 130, 130, 100, 100, 100, 80, 80, 80, 80, 60, 60, 60, 60], classes_num = CLASSES_NUMBER)

# train_ann(model, X_train_val, y_train_val, epochs_per_lr = 3)

# print("Best model result on test data:", ann_accuracy(model, X_test, y_test))

# print("Best model result on train data:", ann_accuracy(model, X_train_val, y_train_val))



# Best model result on test data: 0.5269295

# Best model result on train data: 0.54660165
# Results



# layers = [60, 60]

# Train acc: 53.9%

#[0.5447255 0.5228254 0.5357804 0.5459593 0.5333128]

#53.65%



# layers = [60, 60, 60]

# Train acc: 54%

# [0.52899444 0.5305367  0.52529305 0.5114127  0.52590996]

# 52.44%



# layers = [80, 80, 80]

# Train acc: 54%

# [0.5314621  0.5388649  0.53300434 0.52251697 0.54287475]

# 53.37%



# layers = [120,100, 80]

# Train acc: 53.8%

# [0.52683526 0.5438001  0.5357804  0.52961135 0.53084517]

# 53.34%
# Example of model training and cross validation code

# cv_results = cross_validate_ann(X_train_val, y_train_val, layers = [20, 20], classes_num = CLASSES_NUMBER, folders = 5)

# print(cv_results)

# print(str(round(cv_results.mean() * 100, 2)) + "%")



# Based on hyperparameters search with cross-validation: the best model training and estimating final accuracy on test set

# model = ann_classifier(layers = [60, 60], classes_num = CLASSES_NUMBER)

# train_ann(model, X_train_val, y_train_val)

# print("Best model result on test data:", ann_accuracy(model, X_test, y_test))

# print("Best model result on train data:", ann_accuracy(model, X_train_val, y_train_val))
# model = LogisticRegression(C = 10 ** 15)

# model.fit(X_train_val, y_train_val)

# print("Best model result on test data:", accuracy_sklearn(model, X_test, y_test))

# print("Best model result on training data:", accuracy_sklearn(model, X_train_val, y_train_val))
# RESULTS



# DecisionTreeClassifier(min_samples_split = 500)

# [0.47062452 0.4600432  0.46760259 0.46652268 0.46328294]

# 46.56%





# DecisionTreeClassifier(min_samples_split = 800)

# [0.53817677 0.53509178 0.52923029 0.53863952 0.54034871]

# [0.46861989 0.47053379 0.47253934 0.46066029 0.46868251]

# 46.82%



# DecisionTreeClassifier(min_samples_split = 1000)

# [0.47000771 0.46667695 0.46991669 0.46096884 0.46976242]

# 46.75%

    

# DecisionTreeClassifier(min_samples_split = 1300)

# [0.46707787 0.46821969 0.47053379 0.45988892 0.46683122]

# 46.65%



# Example of model training and cross validation code

# model = DecisionTreeClassifier()

# cv_results = cross_val_score(model, X_train_val, y_train_val, cv = 5, scoring = "accuracy")

# print(cv_results)

# print(str(round(cv_results.mean() * 100, 2)) + "%")



# Based on hyperparameters search with cross-validation: the best model training and estimating final accuracy on test set 

# model = DecisionTreeClassifier(min_samples_split = 800)

# model.fit(X_train_val, y_train_val)

# print("Best model result on test data:", accuracy_sklearn(model, X_test, y_test))

# print("Best model result on training data:", accuracy_sklearn(model, X_train_val, y_train_val))
#RESULTS HISTORY



# RandomForestClassifier(n_estimators=25)

# [0.48789514 0.48210429 0.49259488 0.48333848 0.48981796]

# 48.72%



# RandomForestClassifier(n_estimators=50)

# [0.5074788  0.50539957 0.5023141  0.49583462 0.50107991]

# 50.24%





# RandomForestClassifier(n_estimators=100)

# [0.52058597 0.51558161 0.51511879 0.51141623 0.50879358]

# 51.43%



# RandomForestClassifier(n_estimators=200)

# [0.51580571 0.52236964 0.51943844 0.50709658 0.51311324]

# 51.56%


# Example of model training and cross validation code

# model = RandomForestClassifier(n_estimators=100)

# cv_results = cross_val_score(model, X_train_val, y_train_val, cv = 5, scoring = "accuracy")

# print(cv_results)

# print(str(round(cv_results.mean() * 100, 2)) + "%")



# Based on hyperparameters search with cross-validation: the best model training and estimating final accuracy on test set 

# model = RandomForestClassifier(n_estimators=200)

# model.fit(X_train_val, y_train_val)

# print("Best model result on test data:", accuracy_sklearn(model, X_test, y_test))

# print("Best model result on training data:", accuracy_sklearn(model, X_train_val, y_train_val))
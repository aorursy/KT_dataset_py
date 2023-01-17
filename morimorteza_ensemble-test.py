import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import NearestCentroid, RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy import stats
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn import decomposition, cross_decomposition

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# number of attributes for pca
decompose_to = 5

#random numpy seed
np.random.seed(random.randint(0, 40000))


# load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# shuffle train data
train = shuffle(train)

# prepare train and test data
train_data = np.array(train[train.columns.drop('PlayerID').drop('Name').drop('TARGET_5Yrs')]).astype(float)
train_labels = np.array(train['TARGET_5Yrs'])
test_data = np.array(test[test.columns.drop('PlayerID').drop('Name')]).astype(float)


# delete the rows that have not defined attributes
for i in range(train_data.shape[0]-1, -1, -1):
    if np.isnan(train_data[i][8]):
        train_data = np.delete(train_data, i, 0)
        train_labels = np.delete(train_labels, i, 0)


# use pca on data
pca = decomposition.PCA(n_components=decompose_to)
pca.fit(train_data)
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)
print(train_data_pca.shape)

negs = train_data_pca[train_labels[:]==0]
poss = train_data_pca[train_labels[:]==1]
diff = int((poss.shape[0] - negs.shape[0])/2)
idx = np.random.randint(negs.shape[0], size=diff)
train_data_pca_old = train_data_pca.copy()
train_labels_old = train_labels.copy()

# balance data counts of negative and positive data
train_data_pca = np.concatenate((train_data_pca, negs[idx]))
train_labels = np.concatenate((train_labels, np.array([0]*diff)))
print(negs.shape)
print(poss.shape)
print(train_data_pca.shape)


# cross validation numbner of folds
fold_number = 5
fold_size = train_data_pca.shape[0] // fold_number
folds = []

accuracies = []
percisions = []
recalls = []

class_lists = []
for i in range(fold_number):
    # create cross validation data
    if(i == 0):
        train_data_new = train_data_pca[i*fold_size: (fold_number-1+i)*fold_size]
        train_labels_new = train_labels[i*fold_size: (fold_number-1+i)*fold_size]
        valid_data_new = train_data_pca[(fold_number+i-1)*fold_size:]
        valid_labels_new = train_labels[(fold_number+i-1)*fold_size:]
    else:
        train_data_new = np.concatenate((train_data_pca[i*fold_size: (fold_number-1+i)*fold_size], train_data_pca[:((fold_number+i)%fold_number-1)*fold_size]))
        valid_data_new = train_data_pca[((fold_number+i)%fold_number-1)*fold_size:((fold_number+i)%fold_number)*fold_size]
        train_labels_new = np.concatenate((train_labels[i*fold_size: (fold_number-1+i)*fold_size], train_labels[:((fold_number+i)%fold_number-1)*fold_size]))
        valid_labels_new = train_labels[((fold_number+i)%fold_number-1)*fold_size:((fold_number+i)%fold_number)*fold_size]
    
    # use different classifiers at the same time
    class_list = [
        (RandomForestClassifier(max_features = 4), 0.62),
        (RandomForestClassifier(max_features = 4, min_samples_split = 5), 0.62),
        (SVC(kernel="rbf", gamma='scale', C = 1), 0.7),
        (SVC(kernel="rbf", gamma='scale', C = 1.2), 0.7),
        (SVC(kernel="rbf", gamma='scale', C = 0.8), 0.7),
        (GradientBoostingClassifier(max_depth=5, min_samples_split=5), 0.66),
        (GradientBoostingClassifier(max_depth=6, min_samples_split=10), 0.66),
        (BaggingClassifier(), 0.65),
        (NearestCentroid(), 0.68),
        (KNeighborsClassifier(n_neighbors = 7), 0.67),
        (KNeighborsClassifier(n_neighbors = 6), 0.67),
        (KNeighborsClassifier(n_neighbors = 8), 0.67),
        (MLPClassifier(solver='lbfgs', hidden_layer_sizes=(18, 12, ), max_iter=50, batch_size=200), 0.68),
        (MLPClassifier(solver='lbfgs', hidden_layer_sizes=(18, 12, ), max_iter=60, batch_size=200), 0.68),
        (MLPClassifier(solver='lbfgs', hidden_layer_sizes=(18, 12, ), max_iter=70, batch_size=200), 0.68),
        (MLPClassifier(solver='lbfgs', hidden_layer_sizes=(15, 15, ), max_iter=70, batch_size=200), 0.68),
        (MLPClassifier(solver='lbfgs', hidden_layer_sizes=(20, 15, ), max_iter=50, batch_size=200), 0.68),
        (QuadraticDiscriminantAnalysis(reg_param=0.5), 0.67),
    ]
    
    # we will use this later
    class_lists.append(class_list)
    
    predicts = []
    for classifier, accuracy in class_list:
        # fit data for classifier
        classifier.fit(train_data_new, train_labels_new)
        prd = classifier.predict(valid_data_new)
        
        # cross validation
        #true positive
        tp = 0
        #false positive
        fp = 0
        #true negative
        tn = 0
        #false negative
        fn = 0
        for i in range(len(prd)):
            if (prd[i] == valid_labels_new[i] and prd[i] == 1):
                tp += 1
            elif (prd[i] == valid_labels_new[i] and prd[i] == 0):
                tn += 1
            elif (prd[i] != valid_labels_new[i] and prd[i] == 1):
                fp += 1
            else:
                fn += 1
        print("tn:", tn, "tp:", tp, "fn:", fn, "fp:", fp)
        prd = prd.astype(float)
        prd[prd == 1] = ((tp / (tp + fp + tn + fn)) + 1) / 2
        prd[prd == 0] = ((tn / (tp + fp + tn + fn))) / 2
        predicts.append(prd)
    
    predicts = np.array(predicts)
    
    # use weighted voting for ensemble classification
    md = np.sum(predicts, axis=0)
    md[md < len(class_list) / 2] = 0
    md[md >= len(class_list) / 2] = 1
    predicted = md
    
    #check results
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predicted)):
        if (predicted[i] == valid_labels_new[i] and predicted[i] == 1):
            tp += 1
        elif (predicted[i] == valid_labels_new[i] and predicted[i] == 0):
            tn += 1
        elif (predicted[i] != valid_labels_new[i] and predicted[i] == 1):
            fp += 1
        else:
            fn += 1
    
    accuracies.append((tp + tn) / (tp +tn + fp + fn))
    percisions.append((tp)  / (tp +fp))
    recalls.append((tp)  / (tp +fn))
    
# print average accuracy, precision and recall for all of the classifiers
print('acc:', sum(accuracies)/len(accuracies))
print('pre:', sum(percisions)/len(percisions))
print('rec:', sum(recalls)/len(recalls))

finalscale = 1.05

# predict the test data using all of the ensemble
predicts = []
for class_list in class_lists:
    for classifier, accuracy in class_list:
        prd = classifier.predict(test_data_pca)
        predicts.append(prd)
md = np.sum(np.array(predicts), axis=0)
md[md < (len(predicts) / 2) / finalscale] = 0
md[md >= (len(predicts) / 2) / finalscale] = 1
test_predicted = md.astype(int)


# save the classification as csv
cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': [test_predicted[i] for i in range(440)] }
submission = pd.DataFrame(cols)
print(submission)

submission.to_csv("submission.csv", index=False)

# show data on 2d plots
positives = train_data_pca[train_labels[:]==1]
negatives = train_data_pca[train_labels[:]==0]
positivest = test_data_pca[test_predicted[:]==1]
negativest = test_data_pca[test_predicted[:]==0]

for j in range(decompose_to):
    plt.figure(figsize=(20,15))
    for i in range(decompose_to):
        plt.subplot(4,5,i+1)
        axis = [j,i]
        a=positives[:,axis]
        plt.scatter(*zip(*a), color='r')
        a=negatives[:,axis]
        plt.scatter(*zip(*a), color='b')
        plt.title(str(axis))
    plt.show()
    plt.figure(figsize=(20,15))
    for i in range(decompose_to):
        plt.subplot(4,5,i+1)
        axis = [j,i]
        a=positivest[:,axis]
        plt.scatter(*zip(*a), color='g')
        a=negativest[:,axis]
        plt.scatter(*zip(*a), color='c')
        plt.title(str(axis))
    plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split #Splitting into test and train data



#Other import statements for this assignment

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Ignore the warnings

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
#Loading the train and test data that have already been split into the different csv files

train = pd.read_csv('../input/fashion-mnist_train.csv', dtype=int)

test = pd.read_csv('../input/fashion-mnist_test.csv', dtype=int)



#Gain the x_train and the y_train

x_train = train.drop('label', axis=1) #drop the first column 'label'

y_train = train[['label']] #set y_train to label

#Gain the x_test and the y_test

x_test = test.drop('label', axis=1)

y_test = test[['label']]
#Print the first image from x_train

plt.imshow(x_train[0:1].values.reshape((28, 28)))

plt.axis("off")

plt.show()
#Print the second image from x_train

plt.imshow(x_train[1:2].values.reshape((28, 28)))

plt.axis("off")

plt.show()
#Print the third image from x_train

plt.imshow(x_train[2:3].values.reshape((28, 28)))

plt.axis("off")

plt.show()
#Measuring the accuracy using the GaussianNB regressor

from sklearn.metrics import accuracy_score

clf = GaussianNB()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

#Printing the accuracy score of the test prediction to the train values

print(accuracy_score(y_test, y_pred))
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier





#Array that holds all the possible combinations of max_features and max_depth

combinationArray = []

#Array to hold the set of oob scores and calculate the largest out of all of them

oobArray = []

    

#This function finds a good combination of hyper-parameter values that will work for the RandomForestClassifier and the ExtraTreesClassifier

def grid_search(clf, x_train, y_train, n_estimator = 30):

    #Initialise both max_features and max_depth

    max_features = [1, 4, 16, 64, 'auto']

    max_depth = [1, 4, 16, 64, None]

    



    #creating a new array with all combinations of max_features and max_depth

    for mf in range(len(max_features)):

        for md in range(len(max_depth)):

            combinationArray.append([max_features[mf], max_depth[md]])



    #for all combinations of max_depth + max_features

    for comb in range(len(combinationArray)):

        clf_reg = ""

        #NOTE: NEED TO SPLIT MAX_FEATURES AND MAX_DEPTH

        #eg. [1, 16]; max_features_single = 1 and max_depth_single = 16

        max_features_single = combinationArray[comb][0]

        max_depth_single = combinationArray[comb][1]

        

        #e.g direct_init_clf(ElasticNet, 30, and so on)

        clf_reg = direct_init_clf(clf, n_estimator, max_features_single, max_depth_single)

        # train the classifier using the train data

        clf_reg.fit(x_train, y_train)

        

        #Score of how well the model fits the prediction

        #print('Score: ', clf_reg.score(x_train, y_train))

        

        #Retrieve the oob score and print it to the console (also add it to the oobArray)

        oobArray.append(clf_reg.oob_score_)

        print("The Out-Of-Bag Score For Is: " + str(clf_reg.oob_score_))

        #Retrieve the max_depth and max_feature and print it to the console

        print("max_features and max_depth: " + str(combinationArray[comb]))

    

    loopIndexer = 0

    highestOobScore = 0

    highestIndexer = 0

    currentHighestScore = 0

    #Finding the highest oob score and the max features and depth values at that particular index    

    for loopIndexer in range(len(oobArray)):

        #First index position in the array so it would be the current highest

        if(loopIndexer == 0):

            currentHighestScore = oobArray[loopIndexer]

            highestIndexer = 0

        else:

            if(oobArray[loopIndexer] > currentHighestScore):

                currentHighestScore = oobArray[loopIndexer]

                highestIndexer = loopIndexer

    

    highestOobScore = currentHighestScore

    # return the highest oob score and the respective max_depth and max_features values   

    return highestOobScore, combinationArray[highestIndexer]



#Function to initialise the classifier

#@params classifier name and the alpha value associated

def direct_init_clf(clf, n_estimator, max_features_single, max_depth_single):

    return clf(n_estimator, random_state = 1313512, bootstrap = True, oob_score = True, n_jobs = -1, max_features = max_features_single, max_depth = max_depth_single)
#Working with RandomForestClassifier

grid_search(RandomForestClassifier, x_train, y_train, n_estimator = 30)
#Variable to hold hyper-parameter index position and max oob index

maxOob = 0

maxOobIndex = 0

maxHyp = 0



maxHypFeature = 0

maxHypeDepth = 0

#The best hyper-parameter combination [max_features, max_depth] = [64, 16]

#print (np.amax(oobArray)) #Testing: Printing the maximum oob score

#print (np.where(oobArray == np.amax(oobArray))) #Testing: Printing the index positon of the maximum oob score

#print (combinationArray[17]) #Testing: Printing the index position of the hyper-parameters of the maximum oob score

maxOob = np.amax(oobArray)

for i in range(len(oobArray)):

    if(oobArray[i] == maxOob):

        maxOobIndex = i #maxOobIndex = 17



for m in range(len(combinationArray)):

    if (m == maxOobIndex):

        maxHyp = combinationArray[m] #maxHype = [64, 16]

        

#NEED TO SPLIT THE BEST HYPER-PARAMETER TO INDIVIDUAL PORTION

maxHypFeature = maxHyp[0]

maxHypDepth = maxHyp[1]

#Initialising the RandomForestClassifier with the best hyper-parameters

#Method 1

#rtc_clf = RandomForestClassifier(n_jobs = -1, random_state = 1313512, bootstrap = True, oob_score = True, n_estimator = 30, max_features = 64, max_depth = 16)

#Method 2

#direct_init_clf(clf, n_estimator, max_features_single, max_depth_single)

rtc_clf = direct_init_clf(RandomForestClassifier, 30, maxHypFeature, maxHypDepth)



rtc_clf.fit(x_train, y_train)



y_pred = rtc_clf.predict(x_test)



accuracy = rtc_clf.score(x_test, y_test)

print("The prediction accuracy (tested with the best hyper-parameter) is: {:0.2f}%".format(accuracy * 100))



print("The confusion matrix for RandomForestClassifier:")

print(confusion_matrix(y_test, y_pred))
#Function responsible for plotting the matrix

def plot_matrix(m, t):

    fig = plt.figure(figsize = m.shape)

    ax = fig.add_subplot(111)

    cax = ax.matshow(m)

    fig.colorbar(cax)

    ax.set_title(t)



#Calling the plot_matrix function to plot the confusion matrix

plot_matrix(confusion_matrix(y_test, y_pred), "Train Predictions Confusion Matrix Plot")
#Feature importance for RandomForestClassifier

importances = rtc_clf.feature_importances_



import seaborn as sns

sns.heatmap(importances.reshape(28,28))

plt.title("Feature Importance Under RandomForestClassifier")

plt.show()
#std = np.std([tree.feature_importances_ for tree in rtc_clf.estimators_], axis = 0)

#indices = np.argsort(importances)[::-1]

#print("Feature Rannking:")

#for f in range(x_train.shape[1]):

#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
from sklearn.ensemble import ExtraTreesClassifier



#Clear arrays of existing values before running the next classifier

#Doing this as we dont want the result from RandomForestTree to interfere with the results obtained through ExtraTreesClassifier

oobArray = []

combinationArray = []



grid_search(ExtraTreesClassifier, x_train, y_train.values.ravel(), n_estimator = 100)
#Testing to see if the arrays are cleared of RandomForest Values

#for k in range(len(combinationArray)):

#    print (combinationArray[k])



#Variable to hold hyper-parameter index position and max oob index

maximumOob = 0

maximumOobIndex = 0

maximumHyp = 0



maximumHypFeature = 0

maximumHypeDepth = 0

#The best hyper-parameter combination [max_features, max_depth] = [64, 16]

#print (np.amax(oobArray)) #Testing: Printing the maximum oob score

#print (np.where(oobArray == np.amax(oobArray))) #Testing: Printing the index positon of the maximum oob score

#print (combinationArray[17]) #Testing: Printing the index position of the hyper-parameters of the maximum oob score

maximumOob = np.amax(oobArray)

for i in range(len(oobArray)):

    if(oobArray[i] == maximumOob):

        maximumOobIndex = i #maxOobIndex = 17



for m in range(len(combinationArray)):

    if (m == maximumOobIndex):

        maximumHyp = combinationArray[m] #maxHype = [64, 16]

        

#NEED TO SPLIT THE BEST HYPER-PARAMETER TO INDIVIDUAL PORTION

maximumHypFeature = maximumHyp[0]

maximumHypeDepth = maximumHyp[1]

#Initialising the RandomForestClassifier with the best hyper-parameters

#Method 1

#rtc_clf = RandomForestClassifier(n_jobs = -1, random_state = 1313512, bootstrap = True, oob_score = True, n_estimator = 30, max_features = 64, max_depth = 16)

#Method 2

#direct_init_clf(clf, n_estimator, max_features_single, max_depth_single)

etc_clf = direct_init_clf(ExtraTreesClassifier, 100, maximumHypFeature, maximumHypeDepth)



etc_clf.fit(x_train, y_train.values.reshape(60000))



y_predict = etc_clf.predict(x_test)



accuracyScore = etc_clf.score(x_test, y_test)

print("The prediction accuracy (tested with the best hyper-parameter) is: {:0.2f}%".format(accuracyScore * 100))



print("The confusion matrix for ExtraTreesClassifier:")

print(confusion_matrix(y_test, y_predict))
#Calling the plot_matrix function to plot the confusion matrix

plot_matrix(confusion_matrix(y_test, y_predict), "Train Predictions Confusion Matrix Plot (ExtraTreesClassifier)")
#Feature importance for RandomForestClassifier

importances_etc = etc_clf.feature_importances_



import seaborn as sns

sns.heatmap(importances_etc.reshape(28,28))

plt.title("Feature Importance Under ExtraTreesClassifier")

plt.show()
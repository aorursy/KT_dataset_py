# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

#All the import statements required for the assignment

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone

from sklearn.linear_model import LinearRegression

import numpy as np

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score





#ignore the warnings

import warnings

warnings.filterwarnings('ignore')
#Loading the data, filling in blank columns with 0 and showing the first 5 data entries

wbcData = pd.read_csv('../input/wisconsin_breast_cancer.csv')

wbcData = wbcData.fillna(0)

wbcData.head()
#Showing the information about each column in the data

wbcData.info()
#Split the data to train and test, in a stratified way, 80:20

x = wbcData.iloc[:, 1:10]

y = wbcData.iloc[:, -1]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 1313512)



print(len(x_train), len(x_test), len(y_train), len(y_test))

#Both the trains should be 80% of the data = 559.2 => 559

#Both the tests should be 20% of the data = 139.8 => 140

#My results are fairly close to these values
#Initialising a SGD Classifier on the train dataset

#Random state is my ID number

sgd_clf = SGDClassifier(random_state=1313512)

sgd_clf.fit(x_train, y_train)
#function to determine output all the possible subsets for the 9 input features

def recieve_ss(full_s):

    list_representation = list(full_s)

    

    ss = []

    for i in range(1,2**len(list_representation)):

        subset_array = []

        for j in range(len(list_representation)):

            if i & 1<<j:

                subset_array.append(list_representation[j])

        ss.append(subset_array)

        

    return ss



column_set = set(['thickness', 'size', 'shape', 'adhesion', 'single', 'nuclei', 'chromatin', 'nucleoli', 'mitosis'])

ss = recieve_ss(column_set)

cols = []



#add the means of the validation cross of the fitted model onto the array

for i in range(511):

    cols.append(np.mean(cross_val_score(sgd_clf, x_train[ss[i]], y_train, cv=10, scoring="accuracy")))



#display highest stats information including highest possible accuracy and the subset and location index it's located in

import operator

index, value = max(enumerate(cols), key=operator.itemgetter(1))



#print out all the subsets (commented)

#print(cols)

print("Subset: " + str(ss[index]))

print("Cross-Validation Score: " + str(value))

print("Index Position Of The Above Subset: " + str(index))
column_set = set(['thickness', 'size', 'shape', 'adhesion', 'single', 'nuclei', 'chromatin', 'nucleoli', 'mitosis'])

ss = recieve_ss(column_set)

cols_test = []



#loop through all the subsets and fit the model using the trained data

#predict the test accuracy using the fitted model

#add all the accuracy scores onto the array

for i in range(511):

    sgd_clf.fit(x_train[ss[i]], y_train)

    prediction = sgd_clf.predict(x_test[ss[i]])

    cols_test.append(accuracy_score(y_test, prediction))



#fit the model on the index and predict the result

sgd_clf.fit(x_train[ss[index]], y_train)

y_prediction = sgd_clf.predict(x_test[ss[index]])



#print the accuracy of the prediction

print("Best Accuracy Of This Subset On The Test Data: " + str(accuracy_score(y_test, y_prediction)))

 

#display highest stats information including highest possible accuracy and the subset and location index it's located in

index, value = max(enumerate(cols_test), key=operator.itemgetter(1))





print("Best Possible Test Data Accuracy: " + str(value))

print("Subset: " + str(ss[index]))

print("Index Position: " + str(index))
#scatter-plot of cross validation and test accuracy

sns.scatterplot(cols, cols_test)
#RandomForestClassifier

#fit a random forest classifier on the x_train and y_train

rf_clf = RandomForestClassifier(n_estimators = 30, random_state = 1313512)

rf_clf.fit(x_train, y_train)
column_array = []

#add the means of the validation cross of the fitted model onto the array

for i in range(511):

    column_array.append(np.mean(cross_val_score(rf_clf, x_train[ss[i]], y_train, cv=10, scoring = "accuracy")))



#display highest stats information including highest possible accuracy and the subset and location index it's located in

index, value = max(enumerate(cols), key=operator.itemgetter(1))



print("Subset: " + str(ss[index]))

print("Best Validation Cross For The Set: " + str(value))

print("Index Position: " + str(index))
column_array_test = []



#loop through all the subsets and fit the model using the trained data

#predict the test accuracy using the fitted model

#add all the accuracy scores onto the array

for i in range(511):

    rf_clf.fit(x_train[ss[i]], y_train)

    prediction = rf_clf.predict(x_test[ss[i]])

    column_array_test.append(accuracy_score(y_test, prediction))



#fit the model on the index and predict the result

rf_clf.fit(x_train[ss[index]], y_train)

y_prediction = rf_clf.predict(x_test[ss[index]])



#print the accuracy of the prediction

print("Accuracy: " + str(accuracy_score(y_test, y_prediction)))



#display highest stats information including highest possible accuracy and the subset and location index it's located in

index, value = max(enumerate(column_array_test), key=operator.itemgetter(1))



print("Best Possible Accuracy: " + str(value))

print("Subset Of The Best Accuracy: " + str(ss[index]))

print("Index Position Of The Above Subset: " + str(index))
#creating a scatter plot for the accuracy distribution of the test and cross validation

sns.scatterplot(column_array, column_array_test)
#fit a Gaussian NB classifier on the x_train and y_train

gaussian_clf = GaussianNB(priors=None)

gaussian_clf.fit(x_train, y_train)



column = []



#add the means of the validation cross of the fitted model onto the array

for i in range(511):

    column.append(np.mean(cross_val_score(gaussian_clf, x_train[ss[i]], y_train, cv=10, scoring = "accuracy")))



#display highest stats information including highest possible accuracy and the subset and location index it's located in

index,value = max(enumerate(column), key=operator.itemgetter(1))



print("Subset: " + str(ss[index]))

print("Cross Validation Score: " + str(value))

print("Index Position: " + str(index))
columnTest = []



#loop through all the subsets and fit the model using the trained data

#predict the test accuracy using the fitted model

#add all the accuracy scores onto the array

for i in range(511):

    gaussian_clf.fit(x_train[ss[i]], y_train)

    gaussian_prediction = gaussian_clf.predict(x_test[ss[i]])

    columnTest.append(accuracy_score(y_test, gaussian_prediction))



#fit the model on the index and predict the result

gaussian_clf.fit(x_train[ss[index]], y_train)

y_prediction = gaussian_clf.predict(x_test[ss[index]])



#print the accuracy of the prediction

print("Accuracy: " + str(accuracy_score(y_test, y_prediction)))



#display highest stats information including highest possible accuracy and the subset and location index it's located in

index, value = max(enumerate(columnTest), key=operator.itemgetter(1))



print("Best Possible Accuracy: " + str(value))

print("Subset Of The Best Accuracy: " + str(ss[index]))

print("Index Position Of The Above Subset: " + str(index))
#creating a scatter plot for the accuracy distribution of the test and cross validation

sns.scatterplot(column ,columnTest)
#all the imports required for a roc_curve function and plotting

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_predict

from matplotlib import pyplot



#RANDOM FOREST CLASSIFIER

forest_clf = RandomForestClassifier(random_state = 1313512)

y_probas_forest = cross_val_predict(forest_clf, x, y, cv=10, method="predict_proba")



y_probas_forest



y_scores_forest = y_probas_forest[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y, y_scores_forest)

fpr_forest, tpr_forest, thresholds_forest



#GAUSSIAN NB 

gauss_clf = GaussianNB(priors=None)

y_probas_gauss = cross_val_predict(gauss_clf, x, y, cv=10, method="predict_proba")



y_scores_gauss = y_probas_gauss[: ,1]

fpr_gauss, tpr_gauss, thresholds_gauss = roc_curve(y, y_scores_gauss)

fpr_gauss, tpr_gauss, thresholds_gauss



#SGD CLASSIFIER

y_scores = cross_val_predict(sgd_clf, x, y, cv=10, method="decision_function")

y_scores_sgd = y_scores[:]

fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y, y_scores_sgd)

fpr_sgd, tpr_sgd, thresholds_sgd



#PLOTTING THE ROC_CURVE WITH THE THREE LABELS

pyplot.plot(fpr_forest, tpr_forest, label="Random Forest")

pyplot.plot(fpr_gauss, tpr_gauss, label="Gaussian NB")

pyplot.plot(fpr_sgd, tpr_sgd, label="SGD")

pyplot.legend(loc="lower right")
#See the array values of y_probas_forest, y_probas_gauss and y_scores

print("Random Forest Classifier: ")

y_probas_forest
print("Gaussian NB: ")

y_probas_gauss
print("SGD: ")

y_scores
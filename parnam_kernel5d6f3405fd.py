# Importing python libraries

#kaggle/python docker image: https://github.com/kaggle/docker-python

import os

print(os.listdir("../input")) # Any results you write to the current directory are saved as output.

# Input data files are available in the "../input/" directory in kaggle.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')
# importing the image dataset from kaggle. First dataset was called on kaggle page and then uploaded to this 

#file fetch command of '../input/train.csv' Please run the same on kaggle.

#Importing train dataset in kaggle

MNISTtrain = pd.read_csv('../input/train.csv')

#Importing test dataset in kaggle

MNISTtest = pd.read_csv("../input/test.csv")

#Checking the dataframe

MNISTtrain.head()
# Inspecting train dataframe

print(MNISTtrain.columns)

print(MNISTtrain.shape)

MNISTtrain.info()
# Checking dataset for missing values - both rows and columns

MNISTtrain.isnull().values.any()
#inspecting test dataframe

print(MNISTtest.columns)

print(MNISTtest.shape)

MNISTtest.info()
# Checking for the missing values

MNISTtest.isnull().values.any()
# increasing the display limit as there are 785 columns

pd.set_option('display.max_columns', 785)



# lets visualize the basic statistics of the variables

MNISTtrain.describe()
#Finding is any duplication image is there

order = list(np.sort(MNISTtrain['label'].unique()))

print(order)
#Finding the mean of count of digits

digit_means = MNISTtrain.groupby('label').mean()

digit_means.head()
# Heatmap to find if pixels are correlated

plt.figure(figsize=(30, 20))

sns.heatmap(digit_means)
#See the distribution of the digits

sns.countplot(MNISTtrain['label'])

plt.show()
# lets see the distribution in numbers

MNISTtrain['label'].astype('category').value_counts()
#Taking only 20% of the dataset in training

subset_train = MNISTtrain[0:8000]



y = subset_train.iloc[:,0]



X = subset_train.iloc[:,1:]



print(y.shape)

print(X.shape)
# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images



#Lets see digit "3" images in the data.



plt.figure(figsize=(28,28))



digit_3 = subset_train.loc[subset_train.label==3,:]

image = digit_3.iloc[:,1:]

subplots_loc = 191



for i in range(1,9):

    plt.subplot(subplots_loc)

    four = image.iloc[i].values.reshape(28, 28)

    plt.imshow(four, cmap='gray')

    subplots_loc = subplots_loc +1
# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images



#Lets see digit "4" images in the data.



plt.figure(figsize=(28,28))



digit_4 = subset_train.loc[subset_train.label==4,:]

image = digit_4.iloc[:,1:]

subplots_loc = 191



for i in range(1,9):

    plt.subplot(subplots_loc)

    four = image.iloc[i].values.reshape(28, 28)

    plt.imshow(four, cmap='gray')

    subplots_loc = subplots_loc +1
# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images



#Lets see digit "1" images in the data.





plt.figure(figsize=(28,28))



digit_1 = subset_train.loc[subset_train.label==1,:]

image = digit_1.iloc[:,1:]

subplots_loc = 191



for i in range(1,9):

    plt.subplot(subplots_loc)

    one = image.iloc[i].values.reshape(28, 28)

    plt.imshow(one, cmap='gray')

    subplots_loc = subplots_loc +1
# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images



#Lets see digit "0" images in the data.



plt.figure(figsize=(28,28))



digit_0 = subset_train.loc[subset_train.label==0,:]

image = digit_0.iloc[:,1:]

subplots_loc = 191



for i in range(1,9):

    plt.subplot(subplots_loc)

    zero = image.iloc[i].values.reshape(28, 28)

    plt.imshow(zero, cmap='gray')

    subplots_loc = subplots_loc +1
# Converting 1D array to 2D 28x28 array using reshape , to plot and view grayscale images



#Lets see digit "7" images in the data.



plt.figure(figsize=(28,28))



digit_7 = subset_train.loc[subset_train.label==7,:]

image = digit_7.iloc[:,1:]

subplots_loc = 191



for i in range(1,9):

    plt.subplot(subplots_loc)

    seven = image.iloc[i].values.reshape(28, 28)

    plt.imshow(seven, cmap='gray')

    subplots_loc = subplots_loc +1
#See the distribution of the labels in sliced data

sns.countplot(subset_train.label)
# average feature values

round(MNISTtrain.drop('label', axis=1).mean(), 2)
# Data splitting in train and test data

X_train, X_test,y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
# splitting into X and y

X = MNISTtrain.drop("label", axis = 1)

y = MNISTtrain['label']
#Scaling the data

from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()

X_train_s = scale.fit_transform(X_train)

X_test_s = scale.transform(X_test)
#Importing ML algorithm libraries

from sklearn import svm

from sklearn import metrics



# An initial SVM model with linear kernel is built

svm_linear = svm.SVC(kernel='linear')



# fitting the data in the model

svm_linear.fit(X_train_s, y_train)
# predicting from the built model

predictions = svm_linear.predict(X_test_s)

predictions[:10]
# confusion matrix and accuracy of linear SVM model



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=predictions), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=predictions))
# measure accuracy of linear SVM model

metrics.accuracy_score(y_true=y_test, y_pred=predictions)
# class-wise accuracy

class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

print(class_wise)
# rbf kernel with other hyperparameters kept to default 

svm_rbf = svm.SVC(kernel='rbf')

svm_rbf.fit(X_train_s, y_train)
# predict

predictions = svm_rbf.predict(X_test_s)



# accuracy 

print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
# class-wise accuracy

class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

print(class_wise)
from sklearn.model_selection import KFold

# creating a KFold object with 3 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of parameters (C)  and (gamma) as a list

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]



# specify model

model = svm.SVC(kernel='rbf')



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)   



# fit

model_cv.fit(X_train_s, y_train)
# results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.70, 1.05])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.70, 1.05])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.70, 1.05])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# model with optimal hyperparameters



# model

model = svm.SVC(C=10, gamma=0.01, kernel="rbf")



model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")
#Prediction of test data

#scaling of test.csv data

X_test_df = scale.transform(MNISTtest)
# prediction of test data

predicted_digit = model.predict(X_test_df)
# shape of the predicted digits

predicted_digit.shape
# Creating dataframe

data = pd.DataFrame({'Label': predicted_digit})

data.head()
#Exporting test output

data.to_csv('digi_predictions.csv', sep=",")
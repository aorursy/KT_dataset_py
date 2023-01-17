# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import SDSS DATA 2019

data_sdss = pd.DataFrame(pd.read_csv("../input/SDSS DATA 2019.csv"))
# checking if some data is missing

data_sdss.info()
# the column names

data_sdss.columns
# looking at unique values of class

data_sdss['class'].unique()
# Looking at the distribution of some features which are important for classificaion. Some of the features

# that are important for classification are ra, dec, u, g, r, i, z reshift

fig = plt.figure(figsize = (5,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

ax1.hist(data_sdss.ra)

ax1.set_title("ra")

ax2.hist(data_sdss.dec)

ax2.set_title("dec")

plt.show()
visible_band = ['u', 'g', 'i', 'r', 'z']

ax=[]

for i,band in enumerate(visible_band):

    fig1 = plt.figure()

    ax.append(fig1.add_subplot(3,2,i+1))

    ax[i].hist(data_sdss[band])

    ax[i].set_title(band)

plt.show()
# from above histogram further analysis should be done for i band as it seems not to have a distribution

data_sdss['i'].describe()
sum(data_sdss['i']!=-9999.0)
# it indicates that one of the data in i band is missing. 

# We can just ignore the object. 

# Let us check if there are other missing data

features =['ra', 'dec', 'u', 'g', 'i', 'r', 'z', 'class']

for feature in features:

    total_objects = sum(data_sdss[feature]!=-9999)

    print("The number of objects for feature: %s is %d"%(feature,total_objects))



# This shows only band 'i' has one missing data. Let us make a reduced data frame containing only

# the important features.

data_sdss_reduced = data_sdss[features]

data_sdss_reduced.head()
# replace -9999 by np.nan and drop it

data_filtered = data_sdss_reduced[data_sdss_reduced.i != -9999.00]
data_filtered.describe()
# The missing value is removed lets try to look the distribution of class



unique_class =data_filtered['class'].unique()

for class_tag in unique_class:

    count = sum(data_filtered['class'] == class_tag)

    print("fraction of %s : %f"%(class_tag, count/len(data_filtered['class'])))
# separate data to x and y

x = data_filtered[features[:-1]].values

y = data_filtered[features[-1]].values
from sklearn import preprocessing

from sklearn.model_selection import train_test_split



# labeling the cateogrical data as label encoder. i.e it will be labelled from 0 to max_unique_class -1

label_encoder = preprocessing.LabelEncoder()

label_encoder.fit(y)

y_encoded = label_encoder.transform(y)

y_encoded = y_encoded.reshape(-1,1)
# labeling the cateogrical data as the one hot label encoder i.e each cateogrical data is represented by a list of 3 elements

# the position of 1 defines the  cateogry of the class

y_reshaped = y.reshape(-1,1)

one_hot_encoder = preprocessing.OneHotEncoder()

one_hot_encoder.fit(y_reshaped)

y_hot_encoded = one_hot_encoder.transform(y_reshaped).toarray()
X_train, X_test, y_train, y_test = train_test_split(x, y_hot_encoded, test_size = 0.2, random_state =  42 )
# let us see the distribution in X_train and y_train

y_train_cateogrical = one_hot_encoder.inverse_transform(y_train)

y_train_df = pd.DataFrame(y_train_cateogrical, columns=['class'])

unique_class = y_train_df['class'].unique()

for class_tag in unique_class:

    count = sum(y_train_df['class'] == class_tag)

    print("fraction of %s : %f"%(class_tag, count/len(y_train_df['class'])))
# the distribution in y_test

y_test_cateogrical = one_hot_encoder.inverse_transform(y_test)

y_test_df = pd.DataFrame(y_test_cateogrical, columns=['class'])

unique_class = y_test_df['class'].unique()

for class_tag in unique_class:

    count = sum(y_test_df['class'] == class_tag)

    print("fraction of %s : %f"%(class_tag, count/len(y_test_df['class'])))
# This is the classification problem and I will be testing the SVM, logistic regression classifier, and KNN clasifier and finally try to use ensemble learing 

# using the best classifier to improve ther result  or use top two classifer for ensemble learning. I will also try using the ANN as the classifier.
# use logistic regression for classification

from sklearn import linear_model

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm






y_train = one_hot_encoder.inverse_transform(y_train).ravel()

y_test = one_hot_encoder.inverse_transform(y_test).ravel()
logistic = linear_model.LogisticRegression(solver = 'sag', max_iter = 1000, multi_class = 'multinomial',C = 1e3, warm_start = True, penalty='l2')


logistic.fit(X_train, y_train)
predicted_y = logistic.predict(X_test)
sum(predicted_y == y_test)/len(predicted_y)
print("The error in test set: %f"%logistic.score(X_test,y_test))

print("The error in the train set: %f"%(logistic.score(X_train, y_train)))
print(confusion_matrix(y_test,predicted_y))

plt.matshow(confusion_matrix(y_test,predicted_y))
# KNN classifier

neigh = KNeighborsClassifier(n_neighbors = 10, weights = 'uniform')
neigh.fit(X_train, y_train)
y_predicted = neigh.predict(X_test)
print("The error in test set: %f"%neigh.score(X_test,y_test))

print("The error in the train set: %f"%(neigh.score(X_train, y_train)))
print(confusion_matrix(y_test,y_predicted))

plt.matshow(confusion_matrix(y_test,y_predicted))

plt.show()
# implementation of SVM 
svm_clf = svm.SVC(gamma = 'scale', C = 1000)
svm_clf.fit(X_train, y_train)
y_predicted = svm_clf.predict(X_test)


print("The error in test set: %f"%svm_clf.score(X_test,y_test))

print("The error in the train set: %f"%(svm_clf.score(X_train, y_train)))
print(confusion_matrix(y_test,y_predicted))

plt.matshow(confusion_matrix(y_test,y_predicted))

plt.show()
# From above implementation, its seen that the support vector machine and KNN works good for the classifier. So, its good to use 

# these two classifier in ensemble learning (Voting classifier)

from sklearn.ensemble import VotingClassifier

ensemble_clf = VotingClassifier(estimators=[

    ('knn', neigh), ('svm', svm_clf)],

    voting = 'hard')

ensemble_clf.fit(X_train, y_train)
print("The error in test set: %f"%ensemble_clf.score(X_test,y_test))

print("The error in the train set: %f"%(ensemble_clf.score(X_train, y_train)))
#The ensemble classifier is not doing better than svm so the good classifier is svm.
#Import the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Loading the dataset

data = pd.read_csv('../input/HR_comma_sep.csv')
data.head()
#Describing the data

data.describe()
#Split the labels and the features

label_data = data['left']

features_data = data.drop(['left'], axis = 1)
#One hot encoding for categorical features

features_data_encoded = pd.get_dummies(features_data)



features_data_encoded.head()
#Feature scaling is needed as we have different features on different scales

from sklearn import preprocessing



numerical = ['number_project', 'average_montly_hours', 'time_spend_company']



scaler = preprocessing.MinMaxScaler()

features_data_encoded[numerical] = scaler.fit_transform(features_data_encoded[numerical])



features_data_encoded.head()
#Using SVM without any feature transformation



#Importing libraries

from sklearn.cross_validation import train_test_split

from sklearn import svm

from sklearn.metrics import accuracy_score

from time import time



#Splitting data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(features_data_encoded, label_data, test_size = 0.2, random_state = 42)

clf = svm.SVC()



#Fitting the data into classifier and measuring the time

start = time()

clf.fit(X_train, y_train)

end = time()

print("Training time: {:.2f}".format(end-start))



#Making predictions on testing data and measuring the time

start = time()

pred = clf.predict(X_test)

end = time()

print("Testing time: {:.2f}".format(end- start))



#Fetching accuracy

print("Accuracy: {:.2f}".format(accuracy_score(y_test, pred)))
#Getting pairwise correlation between the features

cov_matrix = features_data_encoded.corr()
#Eigen decomposition of covariance matrix

eig_values, eig_vectors = np.linalg.eig(cov_matrix)
print('Eigenvectors \n%s' %eig_vectors)

print('\nEigenvalues \n%s' %eig_values)
total = sum(eig_values)

explained_var = [(i / total)*100 for i in sorted(eig_values, reverse=True)]
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 6))



    plt.bar(range(20), explained_var, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
from sklearn.decomposition import PCA 

pca = PCA(n_components=12)



features_new = pca.fit_transform(features_data_encoded)
#Containing ndarrays into a dataframe

features_new = pd.DataFrame(features_new)
features_new.head()
#Implementing SVC on our reduced features

X_train, X_test, y_train, y_test = train_test_split(features_new, label_data, test_size = 0.2, random_state = 42)

clf = svm.SVC()



start = time()

clf.fit(X_train, y_train)

end = time()

print("Training time: {:.2f}".format(end-start))



start = time()

pred = clf.predict(X_test)

end = time()

print("Testing time: {:.2f}".format(end- start))



print("Accuracy: {:.2f}".format(accuracy_score(y_test, pred)))
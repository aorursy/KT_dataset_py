# Importing Libraries:

import os

import numpy as np

import pandas as pd

import seaborn as sns

import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
# Reading the file

data = pd.read_csv('../input/data.csv')
# Overall view of the data:

data.info()
# Checking the first few rows:

data.head()
# Target Variable:

data.diagnosis.unique()
data.describe()
# Dropping some of the unwanted variables:

data.drop('id',axis=1,inplace=True)

data.drop('Unnamed: 32',axis=1,inplace=True)
# Binarizing the target variable:

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))

datas.columns = list(data.iloc[:,1:32].columns)

datas['diagnosis'] = data['diagnosis']
#Looking at the number of patients with Malignant and Benign Tumors:

datas.diagnosis.value_counts().plot(kind='bar', alpha = 0.5, facecolor = 'b', figsize=(12,6))

plt.title("Diagnosis (M=1 , B=0)", fontsize = '18')

plt.ylabel("Total Number of Patients")

plt.grid(b=True)
data.columns
data_mean = data[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
plt.figure(figsize=(14,14))

foo = sns.heatmap(data_mean.corr(), vmax=1, square=True, annot=True)
_ = sns.swarmplot(y='perimeter_mean',x='diagnosis', data=data_mean)

plt.show()
# from pandas.tools.plotting import scatter_matrix



# p = sns.PairGrid(datas.ix[:,20:32], hue = 'diagnosis', palette = 'Reds')

# p.map_upper(plt.scatter, s = 20, edgecolor = 'w')

# p.map_diag(plt.hist)

# p.map_lower(sns.kdeplot)

# p.add_legend()



# p.figsize = (30,30)
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn import metrics



predictors = data_mean.columns[2:11]

target = "diagnosis"



X = data_mean.loc[:,predictors]

y = np.ravel(data.loc[:,[target]])



# Split the dataset in train and test:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print ('Shape of training set : %i || Shape of test set : %i' % (X_train.shape[0],X_test.shape[0]) )

print ('The dataset is very small so simple cross-validation approach should work here')

print ('There are very few data points so 10-fold cross validation should give us a better estimate')
# Importing the model:

from sklearn.linear_model import LogisticRegression



# Initiating the model:

lr = LogisticRegression()



scores = cross_val_score(lr, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))
# Importing the model:

from sklearn import svm



# Initiating the model:

svm = svm.SVC()



scores = cross_val_score(svm, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))
# Importing the model:

from sklearn.neighbors import KNeighborsClassifier



# Initiating the model:

knn = KNeighborsClassifier()



scores = cross_val_score(knn, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))
# Importing the model:

from sklearn.linear_model import Perceptron



# Initiating the model:

pct = Perceptron()



scores = cross_val_score(pct, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))
# Importing the model:

from sklearn.ensemble import RandomForestClassifier



# Initiating the model:

rf = RandomForestClassifier()



scores = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))
# Importing the model:

from sklearn.naive_bayes import GaussianNB



# Initiating the model:

nb = GaussianNB()



scores = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))
for i in range(1, 21):

    knn = KNeighborsClassifier(n_neighbors = i)

    score = cross_val_score(knn, X_train, y_train, scoring='accuracy' ,cv=10).mean()

    print("N = " + str(i) + " :: Score = " + str(round(score,2)))
for i in range(1, 21):

    rf = RandomForestClassifier(n_estimators = i)

    score = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()

    print("N = " + str(i) + " :: Score = " + str(round(score,2)))
from sklearn.ensemble import RandomForestClassifier



# Initiating the model:

rf = RandomForestClassifier(n_estimators=18)



rf = rf.fit(X_train, y_train)



predicted = rf.predict(X_test)



acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on test data is %s' % (round(acc_test,2)))
from sklearn.naive_bayes import GaussianNB



# Initiating the model:

nb = GaussianNB()



nb = nb.fit(X_train, y_train)



predicted = nb.predict(X_test)



acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on test data is %s' % (acc_test))
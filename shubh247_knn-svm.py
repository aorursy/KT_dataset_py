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
df = pd.read_csv('../input/data.csv')
df.head()
df.info()
#there are a 33 columns and 569 entries
y = df.diagnosis

list = ['Unnamed: 32','id','diagnosis']

X = df.drop(list,axis=1) # drop unnamed: 32 column

df.head(2)
#import libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.countplot(df['diagnosis'],label='Count')

B,M = y.value_counts()

print('Number of Benign:',B)

print('Number of Malignant:',M)
#lets draw correlation graph , we use correlation because it use to remove multi colinearity it means the column are depending on each other so we should avoid it because it use same column twice.
corr = df[df.columns[1:11]].corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr,cbar ='True',annot=True,linewidths=.5,fmt='.1f',cmap='coolwarm')



#corr = data[features_mean].corr() # .corr is used for find corelation

#plt.figure(figsize=(14,14))

#sns.heatmap(corr, cbar = 'True',square = True, annot='True', fmt= '.2f',annot_kws={'size': 15},

 #          xticklabels= features_mean, yticklabels= features_mean,

  #         cmap= 'coolwarm') # for more on heatmap
X.head()
#import the libraries

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
X = StandardScaler().fit_transform(X.values)
X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    train_size=0.8,

                                                    random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,

                           p=2, metric='minkowski')
knn.fit(X_train, y_train)
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        print("Train Result:\n")

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))

        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))



        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')

        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))

        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

        

    elif train==False:

        print("Test Result:\n")        

        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))

        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))

        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))        
print_score(knn, X_train, y_train, X_test, y_test, train=True)
print_score(knn, X_train, y_train, X_test, y_test, train=False)
from sklearn.svm import SVC #import library
model = SVC()
model.fit(X_train,y_train)  #fit the model
y_pred = model.predict(X_test)  #prediction
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))
from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train,y_train)

y_pred = linear_svc.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))
from sklearn.model_selection import train_test_split, GridSearchCV 
# C -> controls the cost of the misclassification on the training data.

# A large C value gives you the low bias and high variance

# Lower C value gives you the high bias and the lower variance.

# gamma is a free parameter in radial basis function.

# Higher gamma value leads to Higher bias and lower variance value.

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3) # put the verbose = 3
grid.fit(X_train,y_train)  #assignment: scalling the X_strain
grid.best_params_  #we find best parameter
grid.best_estimator_  #find best estimator
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))           #does not identify six value correctly 
print(classification_report(y_test, grid_predictions))
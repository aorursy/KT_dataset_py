# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

print(data.shape)

data.head()

from sklearn import preprocessing



le = preprocessing.LabelEncoder()

for i in range(0,23):

    data.iloc[:,i] = le.fit_transform(data.iloc[:,i])

data.head()
# Feature selection: remove variables no longer containing relevant information

data=data.drop(["veil-type"],axis=1)

data.head(5)
# Imports needed for the script

import seaborn as sns # making statistical graphics in Python

import matplotlib.pyplot as plt

%matplotlib inline



colormap = plt.cm.viridis



plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
data[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class', ascending=False)
# Import train_test_split

from sklearn.model_selection import train_test_split



#Split data into 70% training and 30% test

X=data.drop(['class'], axis=1)

y=data['class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 2)





#train_pct_index = int(0.7 * len(X))

#X_train, X_test = X[:train_pct_index], X[train_pct_index:]

#y_train, y_test = y[:train_pct_index], y[train_pct_index:]



print("Total number of data examples " + str(len(data.index)))

print("Number of training data examples "+str(len(X_train.index)))

print("Number of test data examples "+str(len(X_test.index)))
# Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Import accuracy_score

from sklearn.metrics import accuracy_score



# Instantiate dt

dt = DecisionTreeClassifier(criterion="entropy")
# Fit dt to the training set

dt.fit(X_train,y_train)

# Predict test set labels

y_pred = dt.predict(X_test)

# Evaluate test-set accuracy

print("Accuracy score on the test set: "+str(accuracy_score(y_test, y_pred)))
#Print decision tree

import graphviz

dot_data = export_graphviz(dt, feature_names=X.columns, filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data) 

graph
from sklearn.model_selection import KFold



def computeCVAccuracy(X,y):

    accuracy=[]

    foldAcc=[]

    for i in range(1,21): 

        kf = KFold(10,False) # K-Folds cross-validator: 10 split

        for train_index, test_index in kf.split(X):

            X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1)

            clf = DecisionTreeClassifier(criterion="entropy", max_depth = i).fit(X_train, y_train)

            score=clf.score(X_test, y_test)

            accuracy.append(score)     

        foldAcc.append(np.mean(accuracy))  

    return(foldAcc)

    

cvAccuracy=computeCVAccuracy(X,y)



df1=pd.DataFrame(cvAccuracy)

df1.columns=['10-fold cv Accuracy']

df=df1.reindex(range(1,20))

df.plot()

plt.title("Decision Tree - 10-fold Cross Validation Accuracy vs Depth of tree")

plt.xlabel("Depth of tree")

plt.ylabel("Accuracy")

plt.ylim([0.8,1])

plt.xlim([0,20])
# Instantiate dt2, set 'criterion' to 'gini'

dt2 = DecisionTreeClassifier(criterion='gini', random_state=1)
# Fit dt to the training set

dt2.fit(X_train,y_train)

# Predict test set labels

y_pred2 = dt2.predict(X_test)

# Evaluate test-set accuracy

print("Accuracy score on the test set: "+str(accuracy_score(y_test, y_pred2)))
dot_data = export_graphviz(dt2, out_file=None, 

                         feature_names=X.columns,  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
from sklearn.model_selection import KFold



def computeCVAccuracy(X,y):

    accuracy=[]

    foldAcc=[]

    for i in range(1,21): 

        kf = KFold(10,False) # K-Folds cross-validator: 10 split

        for train_index, test_index in kf.split(X):

            X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1)

            clf = DecisionTreeClassifier(criterion="gini",max_depth = i).fit(X_train, y_train)

            score=clf.score(X_test, y_test)

            accuracy.append(score)     

        foldAcc.append(np.mean(accuracy))  

    return(foldAcc)

    

cvAccuracy=computeCVAccuracy(X,y)



df1=pd.DataFrame(cvAccuracy)

df1.columns=['10-fold cv Accuracy']

df=df1.reindex(range(1,20))

df.plot()

plt.title("Decision Tree - 10-fold Cross Validation Accuracy vs Depth of tree")

plt.xlabel("Depth of tree")

plt.ylabel("Accuracy")

plt.ylim([0.8,1])

plt.xlim([0,20])
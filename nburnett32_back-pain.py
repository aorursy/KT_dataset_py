# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Dataset_spine.csv")

#df.head()

num_col = len(df.columns)

df.index[13]

df.head()

values = np.array(df)

#df.drop('Unnamed:13',1)

X = values[:,0:len(values[0,:])-3]

Y = values[:,len(values[0,:])-2]

#print(X)

#print(Y)
# Split data into test and train



X_train_final, X_test_final, Y_train_final, Y_test_final = train_test_split(X, Y, test_size=0.15, random_state=2)

# Final test set, do not touch!!!
def learning_trend(X_train,Y_train, clf):

    scores = []

    for i in range(1,9):

        X,Xa,Y,Ya = train_test_split(X_train, Y_train, test_size=.1*(1-i), random_state=0)

        clf.fit(X,Y)

        s = clf.score(X_test_final,Y_test_final)

        scores.append(1-s)

    return scores
for i in range(9,-1,1):

    print (i) 
# Function to determine the optimal "k" parameter for the kNN algorithm

def test_params(X_train,Y_train,n):

    scores = []

    scoresSVM = []

    scoresTree = []

    scoresBoost = []

    k_list = []

    for x in range (1,n):

        clf = KNeighborsClassifier(n_neighbors = x)

        clf.fit(X_train,Y_train)

        #s = clf1.score(X_test,Y_test)

        cv_score = cross_val_score(clf,X_train,Y_train)

        s = cv_score.mean()

        scores.append(s)

        k_list.append(x)

    #print(scores)

    #k = np.argmax(scores) + 2

    #k = range(1,n)

    max_score = max(scores)

    #print (len(k), len(max_score))

    return k_list, scores
k_list,score_list = test_params(X_train_final,Y_train_final,20)

best_k = np.argmax(score_list)+2

print ((k_list), (score_list))

error = [1-x for x in score_list]

plt.plot(k_list,error)

clf_kNN = KNeighborsClassifier(n_neighbors = best_k)
clf_svm = svm.SVC()

#clf_svm.fit(X_train_final, Y_train_final)

#acc = clf_svm.score(X_test_final, Y_test_final)

#print(acc)
clf_tree = tree.DecisionTreeClassifier()

clf_tree.fit(X_train_final,Y_train_final)

acc = clf_tree.score(X_test_final, Y_test_final)

print (acc)
def test_params_boost(X_train,Y_train,num):

    scores = []

    #print("In function")

    est_list = []

    #n = int(num/10)

    for x in range(1,num):

        #print("In Loop")

        clf = AdaBoostClassifier(n_estimators=x*10)

        clf.fit(X_train,Y_train)

        #s = clf1.score(X_test,Y_test)

        cv_score = cross_val_score(clf,X_train,Y_train)

        #print (cv_score)

        s = cv_score.mean()

        scores.append(s)

        est_list.append(x)

 

   # max_score = max(scores)

    #print (scores)

    return est_list, scores
clf_boost = AdaBoostClassifier(n_estimators=500)

clf_boost.fit(X_train_final,Y_train_final)

acc = clf_boost.score(X_test_final, Y_test_final)

print (acc)

num = 30

e_list, scores = test_params_boost(X_train_final, Y_train_final, num)

print(np.argmax(e_list)+10, max(scores))

print(e_list,scores)

#plt.plot(e_list*10,scores)
#print (e_list)

#print (scores)

error = [1-x for x in scores]

e_vals = [y*10 for y in e_list]

plt.plot(e_vals,error)

x = (X_test_final[0,:])

pred = clf_boost.predict(X_test_final)

print (pred)
tree_scores = learning_trend(X_train_final,Y_train_final, clf_tree)

SVM_scores = learning_trend(X_train_final,Y_train_final, clf_svm)

boost_scores = learning_trend(X_train_final,Y_train_final, clf_boost)

kNN_scores = learning_trend(X_train_final,Y_train_final, clf_kNN)
x = [y for y in range(1,9)]



trees, = plt.plot(x,tree_scores, label = "Trees")

svms, = plt.plot(x,SVM_scores, label = "SVMs")

boosts, = plt.plot(x,boost_scores, label = "Boosts")

kNN, = plt.plot(kNN_scores, label = "kNN")

plt.legend(handles = [trees, svms, boosts, kNN])

plt.show()

#, SVM_scores, boost_scores,kNN_scores])
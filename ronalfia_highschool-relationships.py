# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_mat = pd.read_csv('../input/student-mat.csv')
data_mat.dtypes
data_mat = data_mat[['address','studytime','failures','activities','nursery','higher','internet','famrel','freetime','absences','G3','romantic']]
binary_features = ['address','activities','nursery','higher','internet','romantic']

for column in binary_features:

    print(column,"-",data_mat[column].unique())
for column in binary_features:

    if (column == 'address'):

        data_mat[column] = data_mat[column].apply(lambda x: 1 if (x == 'U') else 0)

    else:

        data_mat[column] = data_mat[column].apply(lambda x: 1 if (x =='yes') else 0)

    print(column,"-",data_mat[column].unique())
plt.figure(figsize=(15,15))

sns.heatmap(data_mat.corr(),annot = True,fmt = ".2f",cbar = True)

plt.xticks(rotation=90)

plt.yticks(rotation = 0)
data_mat = data_mat[data_mat.columns.drop(['address','activities','nursery','freetime'])]
plt.figure(figsize=(15,15))

sns.heatmap(data_mat.corr(),annot = True,fmt = ".2f",cbar = True)

plt.xticks(rotation=90)

plt.yticks(rotation = 0)
absences = pd.get_dummies(data_mat['absences'], drop_first = True)

failures = pd.get_dummies(data_mat['failures'],drop_first = True)

studytime = pd.get_dummies(data_mat['studytime'],drop_first = True)



data_mat.drop(['absences','failures','studytime'], axis =1, inplace = True)

data_mat = pd.concat([data_mat,absences, failures,studytime],axis = 1)

data_mat.head()
data_matf = data_mat.drop('romantic', axis = 1)

data_matl = data_mat['romantic']
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier



kf=KFold(n_splits=10, shuffle=True, random_state=False)

dtree = DecisionTreeClassifier()



outcomesDt = []

for train_id, test_id in kf.split(data_matf,data_matl):

    X_train, X_test = data_matf.values[train_id], data_matf.values[test_id]

    y_train, y_test = data_matl.values[train_id], data_matl.values[test_id]

    dtree.fit(X_train,y_train)

    predictions = dtree.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    outcomesDt.append(accuracy)

plt.plot(range(10),outcomesDt)

plt.show()

average_error_Dt = np.mean(outcomesDt)

print("the average error is equal to ",average_error_Dt)
from sklearn.ensemble import RandomForestClassifier

Rf=RandomForestClassifier(n_estimators=10)

outcomesRf=[]

for train_id, test_id in kf.split(data_matf,data_matl):

    X_train, X_test = data_matf.values[train_id], data_matf.values[test_id]

    y_train, y_test = data_matl.values[train_id], data_matl.values[test_id]

    Rf.fit(X_train,y_train)

    predictions = Rf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    outcomesRf.append(accuracy)

plt.plot(range(10),outcomesRf)

plt.show()

print("the average error is equal to ",np.mean(outcomesRf))
ForestTreesPerformance = []

for n_trees in range(1,11,1):

    pRf=RandomForestClassifier(n_estimators=n_trees)

    outcomesRfs=[]

    for train_id, test_id in kf.split(data_matf,data_matl):

        X_train, X_test = data_matf.values[train_id], data_matf.values[test_id]

        y_train, y_test = data_matl.values[train_id], data_matl.values[test_id]

        Rf.fit(X_train,y_train)

        predictions = Rf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomesRfs.append(accuracy)

    ForestTreesPerformance.append(np.mean(outcomesRfs))

plt.plot(range(1,11,1),ForestTreesPerformance)

plt.show()

if (min(ForestTreesPerformance) > average_error_Dt):

    print("A decision tree works better than a random forest with respect to the probability error")

else:

    print("A random forest with",np.argmin(ForestTreesPerformance)+1,"trees works better than a decision tree with respect to the probability error")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')

def drawImportances(importances,columns):

    importances = clf.feature_importances_

    indices = np.argsort(importances)[::-1]



    # Print the feature ranking

    print("Feature ranking:")



    for f in range(X.shape[1]):

        print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))



    # Plot the feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],

       color="r",  align="center")

    plt.xticks(range(X.shape[1]), indices)

    plt.xlim([-1, X.shape[1]])

    plt.show()





    print(clf.feature_importances_)

    





columns = ['satisfaction_level', 'last_evaluation',

       'average_montly_hours', 'time_spend_company', 

        'left']

df = df[columns]



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

#df['salary'] = le.fit_transform(df['salary'])

#df['sales'] = le.fit_transform(df['sales'])

X = df.loc[:,'satisfaction_level':'time_spend_company']

y = df.loc[:,'left':]

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#fit a logistic regression on this and check the auc score

clf = tree.DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(roc_auc_score(y_test, y_pred))





importances = clf.feature_importances_

drawImportances(importances,columns)

#using random forest classifier

from sklearn.ensemble import RandomForestClassifier

clf =  RandomForestClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(roc_auc_score(y_test, y_pred))





importances = clf.feature_importances_

drawImportances(importances,columns)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score,ShuffleSplit, train_test_split

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.



def sol_to_list(solution):

  sol_list=[i for i, n in enumerate(solution) if n == 1]

  return sol_list



def evaluate(X,Y,classifier): #CROSS VALIDATION

    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state= 0)

    results = cross_val_score(classifier, X, Y, cv=cv,scoring='accuracy')

    print(results)

    return results.mean()



def evaluate2(X,Y,classifier): #WITHOUT CROSS VALIDATION

    train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=0,test_size=0.1)

    classifier.fit(train_X,train_y)

    predict= classifier.predict(test_X) 

    return metrics.accuracy_score(predict,test_y)



classifier = KNeighborsClassifier(n_neighbors=1)

dataset = pd.read_csv("../input/zoo.csv")

labels = pd.read_csv("../input/class.csv")



solution = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1] # Found with a feature selection algorithm called BSO

features = sol_to_list(solution)

X = dataset.iloc[:,features]

y = dataset.iloc[:,17]



print("With cross validation : ",evaluate2(X,y,classifier))

print("Without cross validation : ",evaluate2(X,y,classifier))

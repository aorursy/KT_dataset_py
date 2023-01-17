# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.neighbors import  KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import  accuracy_score



from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
if __name__ == "__main__":

    

    path = '/kaggle/input/hearts/heart.csv'

    dataset = pd.read_csv(path)



    print(dataset.head(5))

    print('')

    print(dataset['target'].describe())



    x = dataset.drop(['target'], axis=1, inplace=False)

    y = dataset['target']



    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
knn_class = KNeighborsClassifier().fit(x_train, y_train)

knn_pred = knn_class.predict(x_test)



print('')

print('Accuracy KNeighbors:', accuracy_score(knn_pred, y_test))

print('')
classifier = {

        'KNeighbors': KNeighborsClassifier(),

        'LinearSCV': LinearSVC(),

        'SVC': SVC(),

        'SGDC': SGDClassifier(),

        'DecisionTree': DecisionTreeClassifier()

    }



for name, estimator in classifier.items():

    bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=5).fit(x_train, y_train)

    bag_pred = bag_class.predict(x_test)



    print('Accuracy Bagging with {}:'.format(name), accuracy_score(bag_pred, y_test))

    print('')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.ensemble as ensemble

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV as grid_search



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



data  = pd.read_csv('../input/train.csv')



data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



median_age = data['Age'].dropna().median()



if len(data.Age[ data.Age.isnull() ]) > 0:

    data.loc[ (data.Age.isnull()), 'Age'] = median_age



    

data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1)



testdata  = pd.read_csv('../input/test.csv')



testdata['Sex'] = testdata['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



median_age = testdata['Age'].dropna().median()



if len(testdata.Age[ testdata.Age.isnull() ]) > 0:

    testdata.loc[ (testdata.Age.isnull()), 'Age'] = median_age



testdata = testdata.dropna()



ids = testdata['PassengerId'].values

testdata = testdata.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1)





train_dset = data.values

test_dset = testdata.values



forest = ensemble.RandomForestClassifier(n_estimators = 50, oob_score=True)



x_train, x_test, y_train, y_test = train_test_split(train_dset[0::, 1::], train_dset[0::, 0], test_size=0.4)



forest.fit(x_train, y_train)

# output = forest.predict(test_dset).astype(int)



forest.score(x_test, y_test)



parameters = {'n_estimators':np.arange(3,50,1)}

clf = grid_search(ensemble.RandomForestClassifier(), parameters, n_jobs=10)



clf.fit(x_train, y_train)



clf.best_estimator_.score(x_test, y_test)
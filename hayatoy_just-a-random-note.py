# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



X_train = df_train.drop('Activity', axis=1).as_matrix()

y_train = pd.get_dummies(df_train['Activity']).as_matrix()

X_test = df_test.drop('Activity', axis=1).as_matrix()

y_test = pd.get_dummies(df_test['Activity']).as_matrix()



y_train = np.argmax(y_train, axis=1)

y_test = np.argmax(y_test, axis=1)
df_train.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.svm import SVC



clf = SVC()

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV



param_grid = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'kernel': ['rbf']},

]



gs = GridSearchCV(clf, param_grid)

gs.fit(X_train, y_train)



pd.DataFrame(gs.cv_results_)
gs.score(X_test, y_test)
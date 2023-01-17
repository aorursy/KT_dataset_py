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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn import preprocessing

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")
train.shape
train.columns
train_Y=train['label']

train_X=train.loc[:,'pixel0':'pixel783']
#SVC

param_grid = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

 ]

min_max_scaler = preprocessing.MinMaxScaler()

train_X = min_max_scaler.fit_transform(train_X)

svr = SVC()

#clf = SVC()

#clf.fit(train_X, train_Y) 

clf = GridSearchCV(svr, param_grid)

clf.fit(train_X, train_Y) 

results = clf.cv_results_
results
clf.best_params_
#best:{'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
y_pred=clf.predict(train_X)
accuracy_score(train_Y, y_pred)
test=min_max_scaler.fit_transform(test)
y_test=clf.predict(test)
y_test
ids=range(1,len(y_test)+1)

submission=pd.DataFrame(columns=['ImageId','Label'])
submission['ImageId']=ids

submission['Label']=y_test
submission.head()
submission.to_csv("submission.csv", index=False)
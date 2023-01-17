# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/ml-with-python-course-project/test.csv')

train.head()
train.shape
train.gender.value_counts()
import matplotlib.pyplot as plt

_=plt.hist(train.salary)
train.describe()
_=plt.boxplot(train.salary)
train = train[train.salary<360000]
_=plt.boxplot(train.salary)
import seaborn as sns

sns.lmplot('gender','salary',data=train)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
X = train.gender

y=train.salary



X = X.values.reshape(-1,1)

X = StandardScaler().fit(X).transform(X)

X
model_params = {

    'linear': {

        'model': LinearRegression(),

        'params': {

             'fit_intercept':[True]

        }

    },

    

    'ridge' : {

        'model' : Ridge(),

        'params' : {

            'solver' :['auto', 'svd', 'cholesky', 'lsqr','sag']

        }

    },

    

    'lasso' : {

        'model' : Lasso(),

        'params' : {

            'alpha' :[0.001,0.01,0.1,0.3,0.5,0.7,0.9,1],

            'selection' : ['cyclic','random']

        }

    }

}
scores=[]



import time

start_time = time.time()



for model_name, mp in model_params.items():

    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)

    clf.fit(X,y)

    scores.append({

        'model': model_name,

        'best_score': clf.best_score_,

        'best_params': clf.best_params_

    })

    

print("--- %s seconds ---" % (time.time() - start_time))
import pandas as pd



df = pd.DataFrame(scores, columns=['model','best_score','best_params'])

df.sort_values('best_score', ascending=False)

X = train.gender

y=train.salary

X = X.values.reshape(-1,1)

X
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), Ridge(solver='sag'))

pipe.fit(X,y)
test = pd.read_csv('../input/ml-with-python-course-project/test.csv')
x_test = test.gender.values.reshape(-1,1)

y_test = test.salary

y_pred = pipe.predict(x_test)
pipe.score(x_test, y_test)
test
sl_no = test['sl_no']

prediction = pipe.predict(x_test)

output = pd.DataFrame({'Id': sl_no, 'Salary' : prediction})

output
output.to_csv('submission.csv', index=False)
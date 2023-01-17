import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
train= pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()



train_y = train['count']



submission_time = test['datetime']



train['hour'] = train['datetime'].map(lambda x : int(x[11:13]))

test['hour'] = test['datetime'].map(lambda x : int(x[11:13]))

train.drop('datetime', axis=1, inplace=True)

test.drop('datetime', axis = 1, inplace=True)

train.drop('atemp',axis=1,inplace=True)

test.drop('atemp',axis=1,inplace=True)

1
corr = train.corr()

sns.heatmap(corr, annot=True,  linewidths=.5)

1
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

train.head()
test.head()
train_y = train['count']

train_x = train.drop(['casual', 'registered', 'count'], axis = 1)
train_x.head()
train_y.head()
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
models = []

models.append(LinearRegression())

models.append(Ridge())

models.append(Lasso())

models.append(DecisionTreeRegressor())

models.append(KNeighborsRegressor())

models.append(SVR())



for model in models:

    print(type(model))

    model.fit(X_train,y_train)

    print(model.score(X_cv, y_cv))
dtre = models[3]
pred = dtre.predict(test)
print(pred)
def sub(pred):

    submission = pd.DataFrame(data = {'datetime': submission_time, 'count': pred})

    submission = submission[['datetime', 'count']]

    submission['count'] = submission['count'].astype(int)

    submission.to_csv("Submission.csv", index=False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.head()
data_train.pivot_table(index='Pclass',

columns='Survived',

values='PassengerId',

aggfunc='count')
data_train.describe()
data_train[['Pclass', 'Survived']].groupby(['Pclass'], 

as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[["Sex", "Survived"]].groupby(['Sex'], 

as_index=False).count().sort_values(by='Survived', ascending=False)
data_train['IsFemale'] = data_train['Sex'] == 'female'

data_test['IsFemale'] = data_test['Sex'] == 'female'



corr_w_surv = data_train.corr()['Survived'].sort_values()

corr_w_surv
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

rcParams['font.size'] = 14

rcParams['lines.linewidth'] = 2

rcParams['figure.figsize'] = (9, 6)

rcParams['axes.titlepad'] = 14

rcParams['savefig.pad_inches'] = 0.2
title = 'correlation with survival'

corr_w_surv.iloc[:-1].plot(kind='bar', title=title);
data_train.loc[data_train['Sex']=='male', 'Sex'] = 0

data_train.loc[data_train['Sex']=='female', 'Sex'] = 1

data_test.loc[data_test['Sex']=='male', 'Sex'] = 0

data_test.loc[data_test['Sex']=='female', 'Sex'] = 1
X_train = data_train.drop(["Survived","Embarked","Cabin","Ticket","Name","Age"], axis=1)

y_train = data_train["Survived"]

X_test  = data_test.drop(["Embarked","Cabin","Ticket","Name","Age"], axis=1)

X_test = X_test.fillna(X_test.mean())

X_train.shape, y_train.shape, X_test.shape
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_log = round(model.score(X_train, y_train) * 100, 2)

acc_log
model = KNeighborsClassifier(n_neighbors = 3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_knn = round(model.score(X_train, y_train) * 100, 2)

acc_knn
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

model.score(X_train, y_train)

acc_random_forest = round(model.score(X_train, y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame({

        "PassengerId": data_test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv', index=False)
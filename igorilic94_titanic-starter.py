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
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train.dtypes
df_train.head()
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
df_train.isna().sum()
df_train['Embarked'].value_counts()
common_embarked = df_train['Embarked'].mode()[0]

df_train.loc[df_train['Embarked'].isna(), 'Embarked'] = common_embarked
df_train.isna().sum()
df_train['Cabin'].unique()
df_train = df_train.drop(['Cabin'], axis=1)
df_train['Age'].describe()
df_train['Age'].plot.hist()
ages = df_train[df_train['Age'].notna()]['Age'].values
from scipy.stats import lognorm

params = lognorm.fit(ages)
import matplotlib.pyplot as plt



x=np.linspace(min(ages),max(ages),100)

pdf_fitted = lognorm.pdf(x, params[0], loc=params[1], scale=params[2]) # fitted distribution

plt.plot(x,pdf_fitted,'r-')

plt.hist(ages,bins=30,density=True)

plt.show()
lognorm.rvs(params[0], loc=params[1], scale=params[2], size=2)
num_na_ages = df_train['Age'].isna().sum()

random_ages = lognorm.rvs(params[0], loc=params[1], scale=params[2], size=num_na_ages)





df_train.loc[df_train['Age'].isna(), 'Age'] = random_ages
df_train.isna().sum()
df_train.groupby(['Pclass','Survived']).size().unstack().plot.bar()
df_train = pd.concat([df_train,pd.get_dummies(df_train['Pclass'],prefix='Pclass')],axis=1)

df_train.drop(['Pclass'],inplace=True,axis=1)
df_train.groupby(['Embarked','Survived']).size().unstack().plot.bar()
df_train = pd.concat([df_train,pd.get_dummies(df_train['Embarked'],prefix='Embarked')],axis=1)

df_train.drop(['Embarked'],inplace=True,axis=1)
df_train.groupby(['Sex','Survived']).size().unstack().plot.bar()
df_train = pd.concat([df_train,pd.get_dummies(df_train['Sex'],prefix='Sex')],axis=1)

df_train.drop(['Sex'],inplace=True,axis=1)
df_train.columns
X = df_train[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',

       'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_female',

       'Sex_male']].values

y = df_train[['Survived']].values.ravel()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform



logistic = LogisticRegression(solver='saga',tol=1e-2, random_state=0)

distributions = dict(C=uniform(loc=0, scale=4),

                     penalty=['l2', 'l1'])

clf = RandomizedSearchCV(logistic, distributions, random_state=0)

search = clf.fit(X_train, y_train)

logit_params = search.best_params_

print("LogisticRegresssion score: ", clf.score(X_test,y_test))
from scipy.stats import randint



svc = SVC(tol=1e-2, random_state=0)

distributions = dict(C=uniform(loc=0, scale=4),

                     degree=randint(low=1,high=10),

                    )

clf = RandomizedSearchCV(svc, distributions, random_state=0)

search = clf.fit(X_train, y_train)

svc_params = search.best_params_

print("SupportVectorClassifier score: ", clf.score(X_test,y_test))
rfc = RandomForestClassifier(random_state=0)

distributions = dict(n_estimators=randint(low=50, high=200),

                     criterion=("gini","entropy"),

                    )

clf = RandomizedSearchCV(rfc, distributions, random_state=0)

search = clf.fit(X_train, y_train)

rfc_params = search.best_params_

print("RandomForestClassifier score: ", clf.score(X_test,y_test))
knn = KNeighborsClassifier()

distributions = dict(n_neighbors=randint(low=2, high=20),

                     leaf_size=randint(low=10,high=50),

                    )

clf = RandomizedSearchCV(knn, distributions, random_state=0)

search = clf.fit(X_train, y_train)

knn_params = search.best_params_

print("RandomForestClassifier score: ", clf.score(X_test,y_test))
from scipy.stats import randint



svc = SVC(tol=1e-2, random_state=0, **svc_params)

search = svc.fit(X, y)



svc.score(X,y)
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



df_test.isna().sum()
df_test.loc[df_test['Fare'].isna(), 'Fare'] = df_train['Fare'].mean()
passenger_ids = df_test.pop('PassengerId')

df_test = df_test.drop(['Name', 'Ticket'], axis=1)

df_test.loc[df_test['Embarked'].isna(), 'Embarked'] = common_embarked

df_test = df_test.drop(['Cabin'], axis=1)



num_na_ages = df_test['Age'].isna().sum()

random_ages = lognorm.rvs(params[0], loc=params[1], scale=params[2], size=num_na_ages)

df_test.loc[df_test['Age'].isna(), 'Age'] = random_ages



df_test = pd.concat([df_test,pd.get_dummies(df_test['Pclass'],prefix='Pclass')],axis=1)

df_test.drop(['Pclass'],inplace=True,axis=1)

df_test = pd.concat([df_test,pd.get_dummies(df_test['Embarked'],prefix='Embarked')],axis=1)

df_test.drop(['Embarked'],inplace=True,axis=1)

df_test = pd.concat([df_test,pd.get_dummies(df_test['Sex'],prefix='Sex')],axis=1)

df_test.drop(['Sex'],inplace=True,axis=1)



X_subm = df_test[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',

       'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_female',

       'Sex_male']].values



X_subm = scaler.transform(X_subm)
y_subm = svc.predict(X_subm)
df_submission = pd.DataFrame({'Survived': y_subm})
df_submission = pd.concat([passenger_ids,df_submission],axis=1)
df_submission.to_csv('submission.csv', index=False)
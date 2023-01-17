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



train=pd.read_csv('../input/titanic/train.csv')

test =pd.read_csv("../input/titanic/test.csv")
import seaborn as sns

import matplotlib.pyplot as plt
train.head(10)
train['Survived'].value_counts()
sns.countplot(train['Survived'])

train.iloc[:, 11]
train = train.drop(['Name', 'Cabin', 'Ticket'], axis = 1)

test = test.drop(['Name', 'Cabin', 'Ticket'], axis = 1)
train.head()

train.head()
sns.heatmap(train.isnull(),cbar = False)
sns.heatmap(test.isnull(),cbar = False)
"""

### dummy category by sklearn

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



#encode the sex column



train['Sex'] = labelencoder.fit_transform( train['Sex'].values)

test['Sex'] = labelencoder.fit_transform( test['Sex'].values)

print(train['Sex'].unique())

print(test['Sex'].unique())

"""



###dummy by pandas and numpy 

"""

### method to select the columns without missing value, since there is not missing value in the String columns we could

### skip this process

train_null_counts = train.isnull().sum()

print(train_null_counts)

df_no_mv = train[train_null_counts[train_null_counts==0].index]

"""

### to select the columns contains only the String as 'Sex', 'Embarked'

text_cols = train.select_dtypes(include=['object']).columns

print(text_cols)

### and convert its type to 'category' and do the same thing to 'test' data set

for col in text_cols:

    train[col] = train[col].astype('category')

    test[col] = test[col].astype('category')

train.head()
### dummy train data set process by pandas 

dummy_cols = pd.DataFrame()

print(text_cols)

for col in text_cols:

    col_dummies = pd.get_dummies(train[col])

    train = pd.concat([train, col_dummies], axis = 1)

    del train[col]



train.head()

dummy_cols_test = pd.DataFrame()

print(text_cols)

for col in text_cols:

    col_dummies = pd.get_dummies(test[col])

    test = pd.concat([test, col_dummies], axis = 1)

    del test[col]



test.head()
train['Age'].fillna(train['Age'].mean(), inplace = True)

test['Age'].fillna(test['Age'].mean(), inplace = True)

test['Fare'].fillna(test['Fare'].mean(), inplace = True)

sns.heatmap(test.isnull(),cbar = False)
X = train.loc[:, 'Pclass':'S'].values

Y = train.loc[:, 'Survived'].values

print(X)
# let's see if this will help getting a best result

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

print(X)
def models(X_train, Y_train):



  #logistic regression

  from sklearn.linear_model import LogisticRegression

  log = LogisticRegression(random_state = 0)

  log.fit(X_train, Y_train)



  #Knn

  from sklearn.neighbors import KNeighborsClassifier

  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)

  knn.fit(X_train, Y_train)

  

  #SVC

  from sklearn.svm import SVC

  svc_lin = SVC(kernel = 'linear', random_state = 0)

  svc_lin.fit(X_train, Y_train)



  #Gaussian NB

  from sklearn.naive_bayes import GaussianNB

  gauss = GaussianNB()

  gauss.fit(X_train, Y_train)



  #Descision Tree

  from sklearn.tree import DecisionTreeClassifier

  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

  tree.fit(X_train, Y_train)



  #RandomForest

  from sklearn.ensemble import RandomForestClassifier

  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

  forest.fit(X_train, Y_train)



  #print accuracy

  print('[0]logistic regression trainning accuracy: ', log.score(X_train, Y_train))

  print('[1]knn trainning accuracy: ', knn.score(X_train, Y_train))

  print('[2]svc_lin trainning accuracy: ', svc_lin.score(X_train, Y_train))

  print('[3]gauss trainning accuracy: ', gauss.score(X_train, Y_train))

  print('[4]tree trainning accuracy: ', tree.score(X_train, Y_train))

  print('[5]forest trainning accuracy: ', forest.score(X_train, Y_train))

  return log, knn, svc_lin, gauss, tree, forest

model = models(X, Y)
forest = model[5]

importances = pd.DataFrame({'feature': train.iloc[:, 2:8].columns, 'importance': np.round(forest.feature_importances_, 3)})

importances = importances.sort_values('importance', ascending = False).set_index('feature')

importances

importances.plot.bar()
test.head()
X_test = test.loc[:, 'Pclass':'S'].values

train.head()
pred = model[5].predict(X_test)

print(pred)
submission=pd.DataFrame()

ID = test['PassengerId']

submission['PassengerId'] = ID

submission['Survived'] = pred

submission.shape

submission.head()
submission['Survived'].value_counts()
submission.to_csv('submission.csv', index = False)
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
import matplotlib.pyplot as plt

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline

import cufflinks as cf

cf.go_offline()

from plotly.offline import iplot

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
train= pd.read_csv('/kaggle/input/titanic/train.csv')
train.info()
sns.heatmap(train.isnull(), yticklabels= False, cbar = False)
sns.set_style('whitegrid')

sns.countplot(x ='Survived', data = train)
sns.countplot(x = 'Survived', data = train, hue = 'Sex')
sns.countplot(x = 'Survived', data = train, hue = 'Sex', palette = 'RdBu_r')
sns.countplot(x = 'Survived', data = train, hue = 'Pclass')
sns.distplot(train['Age'].dropna(), kde = False, bins = 30)
train['Age'].plot.hist(bins = 50)
sns.countplot(x = 'SibSp', data= train)
train['Fare'].hist(bins = 50)
train['Fare'].hist(bins = 50, figsize = (12,5))
train['Fare'].iplot(kind = 'hist', bins = 50)
train['Age'].iplot(kind = 'hist', bins = 50)
train['Pclass'].iplot(kind = 'hist', bins = 10)
train['SibSp'].iplot(kind = 'bar', bins = 50)
def impute_age(cols):

  Age = cols[0]

  Pclass = cols[1]



  if pd.isnull(Age):

    

    if Pclass == 1:

      return 37

    elif Pclass == 2:

      return 29

    else:

      return 24

  else:

    return Age
train['Age']= train[['Age', 'Pclass']].apply(impute_age, axis = 1)
sns.heatmap(train.isnull(), yticklabels= False, cbar= False, cmap = 'viridis')
train.drop('Cabin', axis = 1, inplace = True)
train.head()
sns.heatmap(train.isnull(), yticklabels= False, cbar= False, cmap = 'viridis')
train.dropna(inplace = True)
sns.heatmap(train.isnull(), yticklabels= False, cbar= False, cmap = 'viridis')
pd.get_dummies(train['Sex'])
sex = pd.get_dummies(train['Sex'], drop_first= True)
embarked = pd.get_dummies(train['Embarked'], drop_first= True)
train = pd.concat([train, sex, embarked], axis = 1)
train.head()
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis =1, inplace = True)
train.drop(['PassengerId'], axis = 1, inplace=True )
train
X = train.drop('Survived', axis = 1)

Y = train['Survived']
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test, prediction))
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, prediction)
train.corr()
sns.heatmap(train.corr(), cmap = 'coolwarm', annot= True)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)
print(lm.intercept_)
lm.coef_
cdf= pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
cdf
predictions = lm.predict(X_test)
predictions
Y_test
sns.distplot(Y_test - predictions, bins = 100)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, predictions))
print('MSE:', metrics.mean_squared_error(Y_test, predictions))
np.sqrt(metrics.mean_squared_error(Y_test, predictions))
from sklearn.ensemble import RandomForestClassifier



y = train["Survived"]

features = ["Pclass",  "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

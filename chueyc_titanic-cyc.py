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



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



df = pd.read_csv('../input/train.csv')
df.info()

# it looks like there are NULL cells in 'Age' and 'Cabin'
df.head()
df.describe()
sns.distplot(df['Age'].dropna(),bins=30)
df[df['Survived']==1].Age.plot.hist(bins=30)
sns.countplot(x='Survived',hue ='Pclass',data = df)
# somehow can't put a number exactly to the countplot above each bar like in excel.

df.groupby(['Pclass', 'Survived']).size()
sns.countplot(x='Sex',hue = 'Survived', data = df)
g = sns.FacetGrid(df, row = 'Survived', col = 'Sex')

g.map(sns.distplot, 'Age')
df['Cabin'].isnull().sum()

# will drop the entire column as I can't figure out how best to put meaningful replacement entries.
df[df['Embarked'].isnull()]

# will drop these two rows as the impact will be minimal (<0.5%)
df_n = df.drop(['Name','Cabin','Ticket'],axis =1)
df_n[df_n['Embarked'].isnull()]
df_n = df_n.drop(df_n.index[[61, 829]])

df_n.info()

df_n[df_n['Embarked'].isnull()]
df_n = df_n.fillna(df_n.mean())

df_n.info()
df_n_sex = pd.get_dummies(df_n['Sex'])

df_n_embark = pd.get_dummies(df_n['Embarked'])
df_new = pd.concat([df_n,df_n_sex,df_n_embark],axis =1)
#df.sort_values(['Name','Pclass'], ascending =[True,True])
df_new.head()
X = df_new[['Pclass','Age','SibSp','Parch','Fare','female','male','C','Q','S']]

y = df_new['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train_minmax, y_train)
pred_log = logmodel.predict(X_test_minmax)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred_log))

print('\n')

print(confusion_matrix(y_test,pred_log))
# test with no scaling

logmodelasis = LogisticRegression()

logmodelasis.fit(X_train,y_train)
pred_logasis = logmodelasis.predict(X_test)
print(classification_report(y_test,pred_logasis))

print('\n')

print(confusion_matrix(y_test,pred_logasis))
from sklearn.svm import SVC
svcmodel = SVC()
svcmodel.fit = (X_train_minmax, y_train)
from sklearn.grid_search import GridSearchCV
param_grid = {'C': [1,3,10,30,90,300,900,3000],'gamma':  [0.01,0.03,0.1,0.3,1,3,10,30] }
grid = GridSearchCV(SVC(),param_grid,verbose = 3)
grid.fit(X_train_minmax,y_train)
grid.best_params_
grid.best_estimator_
grid_pred = grid.predict(X_test_minmax)
print(classification_report(y_test,grid_pred))

print('\n')

print(confusion_matrix(y_test, grid_pred))
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13),max_iter=500)
mlp.fit(X_test_minmax, y_test)
mlp_pred = mlp.predict(X_test_minmax)
print(classification_report(y_test,mlp_pred))

print('\n')

print(confusion_matrix(y_test, mlp_pred))
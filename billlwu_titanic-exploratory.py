# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows',300)
pd.set_option('display.min_rows',1)
pd.set_option('display.max_columns',300)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load train and test data
df=pd.read_csv('../input/titanic/train.csv')
df.set_index('PassengerId', inplace=True)
df_test=pd.read_csv('../input/titanic/test.csv')
df.head()
plt.rcParams['figure.figsize']=[10,7]
df.hist(bins=50)
# see which field, if any, contains missing values
df_test.isnull().any()
# create features
DF=pd.DataFrame(index=df.index)
DF.loc[:, 'survived'] = df.Survived                                           
DF.loc[:, 'class1']=(df.Pclass==1).astype(int) 
DF.loc[:, 'class2']=(df.Pclass==2).astype(int)
DF.loc[:, 'class3']=(df.Pclass==3).astype(int)
DF.loc[:, 'sex1'] = (df.Sex=='female').astype(int)
DF.loc[:, 'sex1_single'] = df.Name.str.contains('Miss').astype(int)
DF.loc[:, 'sex1_married'] = df.Name.str.contains('Mrs').astype(int)
DF.loc[:, 'sex2'] = (df.Sex=='male').astype(int)
DF.loc[:, 'class1sex1'] = ((df.Pclass==1) & (df.Sex=='female')).astype(int)
DF.loc[:, 'class1sex2'] = ((df.Pclass==1) & (df.Sex=='male')).astype(int)
DF.loc[:, 'class2sex1'] = ((df.Pclass==2) & (df.Sex=='female')).astype(int)
DF.loc[:, 'class2sex2'] = ((df.Pclass==2) & (df.Sex=='male')).astype(int)
DF.loc[:, 'class3sex1'] = ((df.Pclass==3) & (df.Sex=='female')).astype(int)
DF.loc[:, 'class3sex2'] = ((df.Pclass==3) & (df.Sex=='male')).astype(int)
DF.loc[:, 'is_master'] = df.Name.str.contains('Master').astype(int)
DF.loc[:, 'sibSp'] = df.SibSp
DF.loc[:, 'Parch'] = df.Parch
DF.loc[:, 'fare'] = df.Fare
DF.loc[:,'fare1']=np.tanh(DF.fare/30)
DF.loc[:, 'age'] = df.Age.fillna(0)
DF.loc[:,'age1']=np.tanh(DF.age/30)
DF.loc[:, 'age_ischild'] = ((df.Age>0) & (df.Age<10)).astype(int)
DF.loc[:, 'age_iselder'] = (df.Age>50).astype(int)
DF.loc[:,'age_missing'] = df.Age.isnull().astype(int)
DF.loc[:,'emb_C'] = (df.Embarked=='C').astype(int)
DF.loc[:,'emb_Q'] = (df.Embarked=='Q').astype(int)
DF.loc[:,'emb_S'] = (df.Embarked=='S').astype(int)
DF.loc[:,'emb_missing'] = df.Embarked.isnull().astype(int)
DF.loc[:, 'cabin_C'] = (df.Cabin.str[0]=='C').astype(int)
DF.loc[:, 'cabin_E'] = (df.Cabin.str[0]=='E').astype(int)
DF.loc[:, 'cabin_G'] = (df.Cabin.str[0]=='G').astype(int)
DF.loc[:, 'cabin_D'] = (df.Cabin.str[0]=='D').astype(int)
DF.loc[:, 'cabin_A'] = (df.Cabin.str[0]=='A').astype(int)
DF.loc[:, 'cabin_B'] = (df.Cabin.str[0]=='B').astype(int)
DF.loc[:, 'cabin_F'] = (df.Cabin.str[0]=='F').astype(int)
DF.loc[:, 'cabin_T'] = (df.Cabin.str[0]=='T').astype(int)
df[~df.Cabin.isnull()].Cabin.str[0].describe()
df[~df.Cabin.isnull()].Cabin.str[0].unique()
# correlation matrix
corr=DF.corr()
corr.style.background_gradient(cmap='coolwarm')
# try a few transformations to make Fare look more normalized
test=pd.DataFrame()
test.loc[:,'fare']=df.Fare
test.loc[:,'fare1']=np.tanh(test.fare/30)
test.loc[:,'fare2']=np.tanh(test.fare/40)
test.loc[:,'fare3']=np.tanh(test.fare/60)
test.hist(bins=50)
# try a few transformations on Age
test=pd.DataFrame()
test.loc[:,'age']=df[df.Age>0].Age
test.loc[:,'age1']=np.tanh(test.age/30)
test.loc[:,'age2']=np.tanh(test.age/40)
test.hist(bins=50)
# fit logistic regression model via CV, compute in-sample prediction accuracy
clf=LogisticRegressionCV(cv=5,random_state=0,max_iter=1000).fit(DF.iloc[:,1:], DF.iloc[:, 0])
rs=pd.DataFrame(index=DF.index)
rs.loc[:,'pred']=clf.predict(DF.iloc[:,1:])
rs.loc[:,'y']=DF.survived
print('in-sample prediction rate via Logistic Regression CV: {:.2f}'.format((rs.pred==rs.y).astype(int).sum()/rs.shape[0]))
# found a nice tutorial on gridsearching hyperparameters at https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1800, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(DF.iloc[:,1:], DF.iloc[:, 0])
rf_random.best_params_
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 15],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [1600,1800,2000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(DF.iloc[:,1:], DF.iloc[:, 0])
rf_model = grid_search.best_estimator_
rs=pd.DataFrame(index=DF.index)
rs.loc[:,'pred']=rf_model.predict(DF.iloc[:,1:])
rs.loc[:,'y']=DF.survived
print('in-sample prediction rate via Logistic Regression CV: {:.2f}'.format((rs.pred==rs.y).astype(int).sum()/rs.shape[0]))
df2=pd.DataFrame()
df2.loc[:, 'class1']=(df_test.Pclass==1).astype(int) 
df2.loc[:, 'class2']=(df_test.Pclass==2).astype(int)
df2.loc[:, 'class3']=(df_test.Pclass==3).astype(int)
df2.loc[:, 'sex1'] = (df_test.Sex=='female').astype(int)
df2.loc[:, 'sex1_single'] = df_test.Name.str.contains('Miss').astype(int)
df2.loc[:, 'sex1_married'] = df_test.Name.str.contains('Mrs').astype(int)
df2.loc[:, 'sex2'] = (df_test.Sex=='male').astype(int)
df2.loc[:, 'class1sex1'] = ((df_test.Pclass==1) & (df_test.Sex=='female')).astype(int)
df2.loc[:, 'class1sex2'] = ((df_test.Pclass==1) & (df_test.Sex=='male')).astype(int)
df2.loc[:, 'class2sex1'] = ((df_test.Pclass==2) & (df_test.Sex=='female')).astype(int)
df2.loc[:, 'class2sex2'] = ((df_test.Pclass==2) & (df_test.Sex=='male')).astype(int)
df2.loc[:, 'class3sex1'] = ((df_test.Pclass==3) & (df_test.Sex=='female')).astype(int)
df2.loc[:, 'class3sex2'] = ((df_test.Pclass==3) & (df_test.Sex=='male')).astype(int)
df2.loc[:, 'is_master'] = df_test.Name.str.contains('Master').astype(int)
df2.loc[:, 'sibSp'] = df_test.SibSp
df2.loc[:, 'Parch'] = df_test.Parch
df2.loc[:, 'fare'] = df_test.Fare.fillna(0)
df2.loc[:,'fare1']=np.tanh(df2.fare/30)
df2.loc[:, 'age'] = df_test.Age.fillna(0)
df2.loc[:,'age1']=np.tanh(df2.age/30)
df2.loc[:, 'age_ischild'] = ((df_test.Age>0) & (df_test.Age<10)).astype(int)
df2.loc[:, 'age_iselder'] = (df_test.Age>50).astype(int)
df2.loc[:,'age_missing'] = df_test.Age.isnull().astype(int)
df2.loc[:,'emb_C'] = (df_test.Embarked=='C').astype(int)
df2.loc[:,'emb_Q'] = (df_test.Embarked=='Q').astype(int)
df2.loc[:,'emb_S'] = (df_test.Embarked=='S').astype(int)
df2.loc[:,'emb_missing'] = df_test.Embarked.isnull().astype(int)
df2.loc[:, 'cabin_C'] = (df_test.Cabin.str[0]=='C').astype(int)
df2.loc[:, 'cabin_E'] = (df_test.Cabin.str[0]=='E').astype(int)
df2.loc[:, 'cabin_G'] = (df_test.Cabin.str[0]=='G').astype(int)
df2.loc[:, 'cabin_D'] = (df_test.Cabin.str[0]=='D').astype(int)
df2.loc[:, 'cabin_A'] = (df_test.Cabin.str[0]=='A').astype(int)
df2.loc[:, 'cabin_B'] = (df_test.Cabin.str[0]=='B').astype(int)
df2.loc[:, 'cabin_F'] = (df_test.Cabin.str[0]=='F').astype(int)
df2.loc[:, 'cabin_T'] = (df_test.Cabin.str[0]=='T').astype(int)
predictions1 = clf.predict(df2)
predictions2 = rf_model.predict(df2)
output1 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions1})
output2 = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions2})
output2.to_csv('submission2.csv', index=False)
output2.head()

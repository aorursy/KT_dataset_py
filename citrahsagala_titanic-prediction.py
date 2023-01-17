# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.shape , df_test.shape
df_train.head(1)
df_test.head(1)
df_train = df_train.drop(['Name', 'PassengerId', 'Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'], axis=1)
df_train.columns
for i in range(len(df_train.columns)):
    print(df_train.columns[i], df_train[df_train.columns[i]].nunique())
df_train.dtypes
df_train.shape
quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
import seaborn as sns
quantitative
sns.heatmap(df_train[quantitative].corr(), annot=True)
df_train.isnull().sum()
df_train[df_train['Age'].isnull()].head()
df_train[df_train['Age'].isnull()][df_train[df_train['Age'].isnull()]['Pclass']==1].head()
plt.hist(df_train[df_train['Age'].isnull()==False][df_train[df_train['Age'].isnull()==False]['Pclass']==1]['Age'])
df_train_age_null = df_train[df_train['Age'].isnull()]
df_train_age_not_null = df_train[df_train['Age'].isnull()==False]
df_train_age_null.shape, df_train_age_not_null.shape
df_train_age_null[df_train_age_null['Embarked']=='C'][df_train_age_null[df_train_age_null['Embarked']=='C']['Pclass']==1][df_train_age_null[df_train_age_null['Embarked']=='C'][df_train_age_null[df_train_age_null['Embarked']=='C']['Pclass']==1]['Sex']=='female']
df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1][df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1]['Sex']=='female'].describe()
age = df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1][df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1]['Sex']=='female']['Age'].mean()
df_q = df_train_age_null[df_train_age_null['Embarked']=='Q']
df_c = df_train_age_null[df_train_age_null['Embarked']=='C']
df_s = df_train_age_null[df_train_age_null['Embarked']=='S']
df_q.shape, df_c.shape, df_s.shape
df_q1 = df_q[df_q['Pclass']==1]
df_q2 = df_q[df_q['Pclass']==2]
df_q3 = df_q[df_q['Pclass']==3]
df_q1.shape, df_q2.shape, df_q3.shape
df_c1 = df_c[df_c['Pclass']==1]
df_c2 = df_c[df_c['Pclass']==2]
df_c3 = df_c[df_c['Pclass']==3]
df_c1.shape, df_c2.shape, df_c3.shape
df_s1 = df_s[df_s['Pclass']==1]
df_s2 = df_s[df_s['Pclass']==2]
df_s3 = df_s[df_s['Pclass']==3]
df_s1.shape, df_s2.shape, df_s3.shape
df_q2['Age'] = 30.0
df_q2
df_q3_f = df_q3[df_q3['Sex']=='female']
df_q3_m = df_q3[df_q3['Sex']=='male']

df_q3_f.shape, df_q3_m.shape
df_train_age_not_null[df_train_age_not_null['Embarked']=='Q'][df_train_age_not_null[df_train_age_not_null['Embarked']=='Q']['Pclass']==3][df_train_age_not_null[df_train_age_not_null['Embarked']=='Q'][df_train_age_not_null[df_train_age_not_null['Embarked']=='Q']['Pclass']==3]['Sex']=='female']['Age'].mean()
df_q3_f['Age'] = 22.0
df_train_age_not_null[df_train_age_not_null['Embarked']=='Q'][df_train_age_not_null[df_train_age_not_null['Embarked']=='Q']['Pclass']==3][df_train_age_not_null[df_train_age_not_null['Embarked']=='Q'][df_train_age_not_null[df_train_age_not_null['Embarked']=='Q']['Pclass']==3]['Sex']=='male']['Age'].mean()
df_q3_m['Age'] = 28.0
df_q3 = df_q3_m.append(df_q3_f)
df_q3.head(1)
df_q = df_q2.append(df_q3)
df_q.shape
df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1][df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1]['Sex']=='female']['Age'].mean()
df_c1_f = df_c1[df_c1['Sex']=='female']
df_c1_m = df_c1[df_c1['Sex']=='male']
df_c1_f.shape, df_c1_m.shape
df_c1_f['Age'] = 36.0
df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1][df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==1]['Sex']=='male']['Age'].median()
df_c1_m['Age'] = 37.0
df_c1 = df_c1_m.append(df_c1_f)
df_c1.head(1)
df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==2][df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==2]['Sex']=='male']['Age'].median()
df_c2['Age'] = 29.0
df_c3_f = df_c3[df_c3['Sex']=='female']
df_c3_m = df_c3[df_c3['Sex']=='male']
df_c3_f.shape, df_c3_m.shape
df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==3][df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==3]['Sex']=='female']['Age'].mean()
df_c3_f['Age'] = 14.0
df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==3][df_train_age_not_null[df_train_age_not_null['Embarked']=='C'][df_train_age_not_null[df_train_age_not_null['Embarked']=='C']['Pclass']==3]['Sex']=='male']['Age'].mean()
df_c3_m['Age'] = 25.0
df_c3 = df_c3_f.append(df_c3_m)
df_c3.shape
df_c = df_c1.append(df_c2)
df_c = df_c.append(df_c3)
df_c.shape
df_s1_f = df_s1[df_s1['Sex']=='female']
df_s1_m = df_s1[df_s1['Sex']=='male']
df_s1_f.shape, df_s1_m.shape
df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==1][df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==1]['Sex']=='female']['Age'].mean()
df_s1_f['Age'] = 33.0
df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==1][df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==1]['Sex']=='male']['Age'].mean()
df_s1_m['Age'] = 42.0
df_s1 = df_s1_f.append(df_s1_m)
df_s1.head(1)
df_s2_f = df_s2[df_s2['Sex']=='female']
df_s2_m = df_s2[df_s2['Sex']=='male']
df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==2][df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==2]['Sex']=='female']['Age'].mean()
df_s2_f['Age'] = 29.0
df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==2][df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==2]['Sex']=='male']['Age'].mean()
df_s2_m['Age'] = 30.0
df_s2 = df_s2_f.append(df_s2_m)
df_s2.head(1)
df_s3_f = df_s3[df_s3['Sex']=='female']
df_s3_m = df_s3[df_s3['Sex']=='male']
df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==3][df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==3]['Sex']=='female']['Age'].mean()
df_s3_f['Age'] = 23.0
df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==3][df_train_age_not_null[df_train_age_not_null['Embarked']=='S'][df_train_age_not_null[df_train_age_not_null['Embarked']=='S']['Pclass']==3]['Sex']=='male']['Age'].mean()
df_s3_m['Age'] = 26.0
df_s3 = df_s3_f.append(df_s3_m)
df_s3.shape
df_q = df_q1.append(df_q2)
df_q = df_q.append(df_q3)
df_q.shape
df_s = df_s1.append(df_s2)
df_s = df_s.append(df_s3)
df_s.shape
df_c = df_c1.append(df_c2)
df_c = df_c.append(df_c3)
df_c.shape
df_train_age_null = df_q.append(df_c)
df_train_age_null = df_train_age_null.append(df_s)
df_train_age_null.shape
df_train = df_train_age_null.append(df_train_age_not_null)
df_train.shape
df_train_c = df_train[df_train['Cabin'].isnull()==False]
df_train_nc = df_train[df_train['Cabin'].isnull()]
df_train_c.shape, df_train_nc.shape
df_train_nc['Cabin'] = 0
df_train_c['Cabin'] = 1
df_train = df_train_nc.append(df_train_c)
df_train.shape
df_train.head(1)
# Label Encode Sex
mapping = { 'male': 1, 'female': 0}
df_train['Sex'] = df_train['Sex'].map(mapping)
df_train.head(1)
# One Hot Encoding : Change Embarked data into binary
ohe = pd.get_dummies(df_train[['Embarked']])
df_train = pd.concat([df_train, ohe], axis=1, join='inner')
df_train = df_train.drop(['Embarked'], axis=1)
df_train.head()
df_train[df_train['Survived'].isnull()==True].shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']
# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# make model
clf = RandomForestClassifier(n_estimators=2000, random_state=42)

# train model
clf.fit(X_train, y_train)
# predict test from train data
y_pred = clf.predict(X_test)

# print accuracy
print(clf.score(X_test, y_test))
from sklearn.metrics import confusion_matrix
from sklearn import metrics
confusion_matrix(y_test, y_pred)
y = y_test.values.ravel()
pred = np.array(y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
df_test_age_null = df_test[df_test['Age'].isnull()]
df_test_age_not_null = df_test[df_test['Age'].isnull()==False]
df_q = df_test_age_null[df_test_age_null['Embarked']=='Q']
df_c = df_test_age_null[df_test_age_null['Embarked']=='C']
df_s = df_test_age_null[df_test_age_null['Embarked']=='S']
df_q.shape, df_c.shape, df_s.shape
df_q1 = df_q[df_q['Pclass']==1]
df_q2 = df_q[df_q['Pclass']==2]
df_q3 = df_q[df_q['Pclass']==3]
df_q1.shape, df_q2.shape, df_q3.shape
df_c1 = df_c[df_c['Pclass']==1]
df_c2 = df_c[df_c['Pclass']==2]
df_c3 = df_c[df_c['Pclass']==3]
df_c1.shape, df_c2.shape, df_c3.shape
df_s1 = df_s[df_s['Pclass']==1]
df_s2 = df_s[df_s['Pclass']==2]
df_s3 = df_s[df_s['Pclass']==3]
df_s1.shape, df_s2.shape, df_s3.shape
df_q2['Age'] = 30.0
df_q3_f = df_q3[df_q3['Sex']=='female']
df_q3_m = df_q3[df_q3['Sex']=='male']
df_q3_f['Age'] = 22.0
df_q3_m['Age'] = 28.0
df_q3 = df_q3_m.append(df_q3_f)
df_q3.shape
df_q = df_q2.append(df_q3)
df_q.shape
df_c1_f = df_c1[df_c1['Sex']=='female']
df_c1_m = df_c1[df_c1['Sex']=='male']
df_c1_f.shape, df_c1_m.shape
df_c1_f['Age'] = 36.0
df_c1_m['Age'] = 37.0
df_c1 = df_c1_m.append(df_c1_f)
df_c1.head(1)
df_c2['Age'] = 29.0
df_c3_f = df_c3[df_c3['Sex']=='female']
df_c3_m = df_c3[df_c3['Sex']=='male']
df_c3_f.shape, df_c3_m.shape
df_c3_f['Age'] = 14.0
df_c3_m['Age'] = 25.0
df_c3 = df_c3_f.append(df_c3_m)
df_c3.shape
df_s1_f = df_s1[df_s1['Sex']=='female']
df_s1_m = df_s1[df_s1['Sex']=='male']
df_s1_f.shape, df_s1_m.shape
df_s1_f['Age'] = 33.0
df_s1_m['Age'] = 42.0
df_s1 = df_s1_f.append(df_s1_m)
df_s1.head(1)
df_s2_f = df_s2[df_s2['Sex']=='female']
df_s2_m = df_s2[df_s2['Sex']=='male']
df_s2_f['Age'] = 29.0
df_s2_m['Age'] = 30.0
df_s2 = df_s2_f.append(df_s2_m)
df_s2.head(1)
df_s3_f = df_s3[df_s3['Sex']=='female']
df_s3_m = df_s3[df_s3['Sex']=='male']
df_s3_f['Age'] = 23.0
df_s3_m['Age'] = 26.0
df_s3 = df_s3_f.append(df_s3_m)
df_s3.shape
df_q = df_q1.append(df_q2)
df_q = df_q.append(df_q3)
df_q.shape
df_s = df_s1.append(df_s2)
df_s = df_s.append(df_s3)
df_s.shape
df_c = df_c1.append(df_c2)
df_c = df_c.append(df_c3)
df_c.shape
df_test_age_null = df_q.append(df_c)
df_test_age_null = df_test_age_null.append(df_s)
df_test_age_null.shape
df_test = df_test_age_null.append(df_test_age_not_null)
df_test.shape
df_test_c = df_test[df_test['Cabin'].isnull()==False]
df_test_nc = df_test[df_test['Cabin'].isnull()]
df_test_c.shape, df_test_nc.shape
df_test_nc['Cabin'] = 0
df_test_c['Cabin'] = 1
df_test = df_test_nc.append(df_test_c)
df_test.shape
mapping = { 'male': 1, 'female': 0}
df_test['Sex'] = df_test['Sex'].map(mapping)
df_test.head(1)
ohe = pd.get_dummies(df_test[['Embarked']])
df_test = pd.concat([df_test, ohe], axis=1, join='inner')
df_test = df_test.drop(['Embarked'], axis=1)
df_test.head()
df_test.isnull().sum()
df_test.columns
df_test.Fare.mean()
df_test['Fare'] = df_test.Fare.astype(float).fillna(df_test.Fare.mean())
t_pred = clf.predict(df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S']])
t_pred
df_pred = pd.DataFrame(df_test['PassengerId'])
df_pred.head()
df_pred['Survived'] = pd.DataFrame(t_pred)
df_pred.head()
# df_pred.to_csv('submission.csv', index=False)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [300, 500, 1000, 1500, 2000]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
grid_search.best_params_
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_train, y_train)
# predict test from train data
grid_pred = best_grid.predict(X_test)

# print accuracy
print(best_grid.score(X_test, y_test))
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
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
print(random_grid)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
# predict test from train data
random_pred = best_random.predict(X_test)

# print accuracy
print(best_random.score(X_test, y_test))

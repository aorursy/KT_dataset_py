# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
submission = pd.read_csv('../input/gender_submission.csv')
submission.head()
# submission.to_csv('my_submission.csv', index=None)
train = pd.read_csv('../input/train.csv')
train.head()
num_cols = train.columns[train.dtypes != 'object']
cat_cols = train.columns[train.dtypes == 'object']
train[num_cols].describe()
train[cat_cols].describe()
print(len(train))
train.isnull().sum()
sns.heatmap(train.corr())
for col in num_cols:
    train[col].hist()
    plt.show()
train.corr()['Survived'].sort_values()
train['Embarked'].value_counts().plot.bar()
train['Cabin'].unique()
train['Age'].hist()
post = train.copy()
post['Age'] = post['Age'].fillna(post['Age'].median())
post['Embarked'] = post['Embarked'].fillna('-')
post['Cabin'] = post['Cabin'].fillna('-')
# post = post[post.columns[post.dtypes != 'object']]
post['Cabin'] = post['Cabin'].str[0]
post = pd.get_dummies(post, columns=['Cabin', 'Sex', 'Embarked'], drop_first=True)
post = post.drop(columns=['Name', 'Ticket'])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
X = post.drop(columns=['Survived'])
y = post['Survived']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
train_X_norm = scaler.fit_transform(train_X)
train_X_norm.shape
pca = PCA(n_components=10)
train_X_pca = pca.fit_transform(train_X_norm)
train_X_pca.shape
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
rf = RandomForestClassifier()
svm = SVC()
train_y.shape
lr.fit(train_X_pca, train_y)
rf.fit(train_X_pca, train_y)
svm.fit(train_X_pca, train_y)
from sklearn.metrics import accuracy_score
test_X_norm = scaler.transform(test_X)
test_X_pca = pca.transform(test_X_norm)

pred_y_lr = lr.predict(test_X_pca)
pred_y_rf = rf.predict(test_X_pca)
pred_y_svm = svm.predict(test_X_pca)
acc_lr = accuracy_score(pred_y_lr, test_y)
acc_rf = accuracy_score(pred_y_rf, test_y)
acc_svm = accuracy_score(pred_y_svm, test_y)

print('Logistic Regression :', acc_lr)
print('Random Forest :', acc_rf)
print('Support Vector Machine :', acc_svm)
steps = [
    ('scaler',StandardScaler()),
    ('pca',PCA()),
    ('svm',SVC())
]
pipe = Pipeline(steps)
param_grid = {
    'pca__n_components':[2,5,10],
    'svm__C':np.logspace(-1,3,4),
    'svm__kernel':['rbf','linear', 'poly']
}
grid = GridSearchCV(pipe, param_grid, scoring='accuracy', iid=True, cv=5)
grid.fit(train_X, train_y)
model = grid.best_estimator_
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
pred = model.predict(test_X)

acc = accuracy_score(test_y, pred)
conf_mat = confusion_matrix(test_y, pred)
print('accuracy :', acc)
print(conf_mat)
test = pd.read_csv('../input/test.csv')
test.isnull().sum()
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Age'] = test['Age'].fillna(train['Age'].median())
test['Embarked'] = test['Embarked'].fillna('-')
test['Cabin'] = test['Cabin'].fillna('-')

test['Cabin'] = test['Cabin'].str[0]

test = pd.get_dummies(test, columns=['Cabin', 'Sex', 'Embarked'], drop_first=True)

test.dtypes

test['Cabin_T'] = 0
test['Embarked_C'] = 0
test.dtypes
test = test.drop(columns=['Name', 'Ticket'])

test = test[post.drop(columns=['Survived']).columns]

pred_test = model.predict(test)
pred_test.shape
submission.shape
submission.columns
sub_test = pd.DataFrame(pred_test)
sub_test.columns = ['Survived']
sub_test['PassengerId'] = submission['PassengerId']
sub_test = sub_test[submission.columns]
sub_test
sub_test.to_csv('my_submission.csv', index=None)
with open('../input/gender_submission.csv', 'r') as f:
    csv = f.read()
csv.splitlines()[0:5]
with open('my_submission.csv', 'r') as f:
    csv = f.read()
csv.splitlines()[0:5]

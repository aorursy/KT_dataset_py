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

ds = pd.read_csv('../input/hmeq-data/hmeq.csv')
ds
df = ds[ds['VALUE'].isnull()]
df = df[df['JOB'].isnull()]
df = df[df['REASON'].isnull()]
df = df[df['DEROG'].isnull()]
df = df[df['DELINQ'].isnull()]
df = df.reset_index()
df = df['index']
df
ds = ds.drop(df)
ds = ds.reset_index()
ds = ds.drop(['index'], axis = 1)

ds.info()
ds.describe()
ds.corr()['BAD'].sort_values()
ds['MORTDUE'].mean()
plt.plot(ds['MORTDUE'])
ds['MORTDUE'] = ds['MORTDUE'].fillna(ds['MORTDUE'].mean())
plt.hist(ds['MORTDUE'])
ds['MORTDUE'] = np.log(ds['MORTDUE'])
ds['VALUE'].mean()
plt.plot(ds['VALUE'])
ds['VALUE'] = ds['VALUE'].fillna(ds['VALUE'].mean())
plt.hist(ds['VALUE'])
ds['VALUE'] = np.log(ds['VALUE'])
ds['REASON'] = ds['REASON'].fillna('DebtCon')
ds['REASON'].unique()
plt.hist(ds['REASON'])
ds['REASON'] = ds['REASON'].replace(['HomeImp', 'DebtCon'], [0, 1])
ds['JOB'].unique()
ds['JOB'].mode()[0]
ds['JOB'] = ds['JOB'].fillna(ds['JOB'].mode()[0])
plt.hist(ds['JOB'])
ds['JOB'] = ds['JOB'].replace(['Other', 'Sales', 'Office', 'Mgr', 'ProfExe', 'Self'], [0, 1, 2, 3, 4, 5])
ds['YOJ'].mean()
ds['YOJ'] = ds['YOJ'].fillna(ds['YOJ'].mean())
plt.plot(ds['YOJ'])
plt.hist(ds['YOJ'])
ds['YOJ'] = np.log(ds['YOJ']+1)
ds['DEROG'].std()
ds['DEROG'] = ds['DEROG'].fillna(ds['DEROG'].std())
plt.plot(ds['DEROG'])
plt.hist(ds['DEROG'])
ds['DELINQ'].std()
ds['DELINQ'] = ds['DELINQ'].fillna(ds['DELINQ'].std())
plt.plot(ds['DELINQ'])
plt.hist(ds['DELINQ'])
ds['CLAGE']
ds['CLAGE'].mean()
ds['CLAGE'] = ds['CLAGE'].fillna(ds['CLAGE'].mean())
plt.plot(ds['CLAGE'])
plt.hist(ds['CLAGE'])
ds['NINQ'].std()
ds['NINQ'] = ds['NINQ'].fillna(ds['NINQ'].std())
plt.plot(ds['NINQ'])
plt.hist(ds['NINQ'])
ds['DEBTINC'].mean()
ds['DEBTINC'] = ds['DEBTINC'].fillna(ds['DEBTINC'].mean())
plt.plot(ds['DEBTINC'])
plt.hist(ds['DEBTINC'])
ds.info()
ds = ds.drop('CLNO', axis = 1)
ds
y = ds['BAD']
X = ds.drop(['BAD'], axis = 1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

acc_dict = {}

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
        
    if name in acc_dict:
        acc_dict[name] += acc
    else:
        acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf]
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns = log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x = 'Accuracy', y = 'Classifier', data = log, color = "b")
acc_dict
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

print('score=',rfc.score(X_test, y_test))
y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
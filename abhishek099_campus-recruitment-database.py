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
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data

data.isnull().sum()
data = data.drop(['sl_no', 'salary'], axis = 1)
data['status'] = np.where(data['status'] == 'Placed',1,0)
data['workex'] = np.where(data['workex'] == 'Yes',1,0)
data['gender'] = np.where(data['gender'] == 'M',1,0)
data['ssc_b'] = np.where(data['ssc_b'] == 'Central',1,0)
data['hsc_b'] = np.where(data['hsc_b'] == 'Central',1,0)
data['degree_t'] = np.where(data['degree_t'] == 'Sci&Tech',1,0)
data['specialisation'] = np.where(data['specialisation'] == 'Mkt&HR',1,0)
data.head(10)
mapping = {'Commerce': 1, 'Science': 2, 'Arts': 3}
data['hsc_s'] = data['hsc_s'].map(mapping)
data.head(10)


y = data['status'].copy()
x = data.drop('status', axis = 1).copy()
x.shape, y.shape
import seaborn as sns
ax = sns.countplot(y, label = 'Count')
y.value_counts()
x.describe()
y.describe()
import matplotlib.pyplot as plt
data_dia = y
data = x
data_n_2 = (data-data.mean())/data.std()
data = pd.concat([y, data_n_2], axis = 1)
data = pd.melt(data, id_vars = 'status', var_name = 'features', value_name = 'value')

plt.figure(figsize = (12,12))
sns.violinplot(x = 'features', y = 'value', hue = 'status', data = data, split = True, inner = 'quart' )
plt.xticks(rotation = 90)
import matplotlib.pyplot as plt
data_dia = y
data = x
data_n_2 = (data-data.mean())/data.std()
data = pd.concat([y, data_n_2], axis = 1)
data = pd.melt(data, id_vars = 'status', var_name = 'features', value_name = 'value')

plt.figure(figsize = (12,12))
sns.boxplot(x = 'features', y = 'value', hue = 'status', data = data )
plt.xticks(rotation = 90)
import matplotlib.pyplot as plt
data_dia = y
data = x
data_n_2 = (data-data.mean())/data.std()
data = pd.concat([y, data_n_2], axis = 1)
data = pd.melt(data, id_vars = 'status', var_name = 'features', value_name = 'value')

plt.figure(figsize = (12,12))
sns.swarmplot(x = 'features', y = 'value', hue = 'status', data = data )
plt.xticks(rotation = 90)
f, ax = plt.subplots(figsize = (12,12))
sns.heatmap(x.corr(),annot = True, linewidth = 0.5, fmt = '.1f', ax = ax)
X = x[['mba_p', 'workex', 'hsc_p', 'ssc_b', 'specialisation']].copy()
X.shape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

model = LogisticRegression().fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(acc)
cm = confusion_matrix(y_test, model.predict(X_test))
cm
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier().fit(X_train, y_train)
accuracy_score(y_test, rfc.predict(X_test))
from sklearn.svm import SVC
svc = SVC().fit(X_train, y_train)
accuracy_score(y_test, svc.predict(X_test))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select_feature = SelectKBest(chi2, k = 5).fit(X_train, y_train)

print('Score list: ', select_feature.scores_)
print('Feature list: ', X_train.columns)


lr = LogisticRegression().fit(X_train,y_train)
accuracy_score(y_test, lr.predict(X_test))


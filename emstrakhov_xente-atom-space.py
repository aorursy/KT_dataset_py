# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import confusion_matrix, f1_score
df = pd.read_csv('../input/xente-challenge/training.csv')

df.head()
test_df = pd.read_csv('../input/xente-challenge/test.csv')

test_df.head()
df['FraudResult'].value_counts()
test_df['PricingStrategy'].value_counts()
test_df['ProductCategory'].value_counts()
# df['tr_id'] = df['TransactionId'].map(lambda s: s.split('_')[-1]).astype(int)

# df.head()
# df['TransactionId'].value_counts()
for name in df.columns:

    if 'Id' in name:

        df[name[:4]+'_id'] = df[name].map(lambda s: s.split('_')[-1]).astype(int)



df.head()
for name in test_df.columns:

    if 'Id' in name:

        test_df[name[:4]+'_id'] = test_df[name].map(lambda s: s.split('_')[-1]).astype(int)



test_df.head()
df_1 = df.drop([name for name in df.columns if 'Id' in name], axis=1)

test_df_1 = test_df.drop([name for name in test_df.columns if 'Id' in name], axis=1)

df_1.head()
df['CurrencyCode'].value_counts()
test_df['CurrencyCode'].value_counts()
df['CountryCode'].value_counts()
test_df['CountryCode'].value_counts()
df_2 = df_1.drop(['CurrencyCode', 'CountryCode'], axis=1)

test_df_2 = test_df_1.drop(['CurrencyCode', 'CountryCode'], axis=1)



df_2.head()
df['ProductCategory'].value_counts()
test_df['ProductCategory'].value_counts()
set(df['ProductCategory'].value_counts().index) ^ set(test_df['ProductCategory'].value_counts().index)
df_3 = pd.get_dummies(df_2, columns=['ProductCategory', 'PricingStrategy'])

test_df_3 = pd.get_dummies(test_df_2, columns=['ProductCategory', 'PricingStrategy'])

df_3.head()
df_3['ProductCategory_retail'] = 0

test_df_3['ProductCategory_other'] = 0
df_3['amount is positive'] = (df_3['Amount'] > 0).astype(int)

df_3['amount != value'] = (np.abs(df_3['Amount']) != np.abs(df_3['Value'])).astype(int)



test_df_3['amount is positive'] = (test_df_3['Amount'] > 0).astype(int)

test_df_3['amount != value'] = (np.abs(test_df_3['Amount']) != np.abs(test_df_3['Value'])).astype(int)



df_3.head()
df_4 = df_3.drop(['Amount'], axis=1)

test_df_4 = test_df_3.drop(['Amount'], axis=1)

df_4.head()
df_4['time'] = pd.to_datetime(df_4['TransactionStartTime'])

test_df_4['time'] = pd.to_datetime(test_df_4['TransactionStartTime'])
df_4.head()
df_4.info()
df_5 = df_4.drop('TransactionStartTime', axis=1)

test_df_5 = test_df_4.drop('TransactionStartTime', axis=1)

df_5.head()
df['PricingStrategy'].value_counts()
df_5['year'] = df_5['time'].map(lambda x: x.year).astype(int)

df_5['month'] = df_5['time'].map(lambda x: x.month).astype(int)

df_5['day'] = df_5['time'].map(lambda x: x.day).astype(int)

df_5['hour'] = df_5['time'].map(lambda x: x.hour).astype(int)

df_5['minute'] = df_5['time'].map(lambda x: x.minute).astype(int)



df_5.head()
test_df_5['year'] = test_df_5['time'].map(lambda x: x.year).astype(int)

test_df_5['month'] = test_df_5['time'].map(lambda x: x.month).astype(int)

test_df_5['day'] = test_df_5['time'].map(lambda x: x.day).astype(int)

test_df_5['hour'] = test_df_5['time'].map(lambda x: x.hour).astype(int)

test_df_5['minute'] = test_df_5['time'].map(lambda x: x.minute).astype(int)
X = df_5.drop(['time', 'FraudResult'], axis=1)

y = df_5['FraudResult']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=19)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=19)

tree.fit(X_train, y_train)
# картинка

from sklearn.tree import export_graphviz

tree_dot = export_graphviz(tree)

print(tree_dot)
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': range(2, 11)}



tree_grid = GridSearchCV(tree, tree_params,

                         cv=5, n_jobs=-1, verbose=True, scoring='f1')



tree_grid.fit(X_train, y_train)
tree_grid.best_params_
best_tree = tree_grid.best_estimator_
pd.DataFrame(tree_grid.cv_results_).T
import matplotlib.pyplot as plt



df_cv = pd.DataFrame(tree_grid.cv_results_)



plt.plot(df_cv['param_max_depth'], df_cv['mean_test_score'])

plt.xlabel("max_depth")

plt.ylabel("f1_score");
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, 

                            class_weight='balanced',

                            random_state=19)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)

confusion_matrix(y_valid, y_pred)
# rf_params = {'max_depth': range(2, 11), 'criterion': ['gini', 'entropy'], 

#              'class_weight': ['balanced', 'balanced_subsample']}



# rf_params_1 = {'max_depth': range(5, 51, 5)}



# rf_grid = GridSearchCV(rf, rf_params_1,

#                        cv=5, n_jobs=-1, verbose=True, scoring='f1')



# rf_grid.fit(X_train, y_train)
# rf_grid.best_params_
# df_cv = pd.DataFrame(rf_grid.cv_results_)



# plt.plot(df_cv['param_max_depth'], df_cv['mean_test_score'])

# plt.xlabel("max_depth")

# plt.ylabel("f1_score");
# best_rf = rf_grid.best_estimator_

# y_pred = best_rf.predict(X_valid)



# confusion_matrix(y_valid, y_pred)
k = 0

for i in range(len(y_valid)):

    if y_valid.values[i] == 1 and y_pred[i] == 1:

        k += 1

print(k, sum(y_valid))
confusion_matrix(y_valid, y_pred)
from sklearn.metrics import f1_score

f1_score(y_valid, y_pred)
test_df_5.head()
X_test = test_df_5.drop('time', axis=1)

y_test_pred = best_rf.predict(X_test)
submit = pd.read_csv('../input/xente-challenge/sample_submission.csv')

submit.head()
X_test.head()
submit['FraudResult'] = y_test_pred

submit.head()
np.sum(y_test_pred)
submit.to_csv('xente_submit.csv', index=False)
import seaborn as sns

sns.boxplot(df['Value']);
sns.boxplot(test_df['Value']);
fraud_df = df[df['FraudResult']==1]

sns.boxplot(fraud_df['Value']);
fraud_df['Value'].describe()
sns.distplot(fraud_df['Value']);
fraud_df = df_5[df_5['FraudResult']==1]

fraud_df['hour']
sns.countplot(data=df_5, x='hour', hue='FraudResult');
fig, ax = plt.subplots(2, 1)

sns.boxplot(df_5[df_5['FraudResult']==0]['hour'], ax=ax[0]);

sns.boxplot(df_5[df_5['FraudResult']==1]['hour'], ax=ax[1]);
fig, ax = plt.subplots(2, 1)

sns.boxplot(df_5[df_5['FraudResult']==0]['month'], ax=ax[0]);

sns.boxplot(df_5[df_5['FraudResult']==1]['month'], ax=ax[1]);
df_5[df_5['FraudResult']==0]['month'].describe()
df_5[df_5['FraudResult']==1]['month'].describe()
df_5[df_5['month']==2]['FraudResult'].value_counts()
sns.countplot(data=df_5, y='month', hue='FraudResult');
df_5.groupby(['month', 'FraudResult'])['Tran_id'].count()
test_df_5['month'].value_counts()
df.groupby(['ProductCategory', 'FraudResult'])['TransactionId'].count()
df.groupby(['PricingStrategy', 'FraudResult'])['TransactionId'].count()
df_5.head()
test_df_5.head()
df_5['TARGET'] = 1

test_df_5['TARGET'] = 0



data = pd.concat([df_5.drop(['FraudResult', 'time'], axis=1), test_df_5.drop('time', axis=1)])

data.head()
X = data.drop('TARGET', axis=1)

y = data['TARGET']



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=19)



rf10 = RandomForestClassifier(n_estimators=10, random_state=19)

rf100 = RandomForestClassifier(n_estimators=100, random_state=19)

rf10.fit(X_train, y_train)

rf100.fit(X_train, y_train)



y_pred10 = rf10.predict_proba(X_valid)[:, 1]

y_pred100 = rf100.predict_proba(X_valid)[:, 1]



from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_valid, y_pred10), roc_auc_score(y_valid, y_pred100))
adv_tree = DecisionTreeClassifier(max_depth=4, random_state=19)

adv_tree.fit(X_train, y_train)



# картинка

from sklearn.tree import export_graphviz

tree_dot = export_graphviz(adv_tree)

print(tree_dot)
X.shape
print(X.columns)

print(np.round(rf100.feature_importances_, 2))
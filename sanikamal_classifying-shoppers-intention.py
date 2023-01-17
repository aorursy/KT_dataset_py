import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv')

df.head()
df.describe(include='all')
# Check for null values in data

nullcount = df.isnull().sum()

print('Total number of null values in dataset:', nullcount.sum())
df.shape
# Visualize the data

sns.countplot(df['Revenue'])

plt.ylim(0,12000)

plt.title('Was the transaction completed?', fontsize= 15)

plt.xlabel('Transaction Completed', fontsize=12)

plt.ylabel('Count (Entries)', fontsize=12)

plt.text(x=-.175, y=11000 ,s='10,422', fontsize=15)

plt.text(x=.875, y=2500, s='1908', fontsize=15)

plt.show()
df.select_dtypes(include=['int64', 'float64']).hist(figsize=(16,22))
feats = df.drop('Revenue', axis=1)

target = df['Revenue']
print(f'Features table has {feats.shape[0]} rows and {feats.shape[1]} columns')

print(f'Target table has {target.shape[0]} rows')
# Checking for number of unique values for each feature

uniques = feats.nunique(axis=0)

print(uniques)
feats['Weekend'].value_counts()
feats['Weekend'].value_counts().plot(kind='bar')
feats['is_weekend'] = feats['Weekend'].apply(lambda row: 1 if row == True else 0)
feats[['Weekend','is_weekend']].tail()
feats.drop('Weekend', axis=1, inplace=True) 
feats['VisitorType'].value_counts()
feats['VisitorType'].value_counts().plot(kind='bar')
colname = 'VisitorType'

visitor_type_dummies = pd.get_dummies(feats[colname], prefix=colname)

pd.concat([feats[colname], visitor_type_dummies], axis=1).tail(n=10)
visitor_type_dummies.drop('VisitorType_Other', axis=1, inplace=True)

visitor_type_dummies.head()
feats = pd.concat([feats, visitor_type_dummies], axis=1)

feats.drop('VisitorType', axis=1, inplace=True) 
feats['Month'].value_counts()
colname = 'Month'

month_dummies = pd.get_dummies(feats[colname], prefix=colname)

month_dummies.drop(colname+'_Feb', axis=1, inplace=True)

feats = pd.concat([feats, month_dummies], axis=1)

feats.drop('Month', axis=1, inplace=True) 
feats.iloc[0]
feats.dtypes
target= target.apply(lambda row: 1 if row==True else 0)

target.head(n=10)
feats['OperatingSystems'].value_counts()
colname = 'OperatingSystems'

operation_system_dummies = pd.get_dummies(feats[colname], prefix=colname)

operation_system_dummies.drop(colname+'_5', axis=1, inplace=True)

feats = pd.concat([feats, operation_system_dummies], axis=1)
feats['Browser'].value_counts()
colname = 'Browser'

browser_dummies = pd.get_dummies(feats[colname], prefix=colname)

browser_dummies.drop(colname+'_9', axis=1, inplace=True)

feats = pd.concat([feats, browser_dummies], axis=1)
feats['TrafficType'].value_counts()
colname = 'TrafficType'

traffic_dummies = pd.get_dummies(feats[colname], prefix=colname)

traffic_dummies.drop(colname+'_17', axis=1, inplace=True)

feats = pd.concat([feats, traffic_dummies], axis=1)
feats['Region'].value_counts()
colname = 'Region'

region_dummies = pd.get_dummies(feats[colname], prefix=colname)

region_dummies.drop(colname+'_5', axis=1, inplace=True)

feats = pd.concat([feats, region_dummies], axis=1)
drop_cols = ['OperatingSystems', 'Browser', 'TrafficType', 'Region']

feats.drop(drop_cols, inplace=True, axis=1)
feats.dtypes
# caluclate the proportion of each target value

target.value_counts()/target.shape[0]*100
y_baseline = pd.Series(data=[0]*target.shape[0])
precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_pred=y_baseline, y_true=target, average='macro', zero_division=1)

print(f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nfscore: {fscore:.4f}')
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)
print(f'Shape of X_train: {X_train.shape}')

print(f'Shape of y_train: {y_train.shape}')

print(f'Shape of X_test: {X_test.shape}')

print(f'Shape of y_test: {y_test.shape}')
model = LogisticRegression(max_iter=10000,random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)

print(f'Accuracy of the model is {accuracy*100:.4f}%')
precision, recall, fscore, _ = metrics.precision_recall_fscore_support(y_pred=y_pred, y_true=y_test, average='binary')

print(f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nfscore: {fscore:.4f}')
coef_list = [f'{feature}: {coef}' for coef, feature in sorted(zip(model.coef_[0], X_train.columns.values.tolist()))]

for item in coef_list:

    print(item)
from sklearn.linear_model import LogisticRegressionCV

# help(LogisticRegressionCV)

Cs = np.logspace(-2, 6, 9)

model_l1 = LogisticRegressionCV(Cs=Cs, penalty='l1', cv=10, solver='liblinear', random_state=42, max_iter=10000)

model_l2 = LogisticRegressionCV(Cs=Cs, penalty='l2', cv=10, random_state=42, max_iter=10000)



model_l1.fit(X_train, y_train)

model_l2.fit(X_train, y_train)
# best hyperparameters

print(f'Best hyperparameter for l1 regularization model: {model_l1.C_[0]}')

print(f'Best hyperparameter for l2 regularization model: {model_l2.C_[0]}')
y_pred_l1 = model_l1.predict(X_test)

y_pred_l2 = model_l2.predict(X_test)
accuracy_l1 = metrics.accuracy_score(y_pred=y_pred_l1, y_true=y_test)

accuracy_l2 = metrics.accuracy_score(y_pred=y_pred_l2, y_true=y_test)

print(f'Accuracy of the model with l1 regularization is {accuracy_l1*100:.4f}%')

print(f'Accuracy of the model with l2 regularization is {accuracy_l2*100:.4f}%')
precision_l1, recall_l1, fscore_l1, _ = metrics.precision_recall_fscore_support(y_pred=y_pred_l1, y_true=y_test, average='binary')

precision_l2, recall_l2, fscore_l2, _ = metrics.precision_recall_fscore_support(y_pred=y_pred_l2, y_true=y_test, average='binary')

print(f'l1\nPrecision: {precision_l1:.4f}\nRecall: {recall_l1:.4f}\nfscore: {fscore_l1:.4f}\n\n')

print(f'l2\nPrecision: {precision_l2:.4f}\nRecall: {recall_l2:.4f}\nfscore: {fscore_l2:.4f}')
coef_list = [f'{feature}: {coef}' for coef, feature in sorted(zip(model_l1.coef_[0], X_train.columns.values.tolist()))]

for item in coef_list:

    print(item)
coef_list = [f'{feature}: {coef}' for coef, feature in sorted(zip(model_l2.coef_[0], X_train.columns.values.tolist()))]

for item in coef_list:

    print(item)
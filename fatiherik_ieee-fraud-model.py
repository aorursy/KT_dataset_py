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
final_data=pd.read_csv('/kaggle/input/ieeefraudfinaldata/final_data.csv')

final_data['TransactionDate']=final_data['TransactionDate'].astype('datetime64')
final_data['_Weekdays'] = final_data['TransactionDate'].dt.dayofweek
final_data['_Hours'] = final_data['TransactionDate'].dt.hour
final_data['_Days'] = final_data['TransactionDate'].dt.day
final_data.info()
df_train, df_test = final_data[final_data['isFraud'] != 2], final_data[final_data['isFraud'] == 2].drop('isFraud', axis=1)
df_train=df_train.drop(['Unnamed: 0'], axis=1)

df_test=df_test.drop(['Unnamed: 0'], axis=1)
df_train.info()
df_test.info()
df_train.head()
df_train.tail()
# df_train.columns.to_list()
df_test.head()
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
X_train = df_train.drop(['isFraud','TransactionDate'], axis=1)
y_train = df_train['isFraud']
model = XGBClassifier(learning_rate=0.1, max_depth=3, min_samples_split=2, n_estimators=100, subsample=0.6)
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_train)


# accuracy = accuracy_score(y_train, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


scores = cross_val_score(model, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())


# feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,cols)), columns=['Value','Feature'])
# plt.figure(figsize=(20, 10))
# sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:20])
# plt.title('XGBClassifier Feature importances')
# plt.tight_layout()
# plt.show()

# x=gc.collect()
sub_data=pd.read_csv('/kaggle/input/sub-data/sub.csv')
sub_data
sub_data['Results']=sub_data['isFraud'].apply(lambda x: 0 if x<0.5 else 1)
sub_data
sub_data=sub_data.drop(['isFraud'], axis=1)
sub_data
df_test=pd.merge(left=df_test, right=sub_data, on='TransactionID', how='left')
df_test
X_test = df_test.drop(['Results','TransactionDate'], axis=1)
y_test = df_test['Results']
model = XGBClassifier(learning_rate=0.1, max_depth=3, min_samples_split=2, n_estimators=100, subsample=0.6)
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# scores = cross_val_score(model, X_test, y_pred, cv=5)
# print("Mean cross-validation score: %.2f" % scores.mean())


# feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,cols)), columns=['Value','Feature'])
# plt.figure(figsize=(20, 10))
# sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:20])
# plt.title('XGBClassifier Feature importances')
# plt.tight_layout()
# plt.show()

# x=gc.collect()
print(accuracy*0.94)
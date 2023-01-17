!unzip /content/CreditCard_Fraud.zip
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns



%matplotlib inline
train_df = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')

train_df.head()
print("Total rows: ", train_df.shape[0])

print("Total columns: ", len(train_df.columns))
train_df.describe()
train_df.isnull().sum()
sns.countplot(x='default.payment.next.month', data=train_df)
plt.figure(figsize = (14,6))

plt.title('Credit Limit - Density Plot')

sns.set_color_codes("pastel")

sns.distplot(train_df[train_df['default.payment.next.month']==1]['LIMIT_BAL'],kde=True,bins=200, color="red")

sns.distplot(train_df[train_df['default.payment.next.month']==0]['LIMIT_BAL'],kde=True,bins=200, color="blue")

plt.show()
df_cnt = train_df[train_df['default.payment.next.month']==1]['LIMIT_BAL'].value_counts().reset_index()

df_cnt_non = train_df[train_df['default.payment.next.month']==0]['LIMIT_BAL'].value_counts().reset_index()
plt.figure(figsize = (14,6))

plt.title('Defaulters')

sns.barplot(x='index', y='LIMIT_BAL', data=df_cnt.loc[:17, :])
plt.figure(figsize = (14,6))

plt.title('Non-Defaulters')

sns.barplot(x='index', y='LIMIT_BAL', data=df_cnt_non.loc[:17, :])
df_corr = train_df.corr()

df_corr
plt.figure(figsize = (10,10))

sns.heatmap(df_corr)
col = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for c in col:

  sns.barplot(x=c, y='LIMIT_BAL', data=train_df[train_df['default.payment.next.month']==1])

  plt.show()
target = 'default.payment.next.month'

predictors = [  'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 

                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 

                'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',

                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
from sklearn.model_selection import train_test_split 



train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier
clf = RandomForestClassifier(random_state=42,

                             criterion='gini',

                             n_estimators=100,

                             verbose=False)

clf.fit(train_df[predictors], train_df[target].values)
preds = clf.predict(val_df[predictors])
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()   
roc_auc_score(val_df[target].values, preds)
clf = AdaBoostClassifier(random_state=42,

                         algorithm='SAMME.R',

                         learning_rate=0.8,

                         n_estimators=100)

clf.fit(train_df[predictors], train_df[target].values)
preds = clf.predict(val_df[predictors])
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show() 
roc_auc_score(val_df[target].values, preds)
clf = CatBoostClassifier(iterations=800,

                             learning_rate=0.02,

                             depth=12,

                             eval_metric='AUC',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             od_wait=100)

clf.fit(train_df[predictors], train_df[target].values,verbose=True)
preds = clf.predict(val_df[predictors])
tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show()   
roc_auc_score(val_df[target].values, preds)
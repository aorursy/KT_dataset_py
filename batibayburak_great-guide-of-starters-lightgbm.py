import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score

import matplotlib.pyplot as plt

import lightgbm as lgb

import numpy as np

import seaborn as sns

import os
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/bank-marketing/bank-additional-full.csv", delimiter=';')

df.head()
y= (df['y'] == 'yes')*1

df.drop('y', axis=1, inplace = True)
df.tail()
df.info()
df.columns
print(df.head())
df.isnull().sum()
df.describe()
df['age'].unique()
sns.distplot(df['age'], hist=True, kde=True, 

             bins=int(180/5), color = 'blue',

             hist_kws={'edgecolor':'black'})
sns.countplot(x='duration',data=df)
sns.countplot(x='cons.price.idx',data=df)
sns.countplot(x='emp.var.rate',data=df)
sns.countplot(y='cons.conf.idx',data=df)
sns.countplot(x='euribor3m',data=df)
sns.lmplot( x="age", y="previous", data=df, fit_reg=False, hue='emp.var.rate', legend=False)
sns.lmplot( x="age", y="campaign", data=df, fit_reg=False, hue='emp.var.rate', legend=False)
sns.lmplot( x="age", y="cons.conf.idx", data=df, fit_reg=False, hue='emp.var.rate', legend=False)
sns.jointplot(x='campaign',y='age',data=df)
sns.stripplot(y='campaign',x='age',data=df,jitter=False)
plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(), annot=True)

plt.show()
from sklearn.preprocessing import LabelEncoder



categorical_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',

                      'day_of_week', 'poutcome']



for i in categorical_column:

    le = LabelEncoder()

    df[i] = le.fit_transform(df[i])

print(df.head())
df.drop('duration', inplace = True, axis=1)


df_train, df_test, y_train, y_test = train_test_split(df, y, train_size = 0.7, test_size = 0.3)
lgb_train = lgb.Dataset(data=df_train, label=y_train,  free_raw_data=False)
# Categorical index

categorical_index = [1,2,3,4,5,6,7,8,9,13]

print('Categorical parametres: ' + str(df_train.columns[categorical_index].values))
#Creat Evaluation Dataset 

lgb_eval = lgb.Dataset(data=df_test, label=y_test, reference=lgb_train,  free_raw_data=False)



# Determinate training parametres

params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 31,

    'learning_rate': 0.05,

    'verbose': -1

}


evals_result={}

gbm = lgb.train(params,

                lgb_train,

                valid_sets = lgb_eval,

                categorical_feature = categorical_index,

                num_boost_round= 150,

                early_stopping_rounds= 25,

                evals_result=evals_result)

y_pred = gbm.predict(df_test, num_iteration=gbm.best_iteration)



print('The Best iteration: ', gbm.best_iteration)

print('roc_auc_score:', roc_auc_score(y_test, y_pred))

print('accuracy_score:', accuracy_score(y_test, ( y_pred>= 0.5)*1))
ax = lgb.plot_metric(evals_result, metric='auc')

ax.set_title('Variation of the Curved Area According to Iteration')

ax.set_xlabel('İteration')

ax.set_ylabel('roc_auc_score')

ax.legend_.remove()
ax = lgb.plot_importance(gbm, max_num_features=10)

ax.set_title('The values of Parametres')

ax.set_xlabel('Values')

ax.set_ylabel('Parametres')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# visualization
import seaborn as sns
color = sns.color_palette()

import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()

import cufflinks as cf
cf.go_offline()

print(os.listdir("../input"))
df_train = pd.read_csv('../input/application_train.csv')
df_test = pd.read_csv('../input/application_test.csv')
df_train.columns.values
print(df_train.shape)
df_train.head()
# checking missing data
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)
df_bureau = pd.read_csv('../input/bureau.csv')
print(df_bureau.shape)
print(df_bureau.columns)
df_bureau.head()
# checking missing data
total = df_bureau.isnull().sum().sort_values(ascending = False)
percent = (df_bureau.isnull().sum()/df_bureau.isnull().count()*100).sort_values(ascending = False)
missing_bureau_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bureau_data.head(8)
df_bureau_balance = pd.read_csv('../input/bureau_balance.csv')
print(df_bureau_balance.shape)
print(df_bureau_balance.columns)
df_bureau_balance.head()
# checking missing data
total = df_bureau_balance.isnull().sum().sort_values(ascending = False)
percent = (df_bureau_balance.isnull().sum()/df_bureau_balance.isnull().count()*100).sort_values(ascending = False)
missing_bureau_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bureau_balance_data.head(3)
df_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')
print(df_cash_balance.shape)
print(df_cash_balance.columns)
df_cash_balance.head()
# checking missing data
total = df_cash_balance.isnull().sum().sort_values(ascending = False)
percent = (df_cash_balance.isnull().sum()/df_cash_balance.isnull().count()*100).sort_values(ascending = False)
missing_POS_CASH_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_POS_CASH_balance_data.head(3)
df_card_balance = pd.read_csv('../input/credit_card_balance.csv')
print(df_card_balance.shape)
print(df_card_balance.columns)
df_card_balance.head()
# checking missing data
total = df_card_balance.isnull().sum().sort_values(ascending = False)
percent = (df_card_balance.isnull().sum()/df_card_balance.isnull().count()*100).sort_values(ascending = False)
missing_credit_card_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_credit_card_balance_data.head(10)
df_previous = pd.read_csv('../input/previous_application.csv')
print(df_previous.shape)
print(df_previous.columns)
df_previous.head()
# checking missing data
total = df_previous.isnull().sum().sort_values(ascending = False)
percent = (df_previous.isnull().sum()/df_previous.isnull().count()*100).sort_values(ascending = False)
missing_previous_application_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_previous_application_data.head(15)
df_installments_payments = pd.read_csv('../input/installments_payments.csv')
print(df_installments_payments.shape)
print(df_installments_payments.columns)
df_installments_payments.head()
# checking missing data
total = df_installments_payments.isnull().sum().sort_values(ascending = False)
percent = (df_installments_payments.isnull().sum()/df_installments_payments.isnull().count()*100).sort_values(ascending = False)
missing_installments_payments_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_installments_payments_data.head(3)
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(df_train["AMT_CREDIT"])
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_INCOME_TOTAL")
ax = sns.distplot(df_train["AMT_ANNUITY"].dropna())
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
ax = sns.distplot(df_train["AMT_GOODS_PRICE"].dropna())
temp = df_train["NAME_TYPE_SUITE"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of Name of type of the Suite in % ",
    xaxis=dict(
        title='Name of type of the Suite',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Name of type of the Suite in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Loan Repayed or not')
temp = df_train["NAME_CONTRACT_TYPE"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      #"name": "Types of Loans",
      #"hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Types of loan",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Loan Types",
                "x": 0.17,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
temp1 = df_train["FLAG_OWN_CAR"].value_counts()
temp2 = df_train["FLAG_OWN_REALTY"].value_counts()

fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [0, .48]},
      "name": "Own Car",
      "hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    },
    {
      "values": temp2.values,
      "labels": temp2.index,
      "text":"Own Reality",
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "Own Reality",
      "hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    }],
  "layout": {
        "title":"Purpose of loan",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Own Car",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Own Reality",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
temp = df_train["NAME_INCOME_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Income sources of Applicant\'s', hole = 0.5)
temp = df_train["NAME_FAMILY_STATUS"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Family Status of Applicant\'s', hole = 0.5)
temp = df_train["OCCUPATION_TYPE"].value_counts()
temp.iplot(kind='bar', xTitle = 'Occupation', yTitle = "Count", title = 'Occupation of Applicant\'s who applied for loan', color = 'green')
temp = df_train["NAME_EDUCATION_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Education of Applicant\'s', hole = 0.5)
temp = df_train["NAME_HOUSING_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Type of House', hole = 0.5)
temp = df_train["ORGANIZATION_TYPE"].value_counts()
temp.iplot(kind='bar', xTitle = 'Organization Name', yTitle = "Count", title = 'Types of Organizations who applied for loan ', color = 'red')
data = [
    go.Heatmap(
        z= df_train.corr().values,
        x= df_train.columns.values,
        y= df_train.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Pearson Correlation of features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
from sklearn import preprocessing
# I check columns, which are dtupe -> 'object'

categorical_features = [
    categorical for categorical in df_train.columns if df_train[categorical].dtype == 'object'
]
for i in categorical_features:
    lben = preprocessing.LabelEncoder()
    lben.fit(list(df_train[i].values.astype('str')) + list(df_test[i].values.astype('str')))
    df_train[i] = lben.transform(list(df_train[i].values.astype('str')))
    df_test[i] = lben.transform(list(df_test[i].values.astype('str')))
df_train.fillna(-999, inplace = True)
df_test.fillna(-999, inplace = True)
import lightgbm as lgb
from sklearn.model_selection import train_test_split 
# target variable 
Y = df_train['TARGET']
test_id = df_test['SK_ID_CURR']

train_X = df_train.drop(['TARGET', 'SK_ID_CURR'], axis = 1)
test_X = df_test.drop(['SK_ID_CURR'], axis = 1)
# prepare training and validation data
x_train, x_val, y_train, y_val = train_test_split(
    train_X, 
    Y, 
    random_state=18
)

lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)
# params for model

params = {
    'task': 'train', 
    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc', 
    'learning_rate': 0.05, 
    'num_leaves': 32, 
    'num_iteration': 500, 
    'verbose': 0 
}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=10)
lgb.plot_importance(model, figsize=(12, 20));
# for competition

pred = model.predict(test_X)
sub = pd.DataFrame()
sub['SK_ID_CURR'] = test_id
sub['TARGET'] = pred
sub.to_csv("baseline_submission.csv", index=False)
sub.head()
from lightgbm import LGBMClassifier
clf = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        colsample_bytree=.8,
        subsample=.8,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
    )
clf.fit(x_train, 
        y_train,
        eval_set= [(x_train, y_train), (x_val, y_val)], 
        eval_metric='auc', 
        verbose=0, 
        early_stopping_rounds=30
       )
# for competition

pred_1 = clf.predict(test_X)
sub = pd.DataFrame()
sub['SK_ID_CURR'] = test_id
sub['TARGET'] = pred
sub.to_csv("submission_clf.csv", index=False)
sub.head()













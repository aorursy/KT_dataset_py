# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data=pd.read_csv(r'../input/train.csv')
test_data=pd.read_csv(r'../input/test.csv')
!../input/Data_Dictionary.xlsx
dict1=pd.read_excel(r'../input/Data_Dictionary.xlsx')
dict1
train_data.head(20)
train_data.tail()
train_data.info()
train_data.describe(include='all')
train_data.isnull().sum()
target_col=train_data['target']

plt.figure(figsize=(8,6),dpi=80)
sns.violinplot(x=train_data['feature_1'],y=train_data['target'],data=train_data,)
plt.xlabel('Target',size=18,color='r')
plt.ylabel('Feature_1',size=18,color='r')
plt.title('Feature_1 Vs Target',size=20,color='blue')
plt.legend();
plt.figure(figsize=(8,6),dpi=80)
sns.violinplot(x=train_data['feature_2'],y=train_data['target'],data=train_data,)
plt.xlabel('Target',size=18,color='r')
plt.ylabel('Feature_2',size=18,color='r')
plt.title('Feature_2 Vs Target',size=20,color='blue')
plt.legend();
plt.figure(figsize=(8,6),dpi=80)
sns.violinplot(x=train_data['feature_3'],y=train_data['target'],data=train_data)
plt.xlabel('Target',size=18,color='r')
plt.ylabel('Feature_3',size=18,color='r')
plt.title('Feature_3 Vs Target',size=20,color='blue')
plt.legend();
plt.figure(figsize=(8,6))
sns.distplot(train_data['target'],bins=50,hist=True,color='#F71212',kde=False)
print('there are {0} sample in target below -30'.format(train_data.loc[train_data.target<-30].shape[0]))
feature1=train_data['feature_1'].value_counts().sort_index(ascending=False)
sns.barplot(x=feature1.index,y=feature1.values,ci=100,color='#FFFF00')
plt.title('Feature_1',color='red',size=18);
feature2=train_data['feature_2'].value_counts().sort_index(ascending=False)
sns.barplot(x=feature2.index,y=feature2.values,ci=100,color='#00FF7F')
plt.title('Feature_2',color='red',size=18);
feature3=train_data['feature_3'].value_counts().sort_index(ascending=False)
sns.barplot(x=feature3.index,y=feature3.values,ci=100,color='#556B2F')
plt.title('Feature_3',color='red',size=18);
train_active_months=train_data['first_active_month'].value_counts().sort_index(ascending=False)
test_active_months=test_data['first_active_month'].value_counts().sort_index(ascending=False)
data = [go.Scatter(x=train_active_months.index, y=train_active_months.values, name='train',opacity=1), 
        go.Scatter(x=test_active_months.index, y=test_active_months.values, name='test',opacity=1)]
layout = go.Layout(dict(title = "Counts of first active month",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Count'),
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout))
test_data.head()
test_data.describe()
test_data.info()
test_data.isnull().sum()
historical_data=pd.read_csv(r'../input/historical_transactions.csv')
historical_data.head(20)
historical_data.tail()
historical_data.info()
historical_data.isnull().sum()
historical_data['installments'].value_counts()
plt.figure(figsize=(8,6),dpi=100)
install=historical_data['installments'].value_counts().sort_index(ascending=False)
sns.barplot(x=install.index,y=install.values,ci=100,color='#900C3F')
plt.title('installment detail',color='red',size=18);

trai=historical_data.groupby(['installments'])['authorized_flag'].value_counts()
trai.head()
new_merchant_data=pd.read_csv(r'../input/new_merchant_transactions.csv')
new_merchant_data.head(120)
new_merchant_data.info()
new_merchant_data.isna().sum()
new_merchant_data['category_2'].fillna(value=0.1,inplace=True)
new_merchant_data['category_3'].fillna('B',inplace=True)
new_merchant_data['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
new_merchant_data['authorized_flag']=new_merchant_data['authorized_flag'].apply(lambda x:1 if x=='Y'else 0)
new_merchant_data['authorized_flag'].value_counts().plot(kind='bar',title='authorize_flag value',color='red')
card_total=new_merchant_data.groupby(['card_id'])['purchase_amount'].sum().sort_values()
card_total.head()
card_total.tail()
hist = historical_data.groupby(["card_id"])
hist= hist["purchase_amount"].size().reset_index()
hist.columns = ["card_id", "hist_transactions"]
train_data = pd.merge(train_data,hist, on="card_id", how="left")
test_data = pd.merge(test_data,hist, on="card_id", how="left")
hist.head()
hist = historical_data.groupby(["card_id"])
hist= hist["purchase_amount"].agg(['sum','mean','max','min','std']).reset_index()
hist.columns=['card_id','sum_hist_tran','mean_hist_tran','max_hist_tran','min_hist_tran','std_hist_tran']
train_data=pd.merge(train_data,hist,on='card_id',how='left')
test_data=pd.merge(test_data,hist,on='card_id',how='left')
hist.head()
train_data.head()
merchant = new_merchant_data.groupby(["card_id"])
merchant= merchant["purchase_amount"].size().reset_index()
merchant.columns = ["card_id", "merchant_transactions"]
train_data = pd.merge(train_data,merchant, on="card_id", how="left")
test_data = pd.merge(test_data,merchant, on="card_id", how="left")
merchant.head()
train_data.head()
merchant= new_merchant_data.groupby(["card_id"])
merchant= merchant["purchase_amount"].agg(['sum','mean','max','min','std']).reset_index()
merchant.columns=['card_id','sum_merchant_tran','mean_merchant_tran','max_merchant_tran','min_merchant_tran','std_merchant_tran']
train_data=pd.merge(train_data,merchant,on='card_id',how='left')
test_data=pd.merge(test_data,merchant,on='card_id',how='left')
merchant.head()
train_data['first_active_month']=pd.to_datetime(train_data['first_active_month'])
test_data['first_active_month']=pd.to_datetime(train_data['first_active_month'])
train_data.drop('target',axis=1,inplace=True)
train_data.head()
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
import lightgbm as lgb
train_data["year"] = train_data["first_active_month"].dt.year
test_data["year"] = test_data["first_active_month"].dt.year
train_data["month"] = train_data["first_active_month"].dt.month
test_data["month"] = test_data["first_active_month"].dt.month

cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month", 
               "hist_transactions", "sum_hist_tran", "mean_hist_tran", "std_hist_tran", 
               "min_hist_tran", "max_hist_tran",
               "merchant_transactions", 'sum_merchant_tran','mean_merchant_tran',
               'max_merchant_tran','min_merchant_tran','std_merchant_tran',
              ]

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_freq" : 5,
        "bagging_seed" : 2019,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 10000, valid_sets=[lgval], 
        early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

train_X = train_data[cols_to_use]
test_X = test_data[cols_to_use]
train_y = target_col.values

pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state=2019, shuffle=True)
for dev_index, val_index in kf.split(train_data):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
pred_test /= 5.
fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax,color='red')
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15,color='r')
plt.show()
submission_data = pd.DataFrame({"card_id":test_data["card_id"].values})
submission_data["target"] = pred_test
submission_data.to_csv("baseline_lgb1.csv", index=False)






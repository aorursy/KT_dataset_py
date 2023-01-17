import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import os
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
from sklearn.linear_model import Lasso
print(os.listdir("../input"))
X_train = pd.read_csv('../input/train.csv')
X_train.head()
X_test = pd.read_csv('../input/test.csv')
X_test.head()
X_train.shape
X_test.shape
feature_cols = [c for c in X_train.columns if c not in ["ID", "target"]]
print(feature_cols)
flat_values = X_train[feature_cols].values.flatten()
flat_values.shape # 4459 x (4993-2)
labels = 'Zero_values','Non-Zero_values'
values = [sum(flat_values==0), sum(flat_values!=0)]
colors = ['rgba(55, 12, 233, .6)','rgba(125, 42, 123, .2)']
Plot = go.Pie(labels=labels, values=values,marker=dict(colors=colors,line=dict(color='#fff', width= 3)))
layout = go.Layout(title='Value distribution', height=400)
fig = go.Figure(data=[Plot], layout=layout)
iplot(fig)
X_train.info()
X_test.info()
X_train.describe()
print(X_train)
X_train_total = X_train.isnull().sum().reset_index()
X_train_total.columns  = ['Feature_Name','Missing_value']
X_train_total_val = X_train_total[X_train_total['Missing_value']>0]
X_train_total_val = X_train_total.sort_values(by ='Missing_value')
X_train_total_val.head()
X_train_total.reset_index()
X_test_total = X_test.isnull().sum().reset_index()
X_test_total.columns  = ['Feature_Name','Missing_value']
X_test_total_val = X_test_total[X_test_total['Missing_value']>0]
X_test_total_val = X_test_total.sort_values(by ='Missing_value')
X_test_total_val.head()
sns.distplot(X_train['target'])
sns.distplot(np.log1p(X_train['target']))
X_train_data = X_train.drop(['ID','target'],axis=1)
X_train_data.head()
y_train_data = np.log1p(X_train["target"])
y_train_data.head()
X_test_data = X_test.drop('ID', axis = 1)
X_test_data.head()
features = SelectKBest(mutual_info_regression,k=200)
print(features)
X_tr = features.fit_transform(X_train_data,y_train_data)
X_te = features.transform(X_test_data)
tr_data = scaler.fit_transform(X_tr)
te_data = scaler.fit_transform(X_te)
reg = Lasso(alpha=0.0000001, max_iter = 10000)
reg.fit(tr_data,y_train_data)
y_pred = reg.predict(te_data)
y_pred
sub = pd.read_csv('../input/sample_submission.csv')
#y_pred = np.clip(y_pred,y_train.min(),y_train.max())
sub["target"] = np.expm1(y_pred)
print(sub.head())
sub.to_csv('sub_las.csv', index=False)
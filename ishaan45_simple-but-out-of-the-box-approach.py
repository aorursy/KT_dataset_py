import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import os
print(os.listdir("../input"))
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
print("The shape of the training set is:",train_data.shape)
print("The shape of the test set is:", test_data.shape)
feature_cols = [c for c in train_data.columns if c not in ["ID", "target"]]
flat_values = train_data[feature_cols].values.flatten()

labels = 'Zero_values','Non-Zero_values'
values = [sum(flat_values==0), sum(flat_values!=0)]
colors = ['rgba(55, 12, 233, .6)','rgba(125, 42, 123, .2)']

Plot = go.Pie(labels=labels, values=values,marker=dict(colors=colors,line=dict(color='#fff', width= 3)))
layout = go.Layout(title='Value distribution', height=400)
fig = go.Figure(data=[Plot], layout=layout)
iplot(fig)
train_data.info()
test_data.info()
train_data.describe()
def missing_data(data): #calculates missing values in each column
    total = data.isnull().sum().reset_index()
    total.columns  = ['Feature_Name','Missing_value']
    total_val = total[total['Missing_value']>0]
    total_val = total.sort_values(by ='Missing_value')
    return total_val
missing_data(train_data).head()
missing_data(test_data).head()

sns.distplot(np.log1p(train_data['target']))
X_train = train_data.drop(['ID','target'],axis=1)
y_train = np.log1p(train_data["target"])
X_test = test_data.drop('ID', axis = 1)
#X = X_train.copy()
#Xa = X_test.copy()
#clf = IsolationForest(max_samples=100, random_state= 0)
#clf.fit(X)
#y_pred = clf.predict(X)
#y_pred_df = pd.DataFrame(data=y_pred,columns = ['Values'])
#y_pred_df['Values'].value_counts()
#anomaly_score = clf.decision_function(X_train)
#anomaly_score
#y_test_pred = clf.predict(Xa)
#y_test_pred_df = pd.DataFrame(data=y_test_pred,columns = ['Out_Values'])
#y_test_pred_df['Out_Values'].value_counts()
#anomaly_score = clf.decision_function(X_test)
#anomaly_score
feat = SelectKBest(mutual_info_regression,k=200)
X_tr = feat.fit_transform(X_train,y_train)
X_te = feat.transform(X_test)
tr_data = scaler.fit_transform(X_tr)
te_data = scaler.fit_transform(X_te)
reg = Lasso(alpha=0.0000001, max_iter = 10000)
reg.fit(tr_data,y_train)
y_pred = reg.predict(te_data)
y_pred
sub = pd.read_csv('../input/sample_submission.csv')
#y_pred = np.clip(y_pred,y_train.min(),y_train.max())
sub["target"] = np.expm1(y_pred)
print(sub.head())
sub.to_csv('sub_las.csv', index=False)




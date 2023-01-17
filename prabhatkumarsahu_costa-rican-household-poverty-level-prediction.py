# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
color = sns.color_palette()
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
train_df=pd.read_csv('../input/train.csv')
train_df.shape
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
sns.countplot(x="Target",  data=train_df)
train_df_2=train_df.drop(['Id','v2a1'],axis=1)
list_to_compare=['no','yes','No','Yes','NO','YES']
for col in train_df_2:
    unique=train_df_2[col].unique()
    if (set(list_to_compare)&set(unique)):
        print("Column ",col,"has the values ",set(list_to_compare)&set(unique))
# Let us replace all yes with 1 and no with 0 as defined in the competition page, data description.

train_df=train_df.replace('no','0')
train_df=train_df.replace('yes','1')
missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df

train_df.groupby(['idhogar','v18q']).aggregate('count')['v18q1'].reset_index()
train_df['rez_esc'].fillna(0, inplace=True)
train_df=train_df[~train_df['meaneduc'].isnull()]
train_df['v18q1'].fillna(0, inplace=True)
train_df['rez_esc'].fillna(0, inplace=True)

train_df_reg=train_df
train_df_reg=train_df
train_df_reg_train=train_df_reg[~train_df_reg['v2a1'].isnull()]
train_df_reg_train.shape
train_df_reg_test=train_df_reg[train_df_reg['v2a1'].isnull()]
train_df_reg_test.shape

train_df_reg_train_Y=train_df_reg_train['v2a1']

train_df_reg_test_Y=train_df_reg_test['v2a1']
train_df_reg_train_X=train_df_reg_train.loc[:, train_df_reg_train.columns != 'v2a1']
train_df_reg_test_X=train_df_reg_test.loc[:, train_df_reg_test.columns != 'v2a1']
train_df_reg_train_X_id=train_df_reg_train_X

train_df_reg_train_X=train_df_reg_train_X.drop(['Id','idhogar'],axis=1)
train_df_reg_test_X_id=train_df_reg_test_X
train_df_reg_test_X=train_df_reg_test_X.drop(['Id','idhogar'],axis=1)

regr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)
regr.fit(train_df_reg_train_X,train_df_reg_train_Y)
train_df_reg_test_Y=regr.predict(train_df_reg_test_X)

train_df_reg_test_Y=np.round(train_df_reg_test_Y)
train_df_reg_test_Y = pd.DataFrame(train_df_reg_test_Y, dtype='float')

train_df_reg_test_Y.shape
train_df_reg_test_Y
train_df_reg_test_X_id.loc[:,'v2a1']=train_df_reg_test_Y.values
train_df_reg_test_X_id.shape
train_df_reg_train_X_id.loc[:,'v2a1']=train_df_reg_train_Y.values
train_df_reg_train_X_id['v2a1']
output_train_df = pd.concat([train_df_reg_train_X_id, train_df_reg_test_X_id], axis=0, ignore_index=True,sort='False')
output_train_df.shape
merge_df=output_train_df[['Id','v2a1']]
train_df_reg_train_X_id['v2a1']

imputed_df=train_df.merge(merge_df,on='Id',how='outer')
imputed_df.shape
imputed_df=imputed_df.drop(['v2a1_x'],axis=1)
imputed_df.shape
imputed_df=imputed_df.rename(index=str, columns={"v2a1_y": "v2a1"})

unique_df = imputed_df.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df
imputed_df= imputed_df.drop(['elimbasu5'],axis=1)
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

labels = []
values = []
for col in train_df.columns:
    if col not in ["Id", "Target"]:
        labels.append(col)
        values.append(spearmanr(train_df[col].values, train_df["Target"].values)[0])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='b')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()

cols_to_use = corr_df[(corr_df['corr_values']>0.21) | (corr_df['corr_values']<-0.21)].col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()
train_X = imputed_df.drop(["Id", "Target"], axis=1)
train_y = imputed_df["Target"].values
import sklearn
from sklearn import ensemble
model = ensemble.ExtraTreesClassifier(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
train_X=train_X.drop(['idhogar'],axis=1)
model.fit(train_X, train_y)

## plot the importances ##
feat_names = train_X.columns.values
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()
from sklearn.linear_model import LogisticRegression

X=train_X
y=train_y
clf = LogisticRegression(random_state=0, solver='lbfgs',  multi_class='multinomial').fit(X, y)
test_df=pd.read_csv('../input/test.csv')
test_df.shape
test_df_orig=test_df
test_df=test_df.replace('no','0')
test_df=test_df.replace('yes','1')

test_df.shape
test_df['rez_esc'].fillna(0, inplace=True)
test_df['meaneduc'].fillna(0, inplace=True)
test_df['SQBmeaned'].fillna(0, inplace=True)
test_df['v18q1'].fillna(0, inplace=True)
test_df['rez_esc'].fillna(0, inplace=True)
test_df.shape

test_df_reg=test_df
test_df_reg_train=test_df_reg[~test_df_reg['v2a1'].isnull()]
test_df_reg_train.shape
test_df_reg_test=test_df_reg[test_df_reg['v2a1'].isnull()]
test_df_reg_test.shape

test_df_reg_train_Y=test_df_reg_train['v2a1']
test_df_reg_test_Y=test_df_reg_test['v2a1']
test_df_reg_train_X=test_df_reg_train.loc[:, test_df_reg_train.columns != 'v2a1']
test_df_reg_test_X=test_df_reg_test.loc[:, test_df_reg_test.columns != 'v2a1']
test_df_reg_train_X_id=test_df_reg_train_X
test_df_reg_train_X=test_df_reg_train_X.drop(['Id','idhogar'],axis=1)
test_df_reg_test_X_id=test_df_reg_test_X
test_df_reg_test_X=test_df_reg_test_X.drop(['Id','idhogar'],axis=1)
regr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)
regr.fit(test_df_reg_train_X,test_df_reg_train_Y)
test_df_reg_test_Y=regr.predict(test_df_reg_test_X)

test_df_reg_test_Y=np.round(test_df_reg_test_Y)

test_df_reg_test_Y = pd.DataFrame(test_df_reg_test_Y, dtype='float')
test_df_reg_test_Y.shape
test_df_reg_test_X_id.loc[:,'v2a1']=test_df_reg_test_Y.values

test_df_reg_test_X_id.shape
test_df_reg_train_X_id.loc[:,'v2a1']=test_df_reg_train_Y.values
test_df_reg_train_X_id['v2a1']
output_test_df = pd.concat([test_df_reg_train_X_id, test_df_reg_test_X_id], axis=0, ignore_index=True,sort='False')

output_test_df.shape
merge_test_df=output_test_df[['Id','v2a1']]

imputed_test_df=test_df.merge(merge_test_df,on='Id',how='outer')
imputed_test_df.shape
imputed_test_df=imputed_test_df.drop(['v2a1_x'],axis=1)
imputed_test_df=imputed_test_df.rename(index=str, columns={"v2a1_y": "v2a1"})
imputed_test_df=imputed_test_df.drop(['Id','idhogar'],axis=1)
imputed_test_df.shape
imputed_test_df= imputed_test_df.drop(['elimbasu5'],axis=1)
target_y=clf.predict(imputed_test_df)
target_y
type(test_df_orig['Id'])
len(test_df_orig['Id'])
submission_df=pd.concat([test_df_orig['Id'], pd.Series(target_y)], axis=1)
submission_df.shape
submission_df=submission_df.rename(index=int, columns={0: "Target"})
submission_df.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=1000, random_state=0,n_jobs=-1)
rf_clf.fit(X, y)

target_y=rf_clf.predict(imputed_test_df)

submission_df=pd.concat([test_df_orig['Id'], pd.Series(target_y)], axis=1)
submission_df=submission_df.rename(index=int, columns={0: "Target"})
submission_df.to_csv('submission_rf.csv', index=False)
from xgboost import XGBClassifier
model = XGBClassifier()
X=X.replace('no',0)
X=X.replace('yes',1)
X['dependency']=X['dependency'].astype(str).astype(float,copy=True)
X['edjefe']=X['edjefe'].astype(str).astype(int,copy=True)
X['edjefa']=X['edjefa'].astype(str).astype(int,copy=True)
type(X['dependency'])
X.to_csv('debug.csv')
dtype_df = X.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
model.fit(X, y)
imputed_test_df['dependency']=imputed_test_df['dependency'].astype(str).astype(float,copy=True)
imputed_test_df['edjefe']=imputed_test_df['edjefe'].astype(str).astype(float,copy=True)
imputed_test_df['edjefa']=imputed_test_df['edjefa'].astype(str).astype(float,copy=True)
target_y=model.predict(imputed_test_df)
submission_df=pd.concat([test_df_orig['Id'], pd.Series(target_y)], axis=1)
submission_df=submission_df.rename(index=int, columns={0: "Target"})
submission_df.to_csv('submission_xgb.csv', index=False)

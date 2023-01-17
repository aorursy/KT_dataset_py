# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm_notebook
from matplotlib.patches import Rectangle
import seaborn as sns
!pip install pydicom
import pydicom as dcm
%matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.api import OLS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score
IS_LOCAL = False

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df= pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df= pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
submission_format= pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
print('Train data shape:', train_df.shape)
print('Test data shape:', test_df.shape)
train_df.head()
test_df
#train_df[[train_df["Patient"]= "ID00419637202311204720264"]]
if 'ID00419637202311204720264' in train_df.columns:
    print("ID00419637202311204720264 Exists")
elif 'ID00421637202311550012437' in train_df.columns:
    print("ID00421637202311550012437 Exists")
elif 'ID00422637202311677017371' in train_df.columns:
    print("ID00422637202311677017371 Exists")
elif 'ID00423637202312137826377' in train_df.columns:
    print("ID00423637202312137826377 Exists")
elif 'ID00426637202313170790466' in train_df.columns:
    print("ID00426637202313170790466 Exists")
else:
    print("None exists..")
train_df.isnull().sum()
train_df.dtypes
train_df.Sex.unique()
train_df.SmokingStatus.unique()
a = sns.distplot(train_df["Age"], color='cyan', hist=False)
a.set_title("Patient Age Distribution", fontsize=16)
f, ax = plt.subplots(1,1, figsize=(6,4))
total = float(len(train_df))
sns.countplot(x = "SmokingStatus", data= train_df, hue="Sex", palette='husl')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(100*height/total),
            ha="center") 
plt.show()
train_df.groupby(["SmokingStatus"]).count()
import plotly.express as px
fig = px.pie(train_df, values='FVC', names='SmokingStatus', title='FVC for different categories')
fig.show()

#plt.subplot(the_grid[0, 0], title='Selected Flavors of Pies')

sns.barplot(y='SmokingStatus',x='FVC', data=train_df, palette='husl')

plt.suptitle('FVC versus SmokingStatus', fontsize=12)
sns.barplot(y='Sex',x='FVC', data=train_df, palette='cubehelix')

plt.suptitle('FVC versus Gender', fontsize=12)
train_df['Sex']= train_df['Sex'].replace({'Female':0,'Male':1})
train_df['SmokingStatus']= train_df['SmokingStatus'].replace({'Ex-smoker':1, 'Never smoked':2, 'Currently smokes':3})
train_df['Sex']= train_df['Sex'].astype('int64')
train_df['SmokingStatus']= train_df['SmokingStatus'].astype('int64')
train_df.head()
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
X= train_df[['SmokingStatus','Age','Sex','Weeks','Percent']]
y= train_df['FVC']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.15, random_state=0)

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledRidge', Pipeline([('Scaler', StandardScaler()),('Ridge', Ridge())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
pipelines.append(('ScaledAda', Pipeline([('Scaler', StandardScaler()),('Ada', AdaBoostRegressor())])))
pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBRegressor())])))
pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('MLP', MLPRegressor())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Algorithm comparison based on mean and standard deviation
fig = plt.figure(figsize=(15,5))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
xgb_model= XGBRegressor(n_estimators= 100, random_state=0).fit(X_train, y_train)
xgb_preds= xgb_model.predict(X_test)
df = pd.DataFrame({'Actuals': y_test, 'Predicts': xgb_preds})
df
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10), rot=0)
plt.show()
test_df['Patient_Week'] = test_df['Patient'].astype(str)+"_"+test_df['Weeks'].astype(str)
test_df.head()
from sklearn.preprocessing import LabelEncoder
cat_features = ['Sex','SmokingStatus']
encode = test_df[cat_features].apply(LabelEncoder().fit_transform)
test_data = test_df[['Patient','Percent','Weeks','Age']].join(encode)
submission= submission_format
submission.head()
submission[['Patient','Weeks']] = submission.Patient_Week.str.split("_",expand=True,)
submission.head()
submission = submission.drop('FVC',1)
submission = submission.drop('Confidence',1)
test_data = test_data.drop('Weeks',1)
submission.head()
test_data.head()
submission2 = pd.merge(submission,test_data,on='Patient',how='left')
submission2.head(100)
X_sub = submission2[['SmokingStatus','Age','Sex','Weeks','Percent']]
X_sub.dtypes
X_sub['SmokingStatus']= X_sub['SmokingStatus'].astype('int64')
X_sub['Sex']= X_sub['Sex'].astype('int64')
X_sub['Weeks']= X_sub['Weeks'].astype('int64')
X_sub.head()
submission2['FVC'] = xgb_model.predict(X_sub)
submission2.head()
submission2['FVC_Mean'] = submission2.groupby(['SmokingStatus','Sex','Age'])['FVC'].transform('mean')
submission2['FVC_Mean'].sample(5)
submission2['Confidence'] = 100*submission2['FVC']/submission2['FVC_Mean']
submission2['Confidence'].sample(5)
submission_final = submission2[['Patient_Week','FVC','Confidence']]
submission_final.head()
submission_final['FVC'] = submission_final['FVC'].astype(int)
submission_final['Confidence'] = submission_final['Confidence'].astype(int)
submission_final.head()
submission_final.to_csv('/kaggle/working/submission_xgboost.csv',index=False)

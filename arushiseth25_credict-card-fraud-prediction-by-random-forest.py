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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from plotly import tools

import gc
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import  roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost  import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
print("credit card fraud detection data - rows:", df.shape[0], "columns: " , df.shape[1])
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count() * 100).sort_values(ascending = False)
pd.concat([total,percent], axis = 1, keys= ['total','percent'])
temp = df['Class'].value_counts()
df_fraud = pd.DataFrame({'Class': temp.index, 'values': temp.values})
df_fraud
import plotly.graph_objs as go
trace = go.Bar(x = df_fraud['Class'], y = df_fraud['values'], 
              name = "Plotting Unbalanced Data where 1 is Fraud and 0 is Non Fraud Data",
              marker = dict(color="Red"),
              text = df_fraud['values'])
data = [trace]
layout = dict(title = "Plotting Unbalanced Data where 1 is Fraud and 0 is Non Fraud Data", 
             xaxis = dict(title = 'Class', showticklabels = True),
             yaxis = dict(title = 'Number of Transactions'),
             hovermode = 'closest',
             width = 800)
fig = dict(data=data, layout = layout)
iplot(fig, filename = 'class')
class_0 = df.loc[df['Class'] == 0]['Time']
class_1 = df.loc[df['Class'] == 1]['Time']
hist_data = [class_0, class_1]
group_labels = ['Not_Fraud', 'Fraud']
fig = ff.create_distplot(hist_data,group_labels, show_hist = False, show_rug = False )
fig['layout'].update(title= "Time Density Plot", xaxis=  dict(title = 'Time[s]'))
iplot(fig, filename = 'dist_only')
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x = "Class", y = 'Amount', hue = 'Class', data = df, palette = 'PRGn', showfliers = True)
s = sns.boxplot(ax = ax2, x = "Class", y = 'Amount', hue = 'Class', data = df, palette = 'PRGn', showfliers = False)
plt.show()
tmp = df[['Amount','Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()
class_1.describe()
fraud =df.loc[df['Class'] == 1]
trace = go.Scatter(x = fraud['Time'], y = fraud['Amount'],
                  name = 'Amount',
                  marker = dict(color = 'rgb(238,23,11)',
                               line = dict(color = 'red', width = 1),
                               opacity = 0.5,
                               ),
                   text = fraud['Amount'],
                   mode = 'markers'
                  )

data = [trace]
layout = dict(title = 'Amount of Fraud Transactions',
             xaxis = dict(title = 'Time[s]', showticklabels = True),
             yaxis =  dict(title = 'Amount'),
              hovermode = 'closest')

fig = dict(data = data, layout = layout)
iplot(fig, filename = 'fraud_amount')
                   
plt.figure(figsize = (14,14))
plt.title('Corelation Plot')
corr = df.corr()
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, linewidths = .1, cmap = 'Reds')
plt.show()
s = sns.lmplot(x = 'V20', y = 'Amount', data = df, hue = 'Class', fit_reg = True, scatter_kws = {'s':2})
s = sns.lmplot(x = 'V7', y = 'Amount', data = df, hue = 'Class', fit_reg = True, scatter_kws = {'s':2})
plt.show()
s = sns.lmplot(x = 'V2', y = 'Amount', data = df, hue = 'Class', fit_reg = True, scatter_kws = {'s':2})
s = sns.lmplot(x = 'V5', y = 'Amount', data = df, hue = 'Class', fit_reg = True, scatter_kws = {'s':2})
plt.show()
df.columns
target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class']
train_df ,test_df = train_test_split(df,  shuffle = True)
#test_size = TEST_SIZE, random_state = RANDOM_STATE,
train_df ,valid_df = train_test_split(train_df, shuffle = True)
#, test_size = VALID_SIZE, random_state = RANDOM_STATE,
clf = RandomForestClassifier(n_estimators=100)
#n_jobs= NO_JOBS, criterion= RFC_METRIC, n_estimators=NUM_ESTIMATORS, verbose=False
clf.fit(train_df[predictors], train_df[target].values)
preds = clf.predict(valid_df[predictors])
tmp = pd.DataFrame({'Feature': predictors , 'Feature Importance' : clf.feature_importances_})
tmp = tmp.sort_values(by = 'Feature Importance')
plt.figure(figsize = (7,4))
plt.title('Feature Importance')
s = sns.barplot(x = 'Feature', y= 'Feature Importance', data = tmp)
s.set_xticklabels(s.get_xticklabels(), rotation =90)
plt.show()
cm = pd.crosstab(valid_df[target].values, preds , rownames = ['Actual'], colnames = ['Predicted'])
fig , (ax1) = plt.subplots(ncols=1, figsize = (5,5))
sns.heatmap(cm, 
            xticklabels = ['Not Fraud', 'Fraud'],
            yticklabels = ['Not Fraud', 'Fraud'],
           annot = True,
           ax = ax1
           , linewidths = .2, linecolor = "Blue", cmap = "Blues")
plt.show()
roc_auc_score(valid_df[target].values, preds)
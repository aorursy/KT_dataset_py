# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly import tools
import cufflinks as cf
cf.go_offline()
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/FIFA 2018 Statistics.csv")
df.head()
df.shape
df.columns
df.info()
df.describe()
sns.countplot(x = "Man of the Match", data = df)
df.loc[:, 'Man of the Match'].value_counts()
plt.figure(figsize =(20,20))
Corr=df[df.columns].corr()
sns.heatmap(Corr,annot=True)
sns.countplot(x = "On-Target", data = df)
df.loc[:, 'On-Target'].value_counts()
sns.countplot(x = "Off-Target", data = df)
df.loc[:, 'Off-Target'].value_counts()
df['index_col'] = df.index
df.tail()
cat_cols   = df.nunique()[df.nunique() < 10].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in df['Man of the Match']]

Id_col     = ['index_col']
target_col = ["Man of the Match"]
winner     = df[df["Man of the Match"] == "Yes"]
runner     = df[df["Man of the Match"] == "No"]
num_cols   = [x for x in df.columns if x not in cat_cols + target_col + Id_col]
#function  for Pieplot for Match attrition types

def plot_pie(column) :
    
    trace1 = go.Pie(values  = winner[column].value_counts().values.tolist(),
                    labels  = winner[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "who won the match",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(245,245,245)")
                                  ),
                    hole    = .6
                   )
    trace2 = go.Pie(values  = runner[column].value_counts().values.tolist(),
                    labels  = runner[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(245,245,245)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Runner Up of the Match" 
                   )

    layout = go.Layout(dict(title = column + " distribution in attrition ",
                            plot_bgcolor  = "rgb(245,245,245)",
                            paper_bgcolor = "rgb(245,245,245)",
                            annotations = [dict(text = "Winner of the match",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = "Runner Up of the match",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .88,y = .5
                                               )
                                          ]
                           )
                      )
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    py.iplot(fig)

#for all categorical columns plot pie
for i in cat_cols :
    plot_pie(i)
#function  for histogram for Match attrition types
def histogram(column) :
    trace1 = go.Histogram(x  = winner[column],
                          histnorm= "percent",
                          name = "Man of the Match",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = runner[column],
                          histnorm = "percent",
                          name = "Runner Up of the Match",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in attrition ",
                             plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)
    
#for all categorical columns plot histogram    
for i in num_cols :
    histogram(i)
#function  for scatter plot matrix  for numerical columns in data
def scatter_matrix(df)  :
    df  = df.sort_values(by = "Man of the Match" ,ascending = True)
    classes = df["Man of the Match"].unique().tolist()
    classes
    
    class_code  = {classes[k] : k for k in range(2)}
    class_code

    color_vals = [class_code[cl] for cl in df["Man of the Match"]]
    color_vals

    pl_colorscale = "Portland"

    pl_colorscale

    text = [df.loc[k,"Man of the Match"] for k in range(len(df))]
    text

    trace = go.Splom(dimensions = [dict(label  = "Attempts",
                                       values = df["Attempts"]),
                                  dict(label  = 'Blocked',
                                       values = df['Blocked']),
                                  dict(label  = 'Corners',
                                       values = df['Corners'])],
                     text = text,
                     marker = dict(color = color_vals,
                                   colorscale = pl_colorscale,
                                   size = 3,
                                   showscale = False,
                                   line = dict(width = .1,
                                               color='rgb(230,230,230)'
                                              )
                                  )
                    )
    axis = dict(showline  = True,
                zeroline  = False,
                gridcolor = "#fff",
                ticklen   = 4
               )
    
    layout = go.Layout(dict(title  =  "Scatter plot matrix for Numerical columns for attrition",
                            autosize = False,
                            height = 800,
                            width  = 800,
                            dragmode = "select",
                            hovermode = "closest",
                            plot_bgcolor  = 'rgba(240,240,240, 0.95)',
                            xaxis1 = dict(axis),
                            yaxis1 = dict(axis),
                            xaxis2 = dict(axis),
                            yaxis2 = dict(axis),
                            xaxis3 = dict(axis),
                            yaxis3 = dict(axis),
                           )
                      )
    data   = [trace]
    fig = go.Figure(data = data,layout = layout )
    py.iplot(fig)
    
 # for scatter plot matrix
scatter_matrix(df)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
df = df.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)
df = df.fillna(df.mean())
df.tail()
features = df.drop(['Man of the Match'], axis = 1)
target = df['Man of the Match']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state =0)
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)
xgb = XGBClassifier(n_estimators=100)
training_start = time.perf_counter()
xgb.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(X_test)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))
xgb = XGBClassifier(n_estimators=100)
scores = cross_val_score(xgb, X_train, y_train, cv=2, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
xgb = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, 
                        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)

xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)

accuracy = (preds == y_test).sum().astype(float) / len(preds)*100

print("XGBoost's prediction accuracy WITH optimal hyperparameters is: %3.2f" % (accuracy))
predictions = cross_val_predict(xgb, features, target, cv=3)
confusion_matrix(target, predictions)
print("Precision:", precision_score(target, predictions, average='micro'))
print("Recall:",recall_score(target, predictions, average='micro'))
print("F1 Score: ", f1_score(target, predictions, average = 'micro'))
print(xgb.feature_importances_)
importances = xgb.feature_importances_
weights = pd.Series(importances,
                 index=features.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')
lgb = LGBMClassifier(n_estimators=100)
training_start = time.perf_counter()
lgb.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = lgb.predict(X_test)
prediction_end = time.perf_counter()
acc_lgb = (preds == y_test).sum().astype(float) / len(preds)*100
lgb_train_time = training_end-training_start
lgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_lgb))
print("Time consumed for training: %4.3f" % (lgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (lgb_prediction_time))
scores = cross_val_score(lgb, X_train, y_train, cv=2, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
lgb = LGBMClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, 
                        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)

lgb.fit(X_train, y_train)
preds = lgb.predict(X_test)

accuracy = (preds == y_test).sum().astype(float) / len(preds)*100

print("XGBoost's prediction accuracy WITH optimal hyperparameters is: %3.2f" % (accuracy))
predictions = cross_val_predict(lgb, features, target, cv=3)
confusion_matrix(target, predictions)
print("Precision:", precision_score(target, predictions, average='micro'))
print("Recall:",recall_score(target, predictions, average='micro'))
print("F1 Score: ", f1_score(target, predictions, average = 'micro'))
print(lgb.feature_importances_)
importances = lgb.feature_importances_
weights = pd.Series(importances,
                 index=features.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')
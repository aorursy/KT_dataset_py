# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Package
import os # accessing directory structure
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

import itertools


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
df = pd.read_csv('../input/us-births-2018/US_births(2018).csv', 
                 low_memory=False)
df.head()
df.drop(df.columns.difference(
        ['DBWT','CIG_0','DOB_YY','SEX','MAGER','ILP_R','PRECARE'
        ,'PWgt_R','M_Ht_In' ]), 1, inplace=True)
df.head()
df.groupby('ILP_R').mean()['DBWT'].plot(kind='bar')
plt.show()
df.groupby('M_Ht_In').mean()['DBWT'].plot(kind='bar')
plt.show()
df.groupby('PWgt_R').mean()['DBWT'].plot(kind='bar')
plt.show()
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df.index[::-1],
        x=df.values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

# Cigaret
df1 = df.groupby('CIG_0')['DBWT'].agg(['count', 'mean'])
df1.columns = ["count", "mean"]
df1 = df1.sort_values(by="count", ascending=False)
trace1 = horizontal_bar_chart(df1["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace2 = horizontal_bar_chart(df1["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Mother’s Height in Total Inches
df1 = df.groupby('M_Ht_In')['DBWT'].agg(['count', 'mean'])
df1.columns = ["count", "mean"]
df1 = df1.sort_values(by="count", ascending=False)
trace3 = horizontal_bar_chart(df1["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace4 = horizontal_bar_chart(df1["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# SEX
df1 = df.groupby('SEX')['DBWT'].agg(['count', 'mean'])
df1.columns = ["count", "mean"]
df1 = df1.sort_values(by="count", ascending=False)
trace5 = horizontal_bar_chart(df1["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace6 = horizontal_bar_chart(df1["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Interval Since Last Pregnancy Recode
df1 = df.groupby('ILP_R')['DBWT'].agg(['count', 'mean'])
df1.columns = ["count", "mean"]
df1 = df1.sort_values(by="count", ascending=False)
trace7 = horizontal_bar_chart(df1["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace8 = horizontal_bar_chart(df1["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Mother’s Single Years of Age
df1 = df.groupby('MAGER')['DBWT'].agg(['count', 'mean'])
df1.columns = ["count", "mean"]
df1 = df1.sort_values(by="count", ascending=False)
trace9 = horizontal_bar_chart(df1["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace10 = horizontal_bar_chart(df1["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Pre-pregnancy Weight Recode
df1 = df.groupby('PWgt_R')['DBWT'].agg(['count', 'mean'])
df1.columns = ["count", "mean"]
df1 = df1.sort_values(by="count", ascending=False)
trace11 = horizontal_bar_chart(df1["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace12 = horizontal_bar_chart(df1["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Month Prenatal Care Began
df1 = df.groupby('PRECARE')['DBWT'].agg(['count', 'mean'])
df1.columns = ["count", "mean"]
df1 = df1.sort_values(by="count", ascending=False)
trace13 = horizontal_bar_chart(df1["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace14 = horizontal_bar_chart(df1["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Creating two subplots
fig = tools.make_subplots(rows=7, cols=2, vertical_spacing=0.04, 
                          subplot_titles=["Ciggrarets - Count","Baby weight - mean",
                                          "Mother’s Height in Total Inches - Count","Baby weight - mean","SEX - Count",
                                          "Baby weight - mean","Interval Since Last Pregnancy Recode - Count",
                                          "Baby weight - mean", "Mother’s Single Years of Age - Count",
                                          "Baby weight - mean","Pre-pregnancy Weight Recode - Count","Baby weight - mean",
                                          "Month Prenatal Care Began","Baby weight - mean"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 3, 1)
fig.append_trace(trace6, 3, 2)
fig.append_trace(trace7, 4, 1)
fig.append_trace(trace8, 4, 2)
fig.append_trace(trace9, 5, 1)
fig.append_trace(trace10, 5, 2)
fig.append_trace(trace11, 6, 1)
fig.append_trace(trace12, 6, 2)
fig.append_trace(trace13, 7, 1)
fig.append_trace(trace14, 7, 2)

fig['layout'].update(height=1800, width=1000, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots')
#Check missing value
df.isna().sum()
mymap = {'M':1, 'F':2}
df1=df.applymap(lambda s: mymap.get(s) if s in mymap else s)
df1.head()
train = df1.sample(frac=0.8,random_state=0)
test = df1.drop(train.index)
test = test.drop(['DBWT'], axis=1)
train.info()
test.info()
train_stats = train.describe()
train_stats.pop("DBWT")
train_stats = train_stats.transpose()
train_stats
import warnings
warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove('DBWT')

mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_new = np.matrix(train.drop('DBWT',axis = 1))
mat_y = np.array(train.DBWT).reshape((3041227,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)

train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)

train.head()

# List of features
COLUMNS = col_train
FEATURES = col_train_bis
LABEL = "DBWT"

# Columns for tensorflow
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# Training set and Prediction set with the features to predict
training_set = train[COLUMNS]
prediction_set = train.DBWT

# Train and Test 
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
training_set.head()

# Training for submission
training_sub = training_set[col_train]
# Same thing but for the test set
y_test = pd.DataFrame(y_test, columns = [LABEL])
testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)
testing_set.head()
# Model
tf.logging.set_verbosity(tf.logging.ERROR)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])#,
                                         #optimizer = tf.train.GradientDescentOptimizer( learning_rate= 0.1 ))
# Reset the index of training
training_set.reset_index(drop = True, inplace =True)
def input_fn(data_set, pred = False):
    
    if pred == False:
        
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        
        return feature_cols, labels

    if pred == True:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        
        return feature_cols
# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)
# Evaluation on the test set created by train_test_split
ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
# Display the score on the testing set
# 0.002X in average
loss_score1 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))
# Predictions
y = regressor.predict(input_fn=lambda: input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
predictions = pd.DataFrame(prepro_y.inverse_transform(np.array(predictions).reshape(434,1)),columns = ['Prediction'])
reality = pd.DataFrame(prepro.inverse_transform(testing_set), columns = [COLUMNS]).SalePrice
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

fig, ax = plt.subplots(figsize=(50, 40))

plt.style.use('ggplot')
plt.plot(predictions.values, reality.values, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()

def to_submit(pred_y,name_out):
    y_predict = list(itertools.islice(pred_y, test.shape[0]))
    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).reshape(len(y_predict),1)), columns = ['SalePrice'])
    y_predict = y_predict.join(ID)
    y_predict.to_csv(name_out + '.csv',index=False)
    
to_submit(y_predict, "submission_continuous")
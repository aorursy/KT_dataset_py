# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.head(3)
df.describe()
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='Positive', x=['Positive'], y=[df[df["SARS-Cov-2 exam result"] == 'positive'].shape[0]], marker=dict(
        color='rgb(99,154,103)'
    )),
    go.Bar(name='Negative', x=['Negative'], y=[df[df["SARS-Cov-2 exam result"] == 'negative'].shape[0]], marker=dict(
        color='rgb(241,146,146)',
    ))
])

fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
        'text': "Cases Results Distribution",
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="",
    yaxis_title="NUmber of Cases",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()
df_positive = df[df["SARS-Cov-2 exam result"] == 'positive']

number_by_age_quantile = df_positive['Patient age quantile'].value_counts().sort_index().reset_index().rename(
    columns={
        'index': 'age_quantile',
        'Patient age quantile': 'number_by_quantile'
    }
)

import plotly.express as px
fig = px.bar(number_by_age_quantile, x='age_quantile', y='number_by_quantile')
fig.update_layout(
    title={
        'y':0.95,
        'x':0.5,
        'text': "Quantile age Distribution of positive COVID cases",
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Age Quantile",
    yaxis_title="Cases by Quantile",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()
df_to_train = df_positive.append(
    df[df["SARS-Cov-2 exam result"] == 'negative'].sample(
        df_positive.shape[0]
    ),
    sort=False
)
for col in df_to_train.columns:
    df_to_train[col] = df_to_train[col].replace('not_detected', 0).copy()
    df_to_train[col] = df_to_train[col].replace('detected', 1).copy()
    df_to_train[col] = df_to_train[col].replace('negative', 0).copy()
    df_to_train[col] = df_to_train[col].replace('positive', 1).copy()
    df_to_train[col] = df_to_train[col].replace('absent', 0).copy()
    df_to_train[col] = df_to_train[col].replace('present', 1).copy()
    
    if (df_to_train[col].isna().sum() == df_to_train.shape[0]) | (col[:5] == 'Urine'):
        df_to_train.drop(col, axis=1, inplace=True)
    elif df_to_train[col].isna().sum() > 0:
        try:
            df_to_train[col] = df_to_train[col].astype(float).copy()
            df_to_train[col].fillna(df_to_train[col].mean(), inplace=True)
            df_to_train[col].fillna(0.0, inplace=True)
        except:
            df_to_train.drop(col, axis=1, inplace=True)
        
df_to_train = df_to_train.sample(frac=1)
n_train = int(df_to_train.shape[0] * 0.8)
x_train = df_to_train.iloc[:n_train].drop('SARS-Cov-2 exam result', axis=1)
y_train = df_to_train.iloc[:n_train]['SARS-Cov-2 exam result'].values
x_test = df_to_train.iloc[n_train:].drop('SARS-Cov-2 exam result', axis=1)
y_test = df_to_train.iloc[n_train:]['SARS-Cov-2 exam result'].values
from sklearn.ensemble import RandomForestClassifier

regr = RandomForestClassifier(n_estimators=1000, random_state=0)

regr.fit(x_train.iloc[:,1:], y_train)

# print(regr.feature_importances_)
from sklearn import metrics

preds = regr.predict(x_test.iloc[:,1:])

print("Evaluating Random Forest model\n")
print('Precision Score:', metrics.precision_score(y_test, preds))  
print('Recall Score:', metrics.recall_score(y_test, preds))  
print('ROC AUC Score:', metrics.roc_auc_score(y_test, preds))  
print('F1 Score:', metrics.f1_score(y_test, preds))  
print('Average Precision:', metrics.average_precision_score(y_test, preds))  
print('Accuracy:', metrics.accuracy_score(y_test, preds))  
def plot_confusion_matrix(cm):
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt='g',cmap=sns.color_palette("GnBu_d"), cbar=False, linewidths=1, linecolor='black');
    ax.set_xlabel('Predict');ax.set_ylabel('True'); 
    ax.set_title('Confusion matrix'); 
    ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative','Positive']);
plot_confusion_matrix(metrics.confusion_matrix(y_test, preds))
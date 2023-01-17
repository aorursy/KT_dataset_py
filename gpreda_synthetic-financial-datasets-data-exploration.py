import numpy as np 

import pandas as pd

import os

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import seaborn as sns

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
data_df = pd.read_csv("/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv")
print(f"Data shape: {data_df.shape}")
data_df.head()
data_df.info()
data_df.describe()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(data_df)
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(data_df)
def plot_count(df, feature, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))

    total = float(len(df))

    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set3')

    plt.title(title)

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.4f}%'.format(100*height/total),

                ha="center") 

    plt.show()

plot_count(data_df, 'type', 'Distribution of type (count & percent)', size=2.5)
plot_count(data_df, 'isFraud', 'Distribution of `isFraud` (count & percent)', size=2.5)
plot_count(data_df, 'isFlaggedFraud', 'Distribution of `isFlaggedFraud` (count & percent)', size=2.5)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="isFraud", y="step", hue="isFraud",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFraud", y="step", hue="isFraud",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="isFlaggedFraud", y="step", hue="isFlaggedFraud",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFlaggedFraud", y="step", hue="isFlaggedFraud",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="isFraud", y="step", hue="type",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFraud", y="step", hue="type",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="isFlaggedFraud", y="step", hue="type",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFlaggedFraud", y="step", hue="type",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="isFraud", y="amount", hue="type",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFraud", y="amount", hue="type",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="isFraud", y="oldbalanceOrg", hue="type",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFraud", y="oldbalanceOrg", hue="type",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="isFraud", y="newbalanceOrig", hue="type",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFraud", y="newbalanceOrig", hue="type",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="isFraud", y="oldbalanceDest", hue="type",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFraud", y="oldbalanceDest", hue="type",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="isFraud", y="newbalanceDest", hue="type",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="isFraud", y="newbalanceDest", hue="type",data=data_df, palette="PRGn",showfliers=False)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
X = data_df.drop(['isFraud', 'isFlaggedFraud'], axis=1)

y = data_df.isFraud
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)
categorical_features_indices = np.where(X.dtypes != np.float)[0]
clf = CatBoostClassifier(iterations=500,

                             learning_rate=0.02,

                             depth=12,

                             eval_metric='AUC',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 20,

                             od_wait=25)
clf.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
preds = clf.predict(X_validation)
cm = pd.crosstab(y_validation.values, preds, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
print(f"ROC-AUC score: {roc_auc_score(y_validation.values, preds)}")
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

data_red_df = pd.read_csv("/kaggle/input/banksim1/bsNET140513_032310.csv")

data_df = pd.read_csv("/kaggle/input/banksim1/bs140513_032310.csv")
print(data_red_df.shape)
print(data_df.shape)
data_df.head()
data_red_df.head()
print(f"unique customers: {data_df.customer.nunique()}")

print(f"unique merchants: {data_df.merchant.nunique()}")

print(f"unique age: {data_df.age.nunique()}")

print(f"unique gender: {data_df.gender.nunique()}")

print(f"unique zipCode Origin: {data_df.zipcodeOri.nunique()}")

print(f"unique zipCode Merchant: {data_df.zipMerchant.nunique()}")

print(f"unique category: {data_df.category.nunique()}")

print(f"unique amount: {data_df.amount.nunique()}")

print(f"unique fraud: {data_df.fraud.nunique()}")
print(f"unique Source: {data_red_df.Source.nunique()}")

print(f"unique Target: {data_red_df.Target.nunique()}")

print(f"unique Weight: {data_red_df.Weight.nunique()}")

print(f"unique typeTrans: {data_red_df.typeTrans.nunique()}")

print(f"unique fraud: {data_red_df.fraud.nunique()}")
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

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count(data_df, 'age', 'Distribution of age (count & percent)', size=2.5)
plot_count(data_df, 'gender', 'Distribution of gender (count & percent)')
plot_count(data_df, 'category', 'Distribution of category (count & percent)', size=4)
temp = data_df["fraud"].value_counts()

df = pd.DataFrame({'fraud': temp.index,'values': temp.values})



trace = go.Bar(

    x = df['fraud'],y = df['values'],

    name="Payments fraud - data unbalance (Not fraud = 0, Fraud = 1)",

    marker=dict(color="Red"),

    text=df['values']

)

data = [trace]

layout = dict(title = 'Payments Fraud - data unbalance (Not fraud = 0, Fraud = 1)',

          xaxis = dict(title = 'Fraud', showticklabels=True), 

          yaxis = dict(title = 'Number of transactions'),

          hovermode = 'closest',width=600

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='class')
def plot_distplot_grouped(df, feature):

    classes = list(df[feature].unique())

    print(classes)

    group_labels = []     

    hist_data = []

    for item in classes:

        crt_class = df.loc[df[feature]==item]["step"]

        group_labels.append(f"{item}")

        hist_data.append(crt_class)

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)

    fig['layout'].update(title=f'Payments Transactions Time Density Plot - grouped by `{feature}`', xaxis=dict(title='Time [step]'))

    iplot(fig, filename='dist_only')     
plot_distplot_grouped(data_df, 'fraud')
plot_distplot_grouped(data_df, 'age')
plot_distplot_grouped(data_df, 'gender')
plot_distplot_grouped(data_df, 'category')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="fraud", y="amount", hue="fraud",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="fraud", y="amount", hue="fraud",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

s = sns.boxplot(ax = ax1, x="gender", y="amount", hue="gender",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="gender", y="amount", hue="gender",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,6))

s = sns.boxplot(ax = ax1, x="age", y="amount", hue="age",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="age", y="amount", hue="age",data=data_df, palette="PRGn",showfliers=False)

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16,12))

s = sns.boxplot(ax = ax1, x="category", y="amount", hue="category",data=data_df, palette="PRGn",showfliers=True)

s = sns.boxplot(ax = ax2, x="category", y="amount", hue="category",data=data_df, palette="PRGn",showfliers=False)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
X = data_df.drop(['fraud'], axis=1)

y = data_df.fraud
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
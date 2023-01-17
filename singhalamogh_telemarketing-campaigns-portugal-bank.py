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
import time



# importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import special, stats



# preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler





# model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold, cross_val_score





# SMOTe

from imblearn.over_sampling import SMOTE



# models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier





# metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, precision_recall_curve 

from sklearn.metrics import recall_score, roc_curve, roc_auc_score, precision_recall_curve, auc, plot_confusion_matrix



# ensemble

from xgboost import XGBClassifier



# warnings

import warnings

warnings.filterwarnings("ignore")



# style

import matplotlib.style as style

style.use('fivethirtyeight')
df = pd.read_csv('/kaggle/input/bank-marketing-campaigns-dataset/bank-additional-full.csv', sep=';')
df.shape
df.head()
df.info()
# Check for null values if any



# This method shows the count of null values, percent and dataTypes



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
missing_data(df)
df['y'].value_counts()
def plot_pie(dataset, target, size=(7,7)):

    return dataset[target].value_counts().plot.pie(autopct = '%1.1f%%', figsize=size)



plot_pie(df, 'y')
# create a new variable `target` that takes 1 for `yes` else 0

df['target'] = np.where(df['y'].isin(['yes']), 1, 0)



df.head()
df['target'].mean()
sns.countplot(df['target'])
df['education'].value_counts()
# We will plot the relationship of `education` with `target`

# estimator is mean to show the likelihood of taking up the loan

def barplot_mean(x, y, df, hue=None, order=None, hue_order=None):

    print(df.groupby(x)[y].mean())

    uniqs = df[x].nunique()

    

    if uniqs > 4:

        plt.figure(figsize=(16,4))

        

    sns.barplot(x=x, y=y, data=df, estimator=np.mean, hue=hue, order=order, hue_order=hue_order)

    plt.show()



barplot_mean('education', 'target', df)
# We will group the basic education groups into one

basic_grps = ['basic.4y', 'basic.6y', 'basic.9y']



df['education'] = np.where(df['education'].isin(basic_grps), 'Basic', df['education'])

df.head()
barplot_mean('education', 'target', df)
df['education'] = np.where(df['education'].isin(['illiterate']), 'unknown', df['education'])

barplot_mean('education', 'target', df)
barplot_mean('day_of_week', 'target', df)
barplot_mean('job', 'target', df)
df['job'] = np.where(df['job'].isin(['unknown']), 'unemployed', df['job'])

barplot_mean('job', 'target', df)
barplot_mean('marital', 'target', df)
barplot_mean('default', 'target', df)
print(df.groupby('default')['target'].sum())

print("*"*30)

# if we look at the count - we do have 3 cases of people who have `defaulted`

print(df.groupby('default')['target'].count())



# But we do have cases of `unknown`. 

# We can go ahead and group them in `yes` 

# because we are not sure which category they belong to
df['default'] = np.where(df['default'].isin(['unknown']), 'yes', df['default'])

barplot_mean('default', 'target', df)
barplot_mean('housing', 'target', df)
barplot_mean('loan', 'target', df)
barplot_mean('contact', 'target', df)
barplot_mean('month', 'target', df)
qtr1 = ['jan', 'feb', 'mar']

qtr2 = ['apr', 'may', 'jun']

qtr3 = ['jul', 'aug', 'sep']

qtr4 = ['oct', 'nov', 'dec']



df['qtr'] = np.where(df['month'].isin(qtr1), 'Q1', 

                                       np.where(df['month'].isin(qtr2), 'Q2', 

                                       np.where(df['month'].isin(qtr3), 'Q3',

                                       np.where(df['month'].isin(qtr4), 'Q4', 0)

                                       )))

df['qtr'].value_counts()
barplot_mean('qtr', 'target', df, order=["Q1","Q2","Q3","Q4"])

# `order` as the name suggest orders the graph in similar fashion as the input list

# here we pass the order as per the quarters
barplot_mean('qtr', 'target', df, hue='contact', order=["Q1","Q2","Q3","Q4"])
df[df['contact'] == "cellular"].groupby('qtr')['target'].mean()



# avg. likelihood across qtr

# Q1    0.505495

# Q2    0.091349

# Q3    0.112053

# Q4    0.163967
barplot_mean('poutcome', 'target', df)
df['poutcome'] = np.where(df['poutcome'].isin(['nonexistent', 'failure']), 0, 1)

barplot_mean('poutcome', 'target', df)
barplot_mean('qtr', 'target', df, hue='poutcome', order=["Q1","Q2","Q3","Q4"])
df[df['poutcome'] == 1].groupby('qtr')['target'].mean()



# avg. likelihood across qtr

# Q1    0.505495

# Q2    0.091349

# Q3    0.112053

# Q4    0.163967
df['age_rank'] = pd.qcut(df['age'].rank(method='first').values, 5, duplicates='drop').codes+1

df['age_rank'].value_counts()



# we have divided age into 5 ranks thery distributing 20% data in each rank

# we can now see if there is any trend with respect to age on target
barplot_mean('age_rank', 'target', df)
barplot_mean('age_rank', 'target', df, hue='qtr', hue_order=["Q1","Q2","Q3","Q4"])
df['duration_rank'] = pd.qcut(df['duration'].rank(method='first').values, 5, duplicates='drop').codes+1

df['duration_rank'].value_counts()
barplot_mean('duration_rank', 'target', df)
df['campaign_rank'] = pd.qcut(df['campaign'].rank(method='first').values, 5, duplicates='drop').codes+1

df['campaign_rank'].value_counts()
barplot_mean('campaign_rank', 'target', df)
print(df.groupby('campaign_rank')['campaign'].min())

print("*"*30)

print(df.groupby('campaign_rank')['campaign'].mean())

print("*"*30)

print(df.groupby('campaign_rank')['campaign'].max())
df['pdays_rank'] = pd.qcut(df['pdays'].rank(method='first').values, 5, duplicates='drop').codes+1

df['pdays_rank'].value_counts()
barplot_mean('pdays_rank', 'target', df)
print(df.groupby('pdays_rank')['pdays'].min())

print("*"*30)

print(df.groupby('pdays_rank')['pdays'].mean())

print("*"*30)

print(df.groupby('pdays_rank')['pdays'].max())
df['prev_rank'] = pd.qcut(df['previous'].rank(method='first').values, 5, duplicates='drop').codes+1

df['prev_rank'].value_counts()
barplot_mean('prev_rank', 'target', df)
barplot_mean('prev_rank', 'target', df, hue='qtr', hue_order=["Q1","Q2","Q3","Q4"])
df['emp.var.rate_rank'] = pd.qcut(df['emp.var.rate'].rank(method='first').values, 5, duplicates='drop').codes+1

barplot_mean('emp.var.rate_rank', 'target', df)
df['cons.price.idx_rank'] = pd.qcut(df['cons.price.idx'].rank(method='first').values, 5, duplicates='drop').codes+1

barplot_mean('cons.price.idx_rank', 'target', df)
df['cons.conf.idx_rank'] = pd.qcut(df['cons.conf.idx'].rank(method='first').values, 5, duplicates='drop').codes+1

barplot_mean('cons.conf.idx_rank', 'target', df)
df['euribor3m_rank'] = pd.qcut(df['euribor3m'].rank(method='first').values, 5, duplicates='drop').codes+1

barplot_mean('euribor3m_rank', 'target', df)
df['nr.employed_rank'] = pd.qcut(df['nr.employed'].rank(method='first').values, 5, duplicates='drop').codes+1

barplot_mean('nr.employed_rank', 'target', df)
df['nr.employed_rank'] = pd.qcut(df['nr.employed'].rank(method='first').values, 10, duplicates='drop').codes+1

barplot_mean('nr.employed_rank', 'target', df)
df['nr.employed_rank'] = np.where(df['nr.employed_rank'].isin(['1']), 'A', 

                                       np.where(df['nr.employed_rank'].isin(['2']), 'B', 'C'))

                                                                            

df['nr.employed_rank'].value_counts()
barplot_mean('nr.employed_rank', 'target', df)
df.info()
# We are not considering education, job, day_of_week, housing, loan

cols_cat = ['default', 'contact', 'poutcome', 'nr.employed_rank']



# We are not considering age, duration, cons.conf.idx 

cols_num = ['campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'euribor3m'] #, 'duration']



# NOTE: duration is added to improve the ROC score
# dummy encoding categorical variable

# ref: https://stackoverflow.com/questions/36631163/what-are-the-pros-and-cons-between-get-dummies-pandas-and-onehotencoder-sciki

cols_cat_dummy = pd.get_dummies(df[cols_cat], drop_first=True)

cols_cat_dummy.head()
X_all = pd.concat([df[cols_num], cols_cat_dummy], axis=1, join='inner')

X_all.head()
# Assigning X and Y

X = X_all

y = df['target']
# Train-Val split 75-25

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=101, test_size=0.30)



print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
lr = LogisticRegression()

lr.fit(X_train, y_train)



# TODO: Do a grid search to explore best parameters

dt = DecisionTreeClassifier(criterion="gini", random_state=101, max_depth=7, min_samples_leaf=5)

dt.fit(X_train, y_train)



rf_1000 = RandomForestClassifier(n_estimators=1000, random_state=101, criterion="gini", max_features="auto", max_depth=2)

rf_1000.fit(X_train, y_train)
y_pred = lr.predict(X_val)

print("Accuracy of logistic regression on test set {:.2f}".format(lr.score(X_val, y_val)))
y_pred_tree = dt.predict(X_val)

print("Accuracy of decision tree on test set {:.2f}".format(dt.score(X_val, y_val)))
y_pred_rf = rf_1000.predict(X_val)

print("Accuracy of random forest on test set {:.2f}".format(rf_1000.score(X_val, y_val)))
rf_1000_train_score = rf_1000.score(X_train, y_train)

rf_1000_test_score = rf_1000.score(X_val, y_val)





print("Training Score:", rf_1000_train_score)

print("Test Score:", rf_1000_test_score)
sns.set_style({'axes.grid' : False})

# logistic regression

plot_confusion_matrix(lr, X_val, y_val)

print(classification_report(y_val, y_pred))
# decision tree

print(classification_report(y_val, y_pred_tree))

plot_confusion_matrix(dt, X_val, y_val)
# random forest

print(classification_report(y_val, y_pred_rf))

plot_confusion_matrix(rf_1000, X_val, y_val)
lr_roc_auc = roc_auc_score(y_val, lr.predict(X_val))

dt_roc_auc = roc_auc_score(y_val, dt.predict(X_val))

rf_roc_auc = roc_auc_score(y_val, rf_1000.predict(X_val))



fpr, tpr, thresholds = roc_curve(y_val, lr.predict_proba(X_val)[:, 1])

fpr, tpr, thresholds = roc_curve(y_val, dt.predict_proba(X_val)[:, 1])

fpr, tpr, thresholds = roc_curve(y_val, rf_1000.predict_proba(X_val)[:, 1])



plt.figure()



plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.2f' % lr_roc_auc)

plt.plot(fpr, tpr, 'r', label = 'DT AUC = %0.2f' % dt_roc_auc)

plt.plot(fpr, tpr, 'g', label = 'RF AUC = %0.2f' % rf_roc_auc)



plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.savefig('LR DT RF ROC Curve')

plt.show()



### The score is coming in the range of late 50s, lets try to improve the model by incorporating `duration`



### After using `duration`, we are able to get the score upto `0.77` for decision tree
# Ranking the probabilities from the logistic regression model



y_pred_prob = lr.predict_proba(X)[:,1]

df['y_pred_P'] = pd.DataFrame(y_pred_prob)

df['P_rank'] = pd.qcut(df['y_pred_P'].rank(method='first').values, 10, duplicates='drop').codes+1

df.groupby('P_rank')['target'].mean()



# The highest rank has a likelihood of 48.28 percent (~ 4.3 times better than the average)
# Ranking the probabilities from the logistic regression model



y_pred_prob_dtree = dt.predict_proba(X)[:,1]

df['y_pred_P_dtree'] = pd.DataFrame(y_pred_prob_dtree)

df['P_rank_dtree'] = pd.qcut(df['y_pred_P_dtree'].rank(method='first').values, 10, duplicates='drop').codes+1

df.groupby('P_rank_dtree')['target'].mean()



# The highest rank has a likelihood of 51.56 percent (~ 4.6 times better than the average)
df.to_csv('telemarketing_model_scored_file.csv')
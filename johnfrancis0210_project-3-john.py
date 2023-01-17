import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

hdt_train = pd.read_csv('../input/health-diagnostics-train.csv', na_values='#NULL!')
hdt_test = pd.read_csv('../input/health-diagnostics-test.csv', na_values='#NULL!')
hdt_train.head()
hdt_test.head()
hdt_train.shape
hdt_train.dtypes
hdt_train.income.sort_values().unique()
hdt_train.maternal.sort_values().unique()
hdt_train['fam-history'].sort_values().unique()
hdt_train['mat-illness-past'].sort_values().unique()
hdt_train['suppl'].sort_values().unique()
hdt_train['mat-illness'].sort_values().unique()
hdt_train['meds'].sort_values().unique()
hdt_train['env'].sort_values().unique()
hdt_train['lifestyle'].sort_values().unique()
hdt_train['target'].sort_values().unique()
hdt_train.isna().sum()
hdt_train.dropna(inplace=True)
hdt_train.isna().sum()
hdt_train.columns
# plt.matshow(hdt_train.corr())
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

plot_corr(hdt_train)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(penalty='l1', class_weight='balanced')

feature_cols = ['income', 'fam-history', 'mat-illness-past', 'suppl',
       'mat-illness', 'meds', 'env', 'lifestyle']
X = hdt_train[feature_cols]
y = hdt_train.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test) # prediction via Log Reg
hdt_train['target'].sum()
print("Baseline accuracy is " + str(1 - 59/32670))
metrics.accuracy_score(y_test,y_pred)
metrics.confusion_matrix(y_test,y_pred)
metrics.roc_auc_score(y_test,y_pred)
name = hdt_train.columns.drop('target')

coef = logreg.coef_[0]

pd.DataFrame([name,coef],index = ['Name','Coef']).transpose()

logreg1 = logreg

feature_cols1 = feature_cols
X = hdt_train[feature_cols1]
y = hdt_train.target

logreg1.fit(X, y)
y_pred1 = logreg1.predict(X) # prediction via Log Reg
metrics.accuracy_score(y,y_pred1)
metrics.confusion_matrix(y,y_pred1)
metrics.roc_auc_score(y,y_pred1)
logreg1
hdt_test.head()
hdt_test.isna().sum()
hdt_test.fillna(value=0, inplace=True)
feature_cols2 = feature_cols1
X = hdt_test[feature_cols2]

y_pred2 = logreg1.predict(X) # prediction via Log Reg
hdt_test['target']=y_pred2
hdt_test
hdt_solution = pd.DataFrame(hdt_test['target'])
hdt_solution.index.name = 'index'
hdt_solution.head()
hdt_solution.to_csv('jf20181013-4.csv')

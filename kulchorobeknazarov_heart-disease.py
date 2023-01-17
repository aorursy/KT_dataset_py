import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df=pd.read_csv('../input/heart-disease/heart.csv')

df.describe(include='all')

df.isnull().sum()
y=df['target']

x=df.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()

clf.fit(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.model_selection import cross_val_score

validation=cross_val_score(clf,x_test,y_test,cv=5)

validation
from sklearn.metrics import roc_curve

y_preds=clf.predict_proba(x_test)

y_preds_positive=y_preds[:,1]

y_preds_positive
from sklearn.metrics import roc_auc_score

roc_score=roc_auc_score(y_test,y_preds_positive)

roc_score
fpr,tpr,thresholds=roc_curve(y_test,clf.predict(x_test))
def plot_roc_curve(fpr,tpr):

    plt.plot(fpr,tpr,color='orange',label='ROC',)

    plt.plot([0,1],[0,1],color='darkblue',label='Guessing',linestyle="--")

    plt.legend()

    plt.show()

    

plot_roc_curve(fpr,tpr)
from sklearn.metrics import confusion_matrix

conf=confusion_matrix(y_test,clf.predict(x_test))

conf
def plot_conf_matrix(conf):

    f,ax=plt.subplots(figsize=(4,4))

    ax=sns.heatmap(conf,annot=True,cbar=False)



plot_conf_matrix(conf)
y_test=y_test.reset_index(drop=True)


df_pf=pd.DataFrame()

df_pf['Actual']=y_test

df_pf['Predicted']=clf.predict(x_test)



pd.options.display.max_rows=999

df_pf
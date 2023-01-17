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
from sklearn.feature_selection import SelectKBest,chi2

from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import roc_curve,roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sn 



framingham_df = pd.read_csv(r'/kaggle/input/framingham-heart-study-dataset/framingham.csv')

framingham_df.head()
framingham_df.dtypes
#The proportion of missing values of each variable

framingham_df.isnull().sum(axis=0)/framingham_df.shape[0]
#Proportion of rows with missing observations

framingham_df.isnull().any(axis=1).sum()/framingham_df.shape[0]
framingham_df.dropna(inplace=True)

framingham_df.head()
framingham_df.groupby('TenYearCHD').size()
X = framingham_df.drop('TenYearCHD',axis=1)

y = framingham_df['TenYearCHD']
#select best features

best_features = SelectKBest(score_func=chi2, k=len(X.columns))

fit = best_features.fit(X, y)



df_scores = pd.DataFrame(fit.scores_)

df_columns = pd.DataFrame(X.columns)



df_feature_scores = pd.concat([df_columns, df_scores], axis=1)

df_feature_scores.columns = ['Feature', 'Score']

df_feature_scores.sort_values(by='Score',ascending=False,inplace=True)

df_feature_scores

select_features = df_feature_scores[df_feature_scores['Score'] > 100]['Feature'].tolist()

select_features
X = framingham_df[select_features]

y = framingham_df['TenYearCHD']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)

# accuracy_score(y_test,y_pred)

print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="Greens")
y_pred_prob_yes = logreg.predict_proba(x_test)[:,1]

auc = roc_auc_score(y_test, y_pred_prob_yes)

print('auc:',auc)

fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob_yes)

plt.plot(fpr,tpr,linewidth=2,label='ROC curve: AUC={0:0.2f}'.format(auc))

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.grid(True)

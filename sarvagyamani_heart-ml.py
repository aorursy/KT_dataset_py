# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pickle

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

#from sklearn.manifold import TSNE

from sklearn.utils import shuffle

import matplotlib.gridspec as gridspec

from sklearn import datasets, linear_model

import seaborn as sn

import statsmodels.api as sm

import scipy.stats as st

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
hrt = pd.read_csv('../input/heart.csv')

X = hrt.iloc[:,:-1]

y = hrt['target']
hrt.target.value_counts()
hrt.describe()
corr = hrt.corr()
fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm')

fig.colorbar(cax)

ticks = np.arange(0,len(hrt.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(hrt.columns)

ax.set_yticklabels(hrt.columns)

plt.show()


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)

cols=hrt.columns[:-1]

model=sm.Logit(hrt.target,hrt[cols])

result=model.fit()

result.summary()
def back_feature_elem (data_frame,dep_var,col_list):

    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest

    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""



    while len(col_list)>0 :

        model=sm.Logit(dep_var,data_frame[col_list])

        result=model.fit(disp=0)

        largest_pvalue=round(result.pvalues,3).nlargest(1)

        if largest_pvalue[0]<(0.05):

            return result

            break

        else:

            col_list=col_list.drop(largest_pvalue.index)



result=back_feature_elem(hrt,hrt.target,cols)
params = np.exp(result.params)

conf = np.exp(result.conf_int())

conf['OR'] = params

pvalue=round(result.pvalues,3)

conf['pvalue']=pvalue

conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']

print ((conf))

new_dat = hrt.drop(['age','trestbps','chol','fbs','restecg','slope'],axis = 1 )

y = hrt['target']

X = new_dat.iloc[:,:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 5, criterion = 'entropy')

model.fit(X_train, y_train)
model.score(X_train,y_train)
model.score(X_test,y_test)
prediction_rf= model.predict(X_test)
from sklearn import metrics

results_rf=metrics.classification_report(y_true=y_test, y_pred=prediction_rf)

print(results_rf)



rf_2=metrics.confusion_matrix(y_true=y_test, y_pred=prediction_rf)

f,ax = plt.subplots(figsize=(5, 4))

sns.heatmap(rf_2, annot=True, linewidths=.5, fmt= '.1f',ax=ax);
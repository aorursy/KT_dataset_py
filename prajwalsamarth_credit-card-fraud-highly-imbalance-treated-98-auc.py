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
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import scipy.stats as st 

import os

df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.info()
df.Class.value_counts()
df_0=df[df['Class']==0].sample(n=9000,random_state=123)

df1=df.drop(df_0.index)

df_1=df[df['Class']==1].sample(n=90,random_state=123)

df1=df1.drop(df_1.index)

test=pd.concat([df_0,df_1],sort=False)
test=test.sample(frac=1,random_state=1)
test.shape
plt.figure(figsize=[15,15])

sns.scatterplot(data=df,x='Time',y='Class')
sns.scatterplot(x=list(set(df_1['Time'])),y=df_1.groupby('Time').agg('count')['V1'])
df1=df1.drop('Time',axis=1)

test=test.drop('Time',axis=1)
from imblearn.under_sampling import RandomUnderSampler



x=df1.drop('Class',axis=1)

y=df1['Class']

rus=RandomUnderSampler(sampling_strategy=0.33,random_state=7)

x,y=rus.fit_resample(x,y)
y.value_counts()
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import *
def model_val(mod,x,y):

    

    pred=mod.predict(x)

    prob=mod.predict_proba(x)[:,1]

    auc=roc_auc_score(y,prob)

    acc=accuracy_score(y,pred)

    recall=recall_score(y,pred)

    #return auc

    print('AUC_ROC= ',auc)

    print('Accuracy= ',acc)

    print('Recall=',recall)

    print('confusion matrix:\n',confusion_matrix(y,pred))

    

def models(x,y,xts,ytx):

    models={'Logistic regression':LogisticRegression(max_iter=10e10),'Random forest':RandomForestClassifier(),'Light GBM':LGBMClassifier()}

    for i in models:

        print('\n',i)

        mod=models[i].fit(x,y)

        model_val(mod,xts,yts)

        

        
xts=test.drop('Class',axis=1)

yts=test['Class']

models(x,y,xts,yts)
from imblearn.over_sampling import RandomOverSampler



x=df1.drop('Class',axis=1)

y=df1['Class']

ros=RandomOverSampler(sampling_strategy=0.33,random_state=7)

x,y=ros.fit_resample(x,y)
y.value_counts()
xts=test.drop('Class',axis=1)

yts=test['Class']

models(x,y,xts,yts)
from imblearn.over_sampling import SMOTE



x=df1.drop('Class',axis=1)

y=df1['Class']

smote=SMOTE(sampling_strategy=0.33,random_state=7)

x,y=smote.fit_resample(x,y)
y.value_counts()
xts=test.drop('Class',axis=1)

yts=test['Class']

models(x,y,xts,yts)
import pandas as pd

import numpy as np

import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dat = pd.read_csv("/kaggle/input/framingham-heart-study-dataset/framingham.csv")

dat.head()

#dat.drop(['education'],axis=1,inplace=True)

#dat.head()
dat.isnull().sum()

dat.dropna(axis=0,inplace=True)

dat.head()

dat.describe()
dat.TenYearCHD.value_counts()

#shows inblance
from statsmodels.tools import add_constant as add_constant

dat_constant = add_constant(dat)

dat_constant.head()

st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)

cols=dat.columns[:-1]

model=sm.Logit(dat.TenYearCHD,dat[cols])

result=model.fit()

result.summary()
import sklearn

features=dat[['age','male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]

x=features.iloc[:,:-1]

y=features.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=5)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
sklearn.metrics.accuracy_score(y_test,y_pred)
from sklearn.metrics import roc_curve

sklearn.metrics.roc_auc_score(y_test,y_pred)
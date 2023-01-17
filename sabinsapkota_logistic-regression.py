# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import scale

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

import seaborn as sns

from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.

data=pd.read_csv("../input/h1b_kaggle.csv")

data=data.dropna()

print(list(data.columns))



data["CASE_STATUS"].value_counts()

data['CASE_STATUS']=data['CASE_STATUS'].astype('str')

data["SUCCESS"] = np.where(data["CASE_STATUS"].str.contains("CERTIFIED"), 1, 0)

data["FULLTIME_CODE"] = np.where(data["FULL_TIME_POSITION"].str.contains("Y"), 1, 0)

lb_make = LabelEncoder()





cols = ['PREVAILING_WAGE', 'YEAR']

data[cols] = data[cols].applymap(np.int64)

data.dtypes



data["EMPLOYER_NAME"] = data["EMPLOYER_NAME"].astype('category')

data["EMPLOYER_NAME"] = data["EMPLOYER_NAME"].cat.codes



data["JOB_TITLE"] = data["JOB_TITLE"].astype('category')

data["JOB_TITLE"]= data["JOB_TITLE"].cat.codes



data["WORKSITE"] = data["WORKSITE"].astype('category')

data["WORKSITE"] = data["WORKSITE"].cat.codes







y=data["SUCCESS"]

x=np.column_stack((data["JOB_TITLE"],data["EMPLOYER_NAME"], data["FULLTIME_CODE"],data["PREVAILING_WAGE"],data["YEAR"],data["WORKSITE"]))

x=sm.add_constant(x, prepend=True)



logit_model=sm.Logit(y,x)

result=logit_model.fit() 

print(result.summary())



print (np.exp(result.params))





data.head(100)









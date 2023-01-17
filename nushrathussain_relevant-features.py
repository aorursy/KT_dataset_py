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
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#backward elimination----

import statsmodels.regression.linear_model as sm

ones=[1 for i in range(0,len(data['Time']))]

ones=pd.DataFrame(ones) #appending a column of 1's

data.insert(0,'ones',ones)
label=data['Class']

data.drop(['Class'],axis=1,inplace=True)

x_opt=data.iloc[:,0:len(data.columns)].copy()

col_names={}

for i in range(0,len(x_opt.columns)): #storing column names for finally renaming them to their original names

    col_names[i]=data.columns[i]

#print(col_names)

temp=[i for i in range(0,len(data.columns))]

x_opt.columns=(temp)
x_opt.head()
ols = sm.OLS(endog = label, exog = x_opt).fit() 

print(ols.summary() )
#remove high P-value column index 24---

x_opt.drop([24],axis=1,inplace=True)
ols2 = sm.OLS(endog = label, exog = x_opt).fit() 

print(ols2.summary() )
for i in x_opt.columns:

    if col_names.get(i):

        x_opt.rename(columns={i:col_names[i]},inplace=True)

#x_opt.rename(columns=col_names,inplace=True)
#final dataset to consider!!!!!

features=x_opt

features.head()
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
data=data.sample(n=100000)
print(list(data.columns))

data["CASE_STATUS"].value_counts()
data['CASE_STATUS']=data['CASE_STATUS'].astype('str')
data["SUCCESS"] = np.where(data["CASE_STATUS"].str.contains("CERTIFIED"), 1, 0)
data["FULLTIME_CODE"] = np.where(data["FULL_TIME_POSITION"].str.contains("Y"), 1, 0)
lb_make = LabelEncoder()
data['SOC_NAME'] = data['SOC_NAME'].astype('str') 

#converting columns into int 
cols = ['PREVAILING_WAGE', 'YEAR']
data[cols] = data[cols].applymap(np.int64)
data.dtypes

data["EMPLOYER_NAME"] = data["EMPLOYER_NAME"].astype('category')
data["EMPLOYER_NAME"] = data["EMPLOYER_NAME"].cat.codes




#binary encoding field types
data['Chief Executives'] = data['SOC_NAME'].str.contains('EXECUTIVES').astype(int)
data['Finance'] = data['SOC_NAME'].str.contains('FINAN').astype(int)
data['Operations'] = data['SOC_NAME'].str.contains('OPERATIONS' or 'ADMIN' or 'MANAGEMENT').astype(int)
data['Public Relations'] = data['SOC_NAME'].str.contains('RELATIONS').astype(int)
data['Engineering'] = data['SOC_NAME'].str.contains('ENGINEER' ).astype(int)
data['Computer Science'] = data['SOC_NAME'].str.contains('COMPUTER' or 'SOFTWARE'or 'WEB').astype(int)
data['Marketing'] = data['SOC_NAME'].str.contains('ADVERTISING' or 'SURVEY' or 'MARKET' or 'SALES').astype(int)
data['Natural Science'] = data['SOC_NAME'].str.contains('SCIENTISTS' or 'BIOLOG' or 'PHYSICIST' or 'ASTRO' or 'CHEM').astype(int)
data['Education'] = data['SOC_NAME'].str.contains('TEACH').astype(int)
data['Health'] = data['SOC_NAME'].str.contains('PHYSICIAN' or 'CLINIC' or 'NURS' or 'DENTIST').astype(int)
data['Purchasing'] = data['SOC_NAME'].str.contains('PURCHASING').astype(int)
data['Analytics'] = data['SOC_NAME'].str.contains('STATIS'or 'DATA').astype(int)
data['Economics'] = data['SOC_NAME'].str.contains('ECONOMISTS').astype(int)
data['Misc'] = data['SOC_NAME'].str.contains('PSYCHO' or 'SOCIO' or 'DESIGN'or 'CURATOR' or 'LAW' or 'ART' or 'ESTATE' or 'ACCOUNT'or 'COACH' or 'ACTUAR').astype(int)

#encoding success rate
a1=data.loc[data['YEAR'] == 2012, 'SUCCESS'].sum()
a2=data.loc[data['YEAR'] == 2012, 'SUCCESS'].count()
a3=a1/a2

b1=data.loc[data['YEAR'] == 2013, 'SUCCESS'].sum()
b2=data.loc[data['YEAR'] == 2013, 'SUCCESS'].count()
b3=b1/b2

c1=data.loc[data['YEAR'] == 2014, 'SUCCESS'].sum()
c2=data.loc[data['YEAR'] == 2014, 'SUCCESS'].count()
c3=c1/c2

d1=data.loc[data['YEAR'] == 2015, 'SUCCESS'].sum()
d2=data.loc[data['YEAR'] == 2015, 'SUCCESS'].count()
d3=d1/d2

e1=data.loc[data['YEAR'] == 2016, 'SUCCESS'].sum()
e2=data.loc[data['YEAR'] == 2016, 'SUCCESS'].count()
e3=e1/e2

f1=data.loc[data['YEAR'] == 2011, 'SUCCESS'].sum()
f2=data.loc[data['YEAR'] == 2011, 'SUCCESS'].count()
f3=f1/f2

#adding data into the column for success rate
data.loc[data['YEAR'] == 2011, 'RATE'] = f3
data.loc[data['YEAR'] == 2012, 'RATE'] = a3
data.loc[data['YEAR'] == 2013, 'RATE'] = b3
data.loc[data['YEAR'] == 2014, 'RATE'] = c3
data.loc[data['YEAR'] == 2015, 'RATE'] = d3
data.loc[data['YEAR'] == 2016, 'RATE'] = e3




#encoding location
data[['City', 'State']] = data['WORKSITE'].str.split(',\s+', expand=True)
lb_make = LabelEncoder()
data["state_code"] = lb_make.fit_transform(data["State"])


y=data["SUCCESS"]
x=np.column_stack((data["EMPLOYER_NAME"], data["FULLTIME_CODE"],data["PREVAILING_WAGE"],data["YEAR"],data["state_code"],data["RATE"],data['Chief Executives'],data['Finance'],data['Operations'],
       data['Public Relations'], data['Engineering'], data['Computer Science'], data['Marketing'],data['Natural Science'],data['Education'],
         data['Health'],  data['Purchasing'],data['Analytics'],data['Economics'],data['Misc']))
x=sm.add_constant(x, prepend=True)

logit_model=sm.Logit(y,x)
result=logit_model.fit() 
print(result.summary())

print (np.exp(result.params))

data.dtypes
data.head(10)


#group_by_success = data.groupby(['YEAR','SUCCESS'])
#count_delays_by_success = group_by_success.size().unstack()
#count_delays_by_success















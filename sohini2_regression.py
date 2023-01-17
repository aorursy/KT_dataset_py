# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import PolynomialFeatures as pf
d= pd.read_csv("../input/data.csv")
d.head()
#Getting the price data
d1=np.array(d.iloc[:,12])
d1
#Getting the data where price is missing
Unknwn_data=d[pd.isna(d1)==True]
Unknwn_data.head()
#Getting the data where price is not null
data=d[~pd.isna(d1)==True]
data.head()
#Dropping the Price column to get our X
data1=data.drop('PRICE',axis=1)
#Getting Y
Price=data.iloc[:,12]
#Finding the column which has missing values
data_na = (data1.isnull().sum() / len(data1)) * 100
d_na=pd.DataFrame(data_na)
d_na.columns=['Missing%']
d_na
#Finding the column which has more than 50% missing values
d_na[d_na['Missing%']>50]
#Dropping the column which has more than 50% missing values
df=data1.drop(['CMPLX_NUM','LIVING_GBA'],axis=1)
df.head()
#Dropping columns which are not necessary
df=df.drop(['ZIPCODE','LATITUDE','LONGITUDE','X','Y','BLDG_NUM','SALEDATE','GIS_LAST_MOD_DTTM','SQUARE','FULLADDRESS','CENSUS_BLOCK','CITY','STATE','NATIONALGRID'],axis=1)
df
#Replacing the missing values for objects with mode
def replace(x):
    x1=df.mode()[x][0]
    df[x].fillna(x1,inplace=True)
    return df
replace('QUADRANT')
replace('STYLE')
replace('STRUCT')
replace('GRADE')
replace('CNDTN')
replace('EXTWALL')
replace('ROOF')
replace('INTWALL')
replace('ASSESSMENT_SUBNBHD')
#Replacing the missing values for float with median
def replace1(x):
    x1=df.median()[x]
    df[x].fillna(x1,inplace=True)
    return df
replace1('AYB')
replace1('NUM_UNITS')
replace1('GBA')
replace1('KITCHENS')
replace1('STORIES')
replace1('YR_RMDL')
#Getting dummy variables
df = pd.get_dummies(df, prefix='HEAT_', columns=['HEAT'])
df = pd.get_dummies(df, prefix='AC_', columns=['AC'])
df = pd.get_dummies(df, prefix='QUALIFIED_', columns=['QUALIFIED'])
df = pd.get_dummies(df, prefix='STYLE_', columns=['STYLE'])
df = pd.get_dummies(df, prefix='STRUCT_', columns=['STRUCT'])
df = pd.get_dummies(df, prefix='GRADE_', columns=['GRADE'])
df = pd.get_dummies(df, prefix='CNDTN_', columns=['CNDTN'])
df = pd.get_dummies(df, prefix='EXTWALL_', columns=['EXTWALL'])
df = pd.get_dummies(df, prefix='ROOF_', columns=['ROOF'])
df = pd.get_dummies(df, prefix='INTWALL_', columns=['INTWALL'])
df = pd.get_dummies(df, prefix='SOURCE_', columns=['SOURCE'])
df = pd.get_dummies(df, prefix='CENSUS_TRACT_', columns=['CENSUS_TRACT'])
df = pd.get_dummies(df, prefix='ASSESSNBHD_', columns=['ASSESSMENT_NBHD'])
df = pd.get_dummies(df, prefix='ASSESSUBNBHD_', columns=['ASSESSMENT_SUBNBHD'])
df = pd.get_dummies(df, prefix='WARD_', columns=['WARD'])
df = pd.get_dummies(df, prefix='QUADRANT_', columns=['QUADRANT'])
X_train,X_test,y_train,y_test=train_test_split(df,Price,random_state=42,test_size=0.2)
# We fit Linear Regression
from statsmodels.api import OLS
model=OLS(y_train,X_train)
results=model.fit().summary()
results
# Now we get the p-values
p=model.fit().pvalues
f = pf(1).fit(X_train)
names=f.get_feature_names(X_train.columns)
d2=pd.DataFrame([names,list(p)]).T
d2
#Dropping those which has p-value None
for i in range(500):
    if(d2.iloc[i,1]==None):
        d3=d2.drop(d2.index[i])
d3
#Then we remove those with insignificant p-values
d3.columns = ['coef', 'p_val']
d4=d3[d3.p_val < 0.05]
d4
index=d4.index
a=X_train.iloc[:,index]
a
model=OLS(y_train,a)
model.fit().summary()
# Now we fit our test data
a1=X_test.iloc[:,index]
model1=OLS(y_test,a1)
model1.fit().summary()
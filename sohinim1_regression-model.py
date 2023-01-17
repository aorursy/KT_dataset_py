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
# Importing necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures as pf
from statsmodels.api import OLS
from sklearn.pipeline import make_pipeline
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
# Reading the data
data=pd.read_csv("../input/DC_Properties.csv")
data.head()
data.info()
# Retrieving the Price column
Price=np.array(data.iloc[:,12])
Price
#a)
# Saving the unknown data
unknown_data=data[pd.isna(Price)==True]
unknown_data.head()
# Remaining data
d1=data[pd.isna(Price)==False]
d1.head()
# Eleminating price column
d2=d1.drop('PRICE', axis=1)
# Finding percentage of missing values
d3 = (d2.isnull().sum() / len(d2)) * 100
d3
# Forming a dataframe eleminating the variables:-
d4=d2.drop(['CMPLX_NUM','LIVING_GBA'],axis=1)
d4.head()

# Forming a dataframe eleminating the variables:-
d4=d4.drop(['ZIPCODE','LATITUDE','LONGITUDE','X','Y','BLDG_NUM','SALEDATE','GIS_LAST_MOD_DTTM','CITY','STATE','NATIONALGRID','FULLADDRESS','CENSUS_BLOCK','SQUARE'],axis=1)

d4.head()
# Finding percentage of missing values
d5 = (d4.isnull().sum() / len(d2)) * 100
d5
# Replace all missing values of categorical columns with the mode of respective columns:-
def replace_na(column):
        mode=d4.mode()[column][0]
        d4[column].fillna(mode,inplace=True)
        return d4
replace_na('QUADRANT')
replace_na('STYLE')
replace_na('STRUCT')
replace_na('GRADE')
replace_na('CNDTN')
replace_na('EXTWALL')
replace_na('ROOF')
replace_na('INTWALL')
replace_na('ASSESSMENT_SUBNBHD')
# Replacing all missing values of non-categorical columns with median:-
def replace_na1(column):
        median=d4.median()[column]
        d4[column].fillna(median,inplace=True)
        return d4
    
replace_na1('NUM_UNITS')    
replace_na1('AYB')
replace_na1('STORIES')
replace_na1('GBA')
replace_na1('KITCHENS')
replace_na1('YR_RMDL')
# Finding mising values
d5 = (d4.isnull().sum() / len(d2)) * 100
d5
# Converting to dummy variables
d4= pd.get_dummies(d4, prefix='HEAT_', columns=['HEAT'])
d4= pd.get_dummies(d4, prefix='AC_', columns=['AC'])
d4= pd.get_dummies(d4, prefix='QUALIFIED_', columns=['QUALIFIED'])
d4= pd.get_dummies(d4, prefix='STYLE_', columns=['STYLE'])
d4= pd.get_dummies(d4, prefix='STRUCT_', columns=['STRUCT'])
d4= pd.get_dummies(d4, prefix='GRADE_', columns=['GRADE'])
d4= pd.get_dummies(d4, prefix='CNDTN_', columns=['CNDTN'])
d4= pd.get_dummies(d4, prefix='EXTWALL_', columns=['EXTWALL'])
d4= pd.get_dummies(d4, prefix='ROOF_', columns=['ROOF'])
d4= pd.get_dummies(d4, prefix='INTWALL_', columns=['INTWALL'])
d4= pd.get_dummies(d4, prefix='SOURCE_', columns=['SOURCE'])
d4= pd.get_dummies(d4, prefix='ASSESS_NBHD_', columns=['ASSESSMENT_NBHD'])
d4= pd.get_dummies(d4, prefix='ASSESS_SUBNBHD_', columns=['ASSESSMENT_SUBNBHD'])
d4= pd.get_dummies(d4, prefix='CENSUS_TRACT_', columns=['CENSUS_TRACT'])
d4= pd.get_dummies(d4, prefix='WARD_', columns=['WARD'])
d4= pd.get_dummies(d4, prefix='QUADRANT_', columns=['QUADRANT'])
d4.head()
# Retrieving price column:-
Price2=np.array(d1.iloc[:,12])
Price2
# Splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split( d4, Price2 , test_size=0.2, random_state=42)
# Fitting OLS using Linear Regression
model=OLS(y_train, X_train)
# Fitting model
model.fit().summary()
# First we get the p-values
p=model.fit().pvalues
f=pf(1).fit(X_train)
names=f.get_feature_names(X_train.columns)
df=pd.DataFrame([names,list(p)]).T
df
# Dropping None Type variables:-
for i in range(500):
    if(df.iloc[i,1]==None):
        df1=df.drop(df.index[i])
df1
df1.columns=['coef','p_val']
df2=df1[df1.p_val<0.05]
index=df2.index
a=X_train.iloc[:,index]
a
# Fitting the model:-
model=OLS(y_train,a)
model.fit().summary()
a1=X_test.iloc[:,index]
model1=OLS(y_test,a1)
model1.fit().summary()
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
hr=pd.read_csv('../input/HR_comma_sep.csv')
hr.head()
hr.info()
hr.columns
hr.describe()
hr.isnull().sum()
num_hr=hr[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years']]
num_hr.head()
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

summary=num_hr.apply(lambda x: var_summary(x)).T
summary
x=hr.iloc[:,[0,1,2,3,4,5,7,8,9]].values
y=hr.iloc[:,6].values
x
y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:,7] = le.fit_transform(x[:,7])
x[:,8] = le.fit_transform(x[:,8])
onehotencoder = OneHotEncoder(categorical_features = [7,8])
x = onehotencoder.fit_transform(x).toarray()
pd.DataFrame(x)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

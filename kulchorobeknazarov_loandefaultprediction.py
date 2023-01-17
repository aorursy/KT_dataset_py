import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df=pd.read_csv('../input/lt-vehicle-loan-default-prediction/train.csv')
df.describe()
df.isnull().sum()
data_cleaned=df.dropna(axis=0)
data_cleaned.columns
data_cleaned.drop(['UniqueID','branch_id','supplier_id','manufacturer_id','Current_pincode_ID','State_ID','Employee_code_ID','Date.of.Birth','DisbursalDate'],axis=1)
AVERAGE_ACCT_AGE_2=data_cleaned['AVERAGE.ACCT.AGE'].replace({'yrs':"",'mon':""," ":"."},regex=True).astype(float)
CREDIT_HISTORY_LENGTH=data_cleaned['CREDIT.HISTORY.LENGTH'].replace({'yrs':"",'mon':""," ":"."},regex=True).astype(float)
data_cleaned['AVERAGE.ACCT.AGE']=AVERAGE_ACCT_AGE_2
data_cleaned['CREDIT.HISTORY.LENGTH']=CREDIT_HISTORY_LENGTH
data_w_d=pd.get_dummies(data_cleaned,drop_first=True)
y=data_w_d['loan_default']
x=data_w_d.drop(['loan_default'],axis=1)

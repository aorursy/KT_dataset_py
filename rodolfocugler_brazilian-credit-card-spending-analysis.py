import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
# reading dataset

df = pd.read_csv('../input/brazilian-real-bank-dataset/MiBolsillo.csv', 
                 encoding='CP1252', 
                 sep=';', 
                 decimal=',', 
                 na_values=' -   ',
                 thousands='.',
                 parse_dates=[8], 
                 dayfirst=True)
# replacing column names from portuguese to english
df.columns = ['id','branch_number','city','state','age','gender','total_credit_card_limit','current_available_limit' ,'date','amount','category_expense','purchase_city','purchase_country']
df.head()
# converting gender column to a binary data
dms = pd.get_dummies(df['gender'])
df = pd.concat([df,dms],axis=1)
df.drop(['gender', 'M'], axis=1,inplace=True)
df.rename(columns={'F': 'female'}, inplace=True)
profile = ProfileReport(df, title="Pandas Profiling Report")
profile
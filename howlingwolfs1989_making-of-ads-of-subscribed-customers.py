import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
data = pd.read_csv('../input/Customer_Sales_Transactional_Data_CSV.txt', delimiter = ",")
df = data.copy()
df.head()
df.columns = ['SaleDate', 'CustomerID', 'SalesAmount']
df.shape
df.dtypes
df.duplicated().sum()
df.isnull().sum()
df['SaleDate'] = pd.to_datetime(df['SaleDate'])
df.dtypes
df = df.sort_values(by=['SaleDate']).reset_index(drop=True)
df.head()
df.describe(include='all')
df['SaleDate'].unique()
df['Week'] = pd.DatetimeIndex(df['SaleDate']).week
df['Week'].unique()
df['Week'] = abs(df['Week'] % 9 - 7)
df['Week'].unique()
"""df_sum = df[['Week'] > 0].groupby(['CustomerID']).agg({
    'SalesAmount':{
        'HISTORIC_SALES':'sum',
        'STD_SALESAMOUNT':'std',
        'VARIATION_SALESAMOUNT':'var',
        'MIN_SALESAMOUNT':'min',
        'MAX_SALESAMOUNT':'max',
    }
}).reset_index()"""
#df_sum.columns = df_sum.columns.droplevel()
"""df_sum.columns = ['CustomerID',
                  'Week',
                  'HISTORIC_SALES', 
                  'STD_SALESAMOUNT',
                  'VARIATION_SALESAMOUNT',
                  'MIN_SALESAMOUNT',
                  'MAX_SALESAMOUNT',
                 ]"""
#df_sum.head()
df_dates = df[df['Week'] > 0].groupby(['CustomerID']).agg({
    'SaleDate':{
        'HISTORIC_VISITS':'count',
        'FIRST_VISIT':'min',
        'LAST_VISIT':'max'
    },
    'SalesAmount':{
        'HISTORIC_SALES':'sum',
        'STD_SALESAMOUNT':'std',
        'VARIATION_SALESAMOUNT':'var',
        'MIN_SALESAMOUNT':'min',
        'MAX_SALESAMOUNT':'max',
    }
}).reset_index()
df_dates.head()
df_dates.columns.droplevel()
df_dates.columns = ['CustomerID',
                    'HISTORIC_VISITS', 
                    'FIRST_VISIT', 
                    'LAST_VISIT',
                    'HISTORIC_SALES', 
                    'STD_SALESAMOUNT',
                    'VARIATION_SALESAMOUNT',
                    'MIN_SALESAMOUNT',
                    'MAX_SALESAMOUNT'
                   ]
#df_dates.columns = ['CustomerID', 'HISTORIC_VISITS', 'FIRST_VISIT', 'LAST_VISIT', 'HISTORIC_SALES', 'STD_SALESAMOUNT', 'VARIATION_SALESAMOUNT', 'MIN_SALESAMOUNT', 'MAX_SALESAMOUNT']
df_dates.head()
def amount_df(week, number):
        df = week.groupby(['CustomerID', 'Week']).agg({
            'SalesAmount':{
                f'W{number}_HISTORIC_VISITS':'count',
                f'W{number}_HISTORIC_SALES':'sum',
                f'W{number}_STD_SALESAMOUNT':'std',
                f'W{number}_VARIATION_SALESAMOUNT':'var', 
                f'W{number}_MIN_SALESAMOUNT': 'min',
                f'W{number}_MAX_SALESAMOUNT': 'max'
            }}).reset_index()
        df.columns = df.columns.droplevel()
        df.columns = ['CustomerID',
                      'Week',
                      f'W{number}_HISTORIC_VISIT',
                      f'W{number}_HISTORIC_SALES',
                      f'W{number}_STD_SALESAMOUNT',
                      f'W{number}_VARIATION_SALESAMOUNT',
                      f'W{number}_MIN_SALESAMOUNT',
                      f'W{number}_MAX_SALESAMOUNT'
                     ]
        return df
week0 = df[df['Week'] == 0].reset_index(drop=True)
week1 = df[df['Week'] == 1].reset_index(drop=True)
week2 = df[df['Week'] == 2].reset_index(drop=True)
week3 = df[df['Week'] == 3].reset_index(drop=True)
week4 = df[df['Week'] == 4].reset_index(drop=True)
week5 = df[df['Week'] == 5].reset_index(drop=True)
week_0_a = amount_df(week0, '0')
week_0_a.drop('Week', axis=1, inplace=True)

week_1_a = amount_df(week1, '1')
week_1_a.drop('Week', axis=1, inplace=True)

week_2_a = amount_df(week2, '2')
week_2_a.drop('Week', axis=1, inplace=True)

week_3_a = amount_df(week3, '3')
week_3_a.drop('Week', axis=1, inplace=True)

week_4_a = amount_df(week4, '4')
week_4_a.drop('Week', axis=1, inplace=True)

week_5_a = amount_df(week5, '5')
week_5_a.drop('Week', axis=1, inplace=True)
week_1_a.tail()
dfs = [df_dates, week_0_a, week_1_a, week_2_a, week_3_a, week_4_a, week_5_a]
df_final = pd.DataFrame()
df_final = reduce(lambda left,right: pd.merge(left,right,how='outer', left_on='CustomerID', right_on='CustomerID'), dfs)
df_final.head()
df_final.shape
from datetime import datetime
d0 = datetime(2014,10,19)
df_final.loc[df_final['FIRST_VISIT'].isnull(),'FIRST_VISIT'] = d0
df_final.loc[df_final['LAST_VISIT'].isnull(),'LAST_VISIT'] = d0
df_final.isnull().sum()
df_final.fillna(0, inplace=True)
df_final.isnull().sum()
df_final[['W0_HISTORIC_VISIT', 
          'W0_HISTORIC_SALES', 
          'W0_STD_SALESAMOUNT', 
          'W0_VARIATION_SALESAMOUNT', 
          'W0_MIN_SALESAMOUNT',
          'W0_MAX_SALESAMOUNT']].head()
def identifyChurn(sale):
    if sale > 0:
        return 0
    else:
        return 1
df_final['CHURN'] = 0
df_final.loc[:, ('CHURN')] = df_final['W0_HISTORIC_SALES'].apply(identifyChurn)
df_final.drop(['W0_HISTORIC_VISIT', 
          'W0_HISTORIC_SALES', 
          'W0_STD_SALESAMOUNT', 
          'W0_VARIATION_SALESAMOUNT', 
          'W0_MIN_SALESAMOUNT',
          'W0_MAX_SALESAMOUNT'], axis=1
              , inplace=True)
df_final['FIRST_VISIT_DAY'] = 4444
df_final.loc[:, ('FIRST_VISIT_DAY')] = d0 - df_final['FIRST_VISIT']
df_final['LAST_VISIT_DAY'] = 4444
df_final.loc[:, ('LAST_VISIT_DAY')] = d0 - df_final['LAST_VISIT']
df_final.dtypes
df_final['FIRST_VISIT_DAY'] = df_final['FIRST_VISIT_DAY'].dt.days
df_final['LAST_VISIT_DAY'] = df_final['LAST_VISIT_DAY'].dt.days
df_final.drop(['LAST_VISIT','FIRST_VISIT'],axis=1,inplace=True)
df_final.var()
feat_var = df_final.var()
tmp = pd.DataFrame({"feature":feat_var.index,"var":feat_var.values})
tmp = tmp.sort_values('var',ascending=False)
tmp
tmp.drop([0,37],axis=0,inplace=True)
feat_log=tmp[tmp['var']>500]['feature']
feat_log.reset_index(drop=True)
df_logtransf = df_final.copy()
for i in feat_log:
    df_logtransf[i] = np.log(1 + df_logtransf[i])
df_logtransf.var()
df_logtransf.head()
final_ADS = df_logtransf.copy()
final_ADS.head()
#final_ADS.to_excel('My_Final_ADS.xlsx')
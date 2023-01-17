import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # accessing directory structure
import matplotlib.pyplot as plt # plotting
print(os.listdir('../input'))
## Load excel file sheets
df_income_statment = pd.read_excel('../input/Australia_historical-dataset-2018-for-publication.xlsx', sheet_name = 'Income Statement dataset')
df_balance_sheet = pd.read_excel('../input/Australia_historical-dataset-2018-for-publication.xlsx', sheet_name = 'Balance Sheet dataset')
df_cash_flow = pd.read_excel('../input/Australia_historical-dataset-2018-for-publication.xlsx', sheet_name = 'Cash flow statement dataset')
## Replace nan with ZERO
df_income_statment = df_income_statment.fillna(0)
df_balance_sheet = df_balance_sheet.fillna(0)
df_cash_flow = df_cash_flow.fillna(0)
df_income_statment = df_income_statment.loc[df_income_statment['CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME BY SECTOR'] != 0]
df_income_statment.head()
df_balance_sheet = df_balance_sheet.loc[df_balance_sheet['CONSOLIDATED STATEMENT OF FINANCIAL POSITION BY SECTOR'] != 0]
df_balance_sheet.head()
df_cash_flow = df_cash_flow.loc[df_cash_flow['CONSOLIDATED STATEMENT OF CASH FLOWS BY SECTOR'] != 0]
df_cash_flow.head()
df_cash_flow=df_cash_flow.drop(columns=['Unnamed: 1','Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1)
df_cash_flow=df_cash_flow.drop([0])
df_cash_flow.head()
df_cash_flow[:1]
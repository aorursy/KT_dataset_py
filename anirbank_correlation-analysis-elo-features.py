# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import percentile
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df_merged=pd.read_csv('../input/elo-cards-merging-month-lag/train_df_merged_V2.csv')
train_df_merged = train_df_merged.rename(columns={'month_lag_x': 'month_lag'})
train_df_merged = train_df_merged.rename(columns={'month_lag_y': 'median_month_lag'})
def extract_features(df):
    df['log_TotalPurchAmtMerch']=np.log(1+df['TotalPurchAmtMerch'])
    df['log_avg_sales_lag3']=np.log(1+df['avg_sales_lag3'])
    df['log_avg_sales_lag6']=np.log(1+df['avg_sales_lag6'])
    df['log_avg_sales_lag12']=np.log(1+df['avg_sales_lag12'])
    df['log_avg_purchases_lag3']=np.log(1+df['avg_purchases_lag3'])
    df['log_avg_purchases_lag6']=np.log(1+df['avg_purchases_lag6'])
    df['log_avg_purchases_lag12']=np.log(1+df['avg_purchases_lag12'])
    df['log_Median_purchases_lag3']=np.log(1+df['Median_avg_purchases_lag3'])
    df['log_Median_purchases_lag6']=np.log(1+df['Median_avg_purchases_lag6'])
    df['log_Median_purchases_lag12']=np.log(1+df['Median_avg_purchases_lag12'])
    df['log_Variance_purchases_lag3']=np.log(1+df['Variance_avg_purchases_lag3'])
    df['log_Variance_purchases_lag6']=np.log(1+df['Variance_avg_purchases_lag6'])
    df['log_Variance_purchases_lag12']=np.log(1+df['Variance_avg_purchases_lag12'])
    df['log_Median_sales_lag3']=np.log(1+df['Median_avg_sales_lag3'])
    df['log_Median_sales_lag6']=np.log(1+df['Median_avg_sales_lag6'])
    df['log_Median_sales_lag12']=np.log(1+df['Median_avg_sales_lag12'])
    df['log_Variance_sales_lag3']=np.log(1+df['Variance_avg_sales_lag3'])
    df['log_Variance_sales_lag6']=np.log(1+df['Variance_avg_sales_lag6'])
    df['log_Variance_sales_lag12']=np.log(1+df['Variance_avg_sales_lag12'])
    
    df["AmtPerCity"] = df["TotalPurchAmtMerch"] / df["CityCtByCard"]
    df["AmtPerState"] = df["TotalPurchAmtMerch"] / df["CtStateMerch"]
    df["AmtPerSubSector"] = df["TotalPurchAmtMerch"] / df["CtSubSectorMerch"]
extract_features(train_df_merged)
train_df_corr = train_df_merged[['month_lag','median_month_lag','AmtPerCity','AmtPerState','AmtPerSubSector','target']]
corr=train_df_corr.corr()
corr
print(train_df_merged['month_lag'].max())
print(train_df_merged['month_lag'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['month_lag'])
print('Pearsons correlation betweeen target and Month lag: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.power(train_df_merged['month_lag'],2))
print('Pearsons correlation betweeen target and sq month lag: %.3f' % corr)
print(train_df_merged['AmtPerCity'].max())
print(train_df_merged['AmtPerCity'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['AmtPerCity'])
print('Pearsons correlation betweeen target and Month lag: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['AmtPerCity']))
print('Pearsons correlation betweeen target and sq month lag: %.3f' % corr)
print(train_df_merged['age_of_card'].max())
print(train_df_merged['age_of_card'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['age_of_card'])
print('Pearsons correlation betweeen target and Age of Card: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['age_of_card']))
print('Pearsons correlation betweeen target and log Age of Card: %.3f' % corr)
print(train_df_merged['CtStateMerch'].max())
print(train_df_merged['CtStateMerch'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['CtStateMerch'])
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.power(train_df_merged['CtStateMerch'],0.5))
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
print(train_df_merged['TransCtMerchCat'].max())
print(train_df_merged['TransCtMerchCat'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['TransCtMerchCat'])
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.power(train_df_merged['TransCtMerchCat'],0.5))
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
print(train_df_merged['MaxTransCtCity'].max())
print(train_df_merged['MaxTransCtCity'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['MaxTransCtCity'])
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.power(train_df_merged['MaxTransCtCity'],0.5))
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['CityCtByCard'])
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['CityCtByCard']))
print('Pearsons correlation betweeen target and log CtStateMerch: %.3f' % corr)
print(train_df_merged['MaxTransCtMerch'].max())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['MaxTransCtMerch'])
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['MaxTransCtMerch']))
print('Pearsons correlation betweeen target and log CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['CtStateMerch'])
print('Pearsons correlation betweeen target and CtStateMerch: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['CtStateMerch']))
print('Pearsons correlation betweeen target and log CtStateMerch: %.3f' % corr)
print(train_df_merged['cat2_2_count'].max())
print(train_df_merged['cat2_2_count'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['cat2_2_count'])
print('Pearsons correlation betweeen target and cat2_1_count: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['cat2_2_count']))
print('Pearsons correlation betweeen target and log cat2_1_count: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['avg_purchases_lag6'])
print('Pearsons correlation betweeen target and avg_sales_lag3: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['avg_purchases_lag6']))
print('Pearsons correlation betweeen target and log avg_sales_lag3: %.3f' % corr)
print(train_df_merged['MaxPurchAmtMerch'].max())
print(train_df_merged['TotalPurchAmtMerch'].max())
print(train_df_merged['MaxPurchAmtMerch'].min())
print(train_df_merged['TotalPurchAmtMerch'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['TotalPurchAmtMerch'])
print('Pearsons correlation betweeen target and Max purch amt: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['TotalPurchAmtMerch']))
print('Pearsons correlation betweeen target and log Max purch amt: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['Median_avg_sales_lag12'])
print('Pearsons correlation betweeen target and Variance lag: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.log(1+train_df_merged['Median_avg_sales_lag12']))
print('Pearsons correlation betweeen target and log Variance lag: %.3f' % corr)
print(train_df_merged['cat3_A_count'].max())
print(train_df_merged['cat3_A_count'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['cat3_A_count'])
print('Pearsons correlation betweeen target and Variance lag: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.power(train_df_merged['cat3_A_count'],0.5))
print('Pearsons correlation betweeen target and log Variance lag: %.3f' % corr)
print(train_df_merged['installments_y'].max())
print(train_df_merged['installments_y'].min())
corr, _ = pearsonr(train_df_merged['target'],train_df_merged['installments_y'])
print('Pearsons correlation betweeen target and Variance lag: %.3f' % corr)
corr, _ = pearsonr(train_df_merged['target'],np.power(train_df_merged['installments_y'],2))
print('Pearsons correlation betweeen target and log Variance lag: %.3f' % corr)
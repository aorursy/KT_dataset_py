# https://github.com/CamDavidsonPilon/lifetimes
!pip install lifetimes
# Importing the required libraries for the analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as pp
import datetime
from lifetimes.plotting import *
from lifetimes.utils import *
import os
file_path = '/kaggle/input/'

file = os.path.join(file_path,'ecommerce-data/data.csv')
df = pd.read_csv(file,encoding='unicode_escape')
df.head(3)
# Checking the datatypes 
print(df.dtypes)

print('#############')
# checking if there are any null values
print(df.isnull().sum()/df['CustomerID'].shape[0])
# converting the invoice date to date datatype
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
# checking if there are -nev values
len(df[df['Quantity'] < 0])
# remove missing values and take +ve quantily rows
df = df[df['Quantity'] >0 & df['CustomerID'].notnull()]

df['amt'] = df['Quantity'] * df['UnitPrice']
# Creating Recency, Frequency & Time period

# Using lifetime package
dfnew = summary_data_from_transaction_data(df,'CustomerID', 'InvoiceDate', monetary_value_col='amt', observation_period_end='2011-12-9')
dfnew['frequency'].plot(kind = 'hist',bins=50)

from lifetimes import BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(dfnew['frequency'],dfnew['recency'],dfnew['T'])
print(bgf)
fig = plt.figure(figsize=(10,8))
plot_frequency_recency_matrix(bgf)
fig = plt.figure(figsize=(10,8))
plot_probability_alive_matrix(bgf)
t = 1
dfnew['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(t, dfnew['frequency'], dfnew['recency'], dfnew['T'])
dfnew.sort_values(by='predicted_purchases').tail(5)

dfnew.sort_values(by='predicted_purchases').head(5)

from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)


summary_cal_holdout = calibration_and_holdout_data(df, 'CustomerID', 'InvoiceDate',
                                        calibration_period_end='2011-06-08',
                                        observation_period_end='2011-12-9' )   
print(summary_cal_holdout.head())

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)





t = 10
individual =dfnew.loc[12380]
bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])


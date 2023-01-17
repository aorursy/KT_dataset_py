# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# read data
DstAll = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv', na_values= ['NA',''])
DstAll.head()
DstAll.dtypes
# convert type of Total Charges
DstAll['TotalCharges'] = DstAll['TotalCharges'].apply(pd.to_numeric, errors='coerce')
# map the values for SeniorCitizen
DstAll.SeniorCitizen.replace((1, 0), ('Yes', 'No'), inplace=True)
print("The Minimum value is: ",min(DstAll['tenure'])," and the Maximum value is: ",max(DstAll['tenure']))
# check for Null values in dataset
DstAll.isnull().values.any()
import matplotlib.pyplot as plt
import plotly.plotly as py

# we will divide data under 9 bins
plt.hist(DstAll['tenure'], bins= 9)
# create the goupping function with the define intervals
def CreateGrp(tn):
    if ((tn >= 0) & (tn <= 12)):
        return('[0-12]')
    elif ((tn > 12) & (tn <= 24)):
        return('[12-24]')
    elif ((tn > 24) & (tn <= 48)):
        return('[24-48]')
    elif ((tn > 48) & (tn <= 60)):
        return('[48-60]')
    else:
        return('[> 60]')
    
DstAll['tenureBins'] = DstAll['tenure'].apply(lambda x: CreateGrp(x))
# check the frequency for each group bin
DstAll['tenureBins'].value_counts().plot(kind='bar')
DstAll.drop(['tenure', 'customerID'], axis = 1, inplace = True)
DstAll.corr(method='pearson')
colors = ["#d0d0d0", "#E69F00"]

def plotStackChart(df, col1, col2 ):
    df = df.groupby([col1, col2])[col2].count().fillna(0).groupby(level=[0]).apply(lambda x: x / x.sum()).unstack(col2)
    df.plot(kind='bar', color=colors, stacked=True, title = col1 + ' vs ' + col2)
 
cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod','tenureBins']
for col in cols:
    plotStackChart(DstAll, col, 'Churn')

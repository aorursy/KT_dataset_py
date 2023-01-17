#importing Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

pd.set_option('display.max_column',None)
pd.set_option('display.max_row',None)

#importing Data
data=pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv',parse_dates=['InvoiceDate'])
#Dataset Shape
data.shape
# Checking for Null Values
data.isna().sum()
# Dropping Null Values in Customer ID Column
data=data.dropna(subset=['Customer ID'])
# Checking for Duplicates
data.duplicated().sum()
# Dropping duplicates
data=data.drop_duplicates()
data['InvoiceMonth']=data['InvoiceDate'].apply(lambda x: dt.datetime(x.year,x.month,1))

grouping= data.groupby('Customer ID')['InvoiceMonth']
data['CohortMonth']=grouping.transform('min')
data.head(3)
def cohort_index(df,column):
    year=df[column].dt.year
    month=df[column].dt.month
    day=df[column].dt.day
    return year,month,day

inv_year,inv_month,inv_day=cohort_index(data,'InvoiceMonth')
coh_year,coh_month,coh_day=cohort_index(data,'CohortMonth')

data['CohortIndex']=((inv_year-coh_year)*12)+(inv_month-coh_month)+1
grouping=data.groupby(['CohortMonth','CohortIndex'])
cohort_data=grouping['Customer ID'].apply(pd.Series.nunique).reset_index()
cohort_data
cohort_counts=cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='Customer ID')
cohort_data=cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_data,axis=0)
retention.index=retention.index.date
retention
plt.figure(figsize=(25,25))
plt.title('Retention Rate')
sns.heatmap(retention,annot=True,fmt='.0%',vmin = 0.0,vmax = 0.5,cmap='Blues')
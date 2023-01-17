import pandas as pd 

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
df =  pd.read_csv('../input/data.csv',encoding="ISO-8859-1")
df=df[(df.CustomerID.notnull()) & (df.UnitPrice!=0)]
df.drop_duplicates(inplace=True)
df.CustomerID = df.CustomerID.astype('int64')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'],infer_datetime_format=True)
df['InvoiceMonth'] = df['InvoiceDate'].dt.strftime('%Y-%m') 
cohorts= df.groupby('CustomerID',as_index=False)['InvoiceMonth'].min()

cohorts.rename(columns = {'InvoiceMonth':'Cohort'},inplace=True)

cohorts.head()
df_merged= pd.merge(df,cohorts , how='left', on='CustomerID')

df_merged.head()
def cohort_period(df):

    """

    Creates column CohortPeriod

    """

    df['CohortPeriod'] = np.arange(len(df))

    return df
cohorts_group = df_merged.groupby(['Cohort', 'InvoiceMonth']).agg({'CustomerID': pd.Series.nunique})

cohorts_group.rename(columns={'CustomerID': 'TotalUsers',

                        'InvoiceNo': 'TotalOrders'}, inplace=True)

cohorts_group = cohorts_group.groupby(level=0).apply(cohort_period)

cohorts_group.reset_index(inplace=True)

cohorts_group.set_index(['Cohort', 'CohortPeriod'], inplace=True) 

cohort_group_size = cohorts_group['TotalUsers'].groupby(level=0).first()

user_retention = cohorts_group['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)



sns.set(style='white')

plt.figure(figsize=(12, 8))

plt.title('Cohorts: User Retention')

sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');
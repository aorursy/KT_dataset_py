# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt
online = pd.read_csv("/kaggle/input/online-retail-ii-uci/online_retail_II.csv", encoding='utf-8')



#convert into datetime objects 

online['InvoiceDate'] = pd.to_datetime(online['InvoiceDate'])
# Make and apply function to truncate given date in column to a first day of the month



def get_month(x): return dt.datetime(x.year, x.month, 1)

online['InvoiceMonth'] = online['InvoiceDate'].apply(get_month)



# --group same-month-purchase customers into cohorts -

online.rename(columns={ online.columns[6]: "CustomerID" }, inplace = True)

grouped = online.groupby('CustomerID')['InvoiceMonth']

online['CohortMonth'] = grouped.transform('min')



print(online.describe())

#now we create a way to index cohorts based on how many months have passed since the time of acquisition & first purchase 



def get_date_int(df, column):

    year = df[column].dt.year

    month = df[column].dt.month

    day = df[column].dt.day

    return year, month, day

invoice_year, invoice_month, _ = get_date_int(online, 'InvoiceMonth')

cohort_year, cohort_month, _ = get_date_int(online, 'CohortMonth')



years_diff = invoice_year - cohort_year

months_diff = invoice_month - cohort_month



online['CohortIndex'] = years_diff * 12 + months_diff + 1

online.head()
grouping_count = online.groupby(['CohortMonth', 'CohortIndex'])

cohort_data = grouping_count['CustomerID'].apply(pd.Series.nunique)

cohort_data = cohort_data.reset_index()

cohort_counts = cohort_data.pivot(index='CohortMonth',

                                  columns='CohortIndex',

                                  values='CustomerID')

print(cohort_counts.head())
#Retention analysis 

cohort_sizes = cohort_counts.iloc[:,0]

retention = cohort_counts.divide(cohort_sizes, axis=0)

retention.round(3) * 100

retention.index = retention.index.strftime('%m-%Y')
#Average quantity analysis 

grouping_qty = online.groupby(['CohortMonth', 'CohortIndex'])

cohort_data_qty = grouping_qty['Quantity'].mean()

cohort_data_qty = cohort_data_qty.reset_index()

average_quantity = cohort_data_qty.pivot(index='CohortMonth',

                                     columns='CohortIndex',

                                     values='Quantity')

average_quantity.index = average_quantity.index.strftime('%m-%Y')



#Average price analysis 

grouping_price = online.groupby(['CohortMonth', 'CohortIndex'])

cohort_data_price = grouping_price['Price'].mean()

cohort_data_price = cohort_data_price.reset_index()

average_price = cohort_data_price.pivot(index='CohortMonth',

                                     columns='CohortIndex',

                                     values='Price')

average_price.index = average_price.index.strftime('%m-%Y')
# --Plot heatmaps using seaborn--



# Plot retention rates

plt.figure(figsize=(20, 8))

plt.title('Retention Rates')

sns.heatmap(data = retention, annot = True, fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap = 'Reds')

plt.show()



# Plot average quantity

plt.figure(figsize=(20, 8))

plt.title('Average Quantity')

sns.heatmap(data = average_quantity, annot=True, cmap='Greens')

plt.show()



# Plot average price

plt.figure(figsize=(20, 8))

plt.title('Average Price')

sns.heatmap(data = average_price, annot=True, cmap='Blues')

plt.show()
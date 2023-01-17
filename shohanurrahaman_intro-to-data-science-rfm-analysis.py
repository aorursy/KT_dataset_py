# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/sample-sales-data/sales_data_sample.csv', encoding='unicode_escape')

display(df.head(5))
cols = ['CUSTOMERNAME','ORDERDATE','ORDERNUMBER','SALES']

df = df[cols]

print(df.head(5))
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])



#group the data by CUSTOMERNAME and only retrive the ORDERDATE column 

recent_order = df.groupby('CUSTOMERNAME')['ORDERDATE'].max()



most_recent = df['ORDERDATE'].max()



def subtract_date(date):

    days = (most_recent - date).days

    return days



recency = recent_order.apply(subtract_date)



print('Recency days : ', recency)

#print(recent_order.head(10))

#print(df.groupby(['CUSTOMERNAME','ORDERDATE']).count().head())
frequency = df.groupby(['CUSTOMERNAME','ORDERNUMBER']).size()

frequency = frequency.groupby('CUSTOMERNAME').size()

print(frequency.head())
#groupby the CUSTOMERNAME and only retrive the SALES and sum

monetary = df.groupby('CUSTOMERNAME')['SALES'].sum()

print(monetary.head())
rfm = pd.DataFrame()

rfm['recency'] = recency 

rfm['frequency'] = frequency 

rfm['monetary'] = monetary 



print(rfm.head())
quantile_df = rfm.quantile([0.25,0.50,0.75])

display(quantile_df)
def quantile_classes(x, quantile_value, attribute):

    if attribute == 'recency':

        if x <= quantile_value.loc[0.25,attribute]: # receny is less than 0.25%

            return '4'

        elif x >= quantile_value.loc[0.25,attribute] and x <= quantile_value.loc[0.50,attribute]: # recency is larger than 25%

            return '3'

        elif x >= quantile_value.loc[0.50,attribute] and x <= quantile_value.loc[0.75,attribute]:

            return '2'

        else:

            return '1'

    else:

        #frequncy and monetary 

        if x <= quantile_value.loc[0.25,attribute]: # frequncy/monetary is less than 0.25%

            return '1'

        elif x >= quantile_value.loc[0.25,attribute] and x <= quantile_value.loc[0.50,attribute]: # frequncy/monetary is larger than 25%

            return '2'

        elif x >= quantile_value.loc[0.50,attribute] and x <= quantile_value.loc[0.75,attribute]:

            return '3'

        else:

            return '4'

        



#convert rfm table raw value to class 

rfm['recency_class'] = rfm['recency'].apply(quantile_classes, args = (quantile_df,'recency'))

rfm['frequency_class'] = rfm['frequency'].apply(quantile_classes, args = (quantile_df,'frequency'))

rfm['monetary_class'] = rfm['monetary'].apply(quantile_classes, args = (quantile_df,'monetary'))



display(rfm.head())
#join the string values

rfm['rfm_comb'] = rfm['recency_class'] + rfm['frequency_class'] + rfm['monetary_class']



#convert to numeric value 

rfm['rfm_comb'] = pd.to_numeric(rfm['rfm_comb'])



#sort values 

rfm = rfm.sort_values(by=['rfm_comb'], ascending=False)



#display top 10 customer 

display(rfm.head(10))
# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas_profiling as pp
import seaborn as sns

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading the data

data=pd.read_csv("../input/sales_data_sample.csv",encoding='unicode_escape')
data.shape
data.info()
data.head()
# Dropping Unnecessary columns 
temp=['ADDRESSLINE1','ADDRESSLINE2','POSTALCODE', 'TERRITORY', 'PHONE', 'CITY' , 'STATE','CONTACTFIRSTNAME', 'CONTACTLASTNAME' ]
data.drop(temp,axis=1,inplace=True)
# Regrouping product code.
data['PRODUCTINITIAL'] = data['PRODUCTCODE'].str[:3]
data.drop('PRODUCTCODE',axis=1,inplace=True)
# Recheck columns
data.info()
# Let's plot the data to get more insight.

plt.rcParams['figure.figsize'] = [18, 16]
data.plot(kind="box",subplots=True,layout=(4,4),sharex=False,sharey=False)
plt.show()
plt.rcParams['figure.figsize'] = [18, 16]
data.plot(kind="density",subplots=True,layout=(4,4),sharex=False,sharey=False)
plt.show()
# Checking null values
data.isnull().sum()
plt.rcParams['figure.figsize'] = [4, 4]
sns.regplot(x="YEAR_ID",y="QTR_ID",data=data)
plt.show()
data['STATUS'].value_counts()
sns.countplot(y='STATUS',data=data,hue='YEAR_ID')
sns.countplot(y='STATUS',data=data,hue='QTR_ID')
# Comparing sales for each year(Quaterwise)

data1=data.groupby(['YEAR_ID','QTR_ID']).agg({'SALES': lambda x: x.sum() })
print(data1.info())
print(data1.head())
data1.reset_index(inplace=True)
data1.head()

sns.factorplot(y='SALES', x='QTR_ID',data=data1,kind="bar" ,hue='YEAR_ID')
import warnings
warnings.filterwarnings('ignore')
temp=['CUSTOMERNAME', 'ORDERNUMBER', 'ORDERDATE', 'SALES']
RFM_data=data[temp]
RFM_data.shape
RFM_data.head()
RFM_data['ORDERDATE'] = pd.to_datetime(RFM_data['ORDERDATE'])
RFM_data['ORDERDATE'].max()
import datetime as dt
NOW = dt.datetime(2005,5,31)
RFM_table=RFM_data.groupby('CUSTOMERNAME').agg({'ORDERDATE': lambda x: (NOW - x.max()).days, # Recency
                                                'ORDERNUMBER': lambda x: len(x.unique()), # Frequency
                                                'SALES': lambda x: x.sum()})    # Monetary 

RFM_table['ORDERDATE'] = RFM_table['ORDERDATE'].astype(int)

RFM_table.rename(columns={'ORDERDATE': 'recency', 
                         'ORDERNUMBER': 'frequency',
                         'SALES': 'monetary_value'}, inplace=True)
RFM_table.head()
quantiles = RFM_table.quantile(q=[0.25,0.5,0.75])
quantiles

# Converting quantiles to a dictionary, easier to use.
quantiles = quantiles.to_dict()
quantiles 
RFM_Segment = RFM_table.copy()
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def R_Class(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
    
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FM_Class(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
RFM_Segment['R_Quartile'] = RFM_Segment['recency'].apply(R_Class, args=('recency',quantiles,))
RFM_Segment['F_Quartile'] = RFM_Segment['frequency'].apply(FM_Class, args=('frequency',quantiles,))
RFM_Segment['M_Quartile'] = RFM_Segment['monetary_value'].apply(FM_Class, args=('monetary_value',quantiles,))
RFM_Segment['RFMClass'] = RFM_Segment.R_Quartile.map(str) \
                            + RFM_Segment.F_Quartile.map(str) \
                            + RFM_Segment.M_Quartile.map(str)
#Who are my best customers? (BY RFMClass = 444)
RFM_Segment[RFM_Segment['RFMClass']=='444'].sort_values('monetary_value', ascending=False).head(5)
#Which customers are at the verge of churning?
#Customers who's recency value is low

RFM_Segment[RFM_Segment['R_Quartile'] <= 2 ].sort_values('monetary_value', ascending=False).head(5)
#Who are lost customers?
#Customers who's recency, frequency as well as monetary values are low 

RFM_Segment[RFM_Segment['RFMClass']=='111'].sort_values('recency',ascending=False).head(5)
#Who are your loyal customers?
#Customers with high frequency value

RFM_Segment[RFM_Segment['F_Quartile'] >= 3 ].sort_values('monetary_value', ascending=False).head(5)
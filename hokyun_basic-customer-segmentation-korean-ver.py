# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt



# import matplotlib

# matplotlib.rc('font', family='Malgun Gothic') # 한글 폰트 설정



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
commerce = pd.read_csv('../input/data.csv', encoding="ISO-8859-1",

                      parse_dates=['InvoiceDate'], dtype={'CustomerID':str})



print(commerce.info())

print("DataFrame Dimensions: ", commerce.shape)

print("\n")

print("null values: {} => {}".format(commerce.columns.values ,commerce.isnull().sum().values))

# 75% of data left

commerce = commerce[~pd.isnull(commerce.CustomerID)]



commerce.info()

commerce.head(2)
# Add Invoice Month

commerce['InvoiceMonth'] = commerce['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, 1))



# Add Acquisition Month

commerce['AcquisitionMonth'] = commerce.groupby(by=['CustomerID'])['InvoiceMonth'].transform('min')



# Top 5 rows

commerce.head()
# Add Cohort Index by assigning time offset value

years_diff = commerce.InvoiceMonth.dt.year - commerce.AcquisitionMonth.dt.year

months_diff = commerce.InvoiceMonth.dt.month - commerce.AcquisitionMonth.dt.month



time_offset = years_diff * 12 + months_diff



# Add column

commerce['CohortIndex'] = time_offset

# Top 2 rows

commerce.head(2)
# Add Total Amount

commerce['Amt'] = commerce.UnitPrice * commerce.Quantity
# Acquisition

ac_cohort = commerce.pivot_table(index='AcquisitionMonth', columns='CohortIndex',

                                 values='CustomerID', aggfunc=lambda x: x.nunique())

ac_cohort
# Retention Rate Cohort



# 월별 최초 유입 고객수

size = ac_cohort.iloc[:,0]



# 잔존율

retention = ac_cohort.divide(size, axis=0)

retention.round(2) * 100
retention.index = retention.index.astype(str)
# Retention Rate 

plt.figure(figsize=(10,8))

plt.title('Customer Retention Rate', fontsize=20)

sns.heatmap(data=retention, annot=True, fmt='0.0%', vmin=0.0, vmax=0.5, cmap='YlGnBu')
print("first date: ", commerce.InvoiceDate.min(), 

      "\nlast date: ",commerce.InvoiceDate.max())
snapshot_date = commerce.InvoiceDate.max() + dt.timedelta(days=1)

snapshot_date
# RFM Table

RFM_Table = commerce.pivot_table(index='CustomerID', values=['InvoiceDate', 'InvoiceNo', 'Amt'],

                                 aggfunc={'InvoiceDate': lambda x: (snapshot_date - x.max()).days,

                                          'InvoiceNo': 'count',

                                          'Amt': 'sum'}).round()

RFM_Table.rename(columns={'InvoiceDate': 'Recency',

                          'InvoiceNo': 'Frequency',

                          'Amt': 'MonetaryValue'}, inplace=True)



RFM_Table = RFM_Table[['Recency', 'Frequency', 'MonetaryValue']]



RFM_Table.sort_values(by=['MonetaryValue'], ascending=False).head()
RFM_Table.sort_values('MonetaryValue', ascending=True).head(5)
# Describe Recency Value

RFM_Table.Recency.describe()
# Describe Frequency Value

RFM_Table.Frequency.describe()
# Describe MonetaryValue Value

RFM_Table.MonetaryValue.describe()
# 실수값을 기준으로 등급(카테고리)를 나누겠습니다. 

# 사분위수를 기준으로 갯수가 똑같은 그룹을 만들기 위해서 qcut을 이용합니다.

R_labels = range(4,0,-1); F_labels = range(1,5); M_labels = range(1,5)



R_score = pd.qcut(RFM_Table.Recency, 4, labels=R_labels)

F_score = pd.qcut(RFM_Table.Frequency, 4, labels=F_labels)

M_score = pd.qcut(RFM_Table.MonetaryValue, 4, labels=M_labels)



RFM_Table['R'] = pd.to_numeric(R_score)

RFM_Table['F'] = pd.to_numeric(F_score)

RFM_Table['M'] = pd.to_numeric(M_score)
RFM_Table['RFM_Segment'] = RFM_Table.R.astype(str) + RFM_Table.F.astype(str) + RFM_Table.M.astype(str)

RFM_Table['RFM_Score'] = RFM_Table[['R','F','M']].sum(axis=1)



RFM_Table.head(3)
RFM_Summary = RFM_Table.reset_index().pivot_table(index='RFM_Score', 

                                                  values=['CustomerID','Recency','Frequency','MonetaryValue',

                                                          'R', 'F', 'M'],

                                                  aggfunc={'CustomerID': 'count',

                                                           'Recency': 'mean',

                                                           'Frequency': 'mean',

                                                           'MonetaryValue': 'mean',

                                                           'R': 'mean',

                                                           'F': 'mean',

                                                           'M': 'mean'}).round(1)

# 각 Segment별 비율을 추가합니다. 전체 고객수로 각 세그먼트를 나누어줍니다.

total_customer = RFM_Summary['CustomerID'].sum()

RFM_Summary['Percentage'] = (RFM_Summary['CustomerID'] / total_customer * 100).round(1)



# 컬럼 순서를 조정합니다.

RFM_Summary = RFM_Summary[['CustomerID', 'Percentage', 'R', 'F', 'M', 

                           'Recency','Frequency','MonetaryValue']]



# 컬럼명을 변경합니다.

RFM_Summary.rename(columns={'CustomerID': 'Count',

                            'Recency': 'Recency Mean',

                            'Frequency': 'Frequency Mean',

                            'MonetaryValue': 'MonetaryValue Mean'}, inplace=True) 

# Count 내림차순 정렬

RFM_Summary.sort_values('Count', ascending=False, inplace=True)



# 출력

RFM_Summary
No_Cust = RFM_Table.groupby('RFM_Score').agg({'R': 'count'})

No_Cust = No_Cust.reset_index().sort_values(['RFM_Score'], ascending=False)

No_Cust.rename(columns={'R': '# of Customer'}, inplace=True)



# barplot

plt.figure(figsize=(8,6))

ax = sns.barplot(data=No_Cust, x='RFM_Score', y='# of Customer',

                 order=No_Cust['RFM_Score'])

for i in ax.patches:

    ax.annotate(i.get_height(), (i.get_x()+0.1, i.get_height()+5))

    
# Receny X Frequency Table

RxF = RFM_Table.reset_index().pivot_table(index='R', columns='F', 

                                          values='CustomerID', aggfunc='count')

total = RxF.sum().sum()

print(RxF)
# Percentage table

RxF = (RxF / total).round(3)

RxF
plt.figure(figsize=(8,6))

plt.title('Recency X Frequency', fontsize=20)

ax = sns.heatmap(data=RxF, annot=True, fmt='0.0%', cmap='YlGnBu')

ax.xaxis.tick_top()

ax.xaxis.set_label_position('top')
Ave_Recency = RFM_Summary.iloc[:,[5]].sort_index(ascending=False).reset_index()

Ave_Recency.RFM_Score = Ave_Recency.RFM_Score.astype('category')



plt.figure(figsize=(8,6))

plt.title('Average Receny days', fontsize=20)

ax = sns.barplot(data=Ave_Recency,x='Recency Mean', y='RFM_Score', 

                 order=Ave_Recency['RFM_Score'])

for i in ax.patches:

    ax.annotate(i.get_width(), (i.get_width()+0.5, i.get_y()+0.5))
Ave_Money = RFM_Summary.iloc[:,[7]].sort_index(ascending=False).reset_index()

Ave_Money.RFM_Score = Ave_Recency.RFM_Score.astype('category')



plt.figure(figsize=(8,6))

plt.title('Average Monetary Value', fontsize=20)

ax = sns.barplot(data=Ave_Money,x='MonetaryValue Mean', y='RFM_Score', 

                 order=Ave_Money['RFM_Score'])

for i in ax.patches:

    ax.annotate(i.get_width(), (i.get_width()+0.5, i.get_y()+0.5))
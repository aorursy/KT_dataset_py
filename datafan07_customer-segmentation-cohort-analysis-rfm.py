import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

#

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



import os

print(os.listdir("../input"))
df = pd.read_excel('/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')
df.sample(5)
display(df.describe())
plt.subplots(figsize=(10, 8))

sns.heatmap(df.isnull(), yticklabels = False,cbar = False, cmap='cubehelix')

plt.show()
df.drop_duplicates(inplace=True)

df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'].notnull())]
df.shape
def get_month(x): return dt.datetime(x.year, x.month, 1)

df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)

grouping = df.groupby('CustomerID')['InvoiceMonth']

df['CohortMonth'] = grouping.transform('min')
def get_date_int(df, column):

    year = df[column].dt.year

    month = df[column].dt.month

    day = df[column].dt.day

    return year, month, day
invoice_year, invoice_month, invoice_day = get_date_int(df, 'InvoiceMonth')

cohort_year, cohort_month, cohort_day = get_date_int(df, 'CohortMonth')



years_diff = invoice_year - cohort_year

months_diff = invoice_month - cohort_month



df['CohortIndex'] = years_diff * 12 + months_diff + 1



df.tail()
grouping = df.groupby(['CohortMonth', 'CohortIndex'])



cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)



cohort_data = cohort_data.reset_index()



cohort_counts = cohort_data.pivot(index='CohortMonth',

                                 columns='CohortIndex',

                                 values='CustomerID')
cohort_sizes = cohort_counts.iloc[:,0]

retention = cohort_counts.divide(cohort_sizes, axis=0)

retention.index=retention.index.date
#sns.set()

plt.style.use('fivethirtyeight')

plt.figure(figsize=(12,10))

plt.title('Retention Rates')



sns.heatmap(data = retention,

           annot = True,

           fmt = '.0%',

           vmin= 0.0,

           vmax=0.5,

           cmap='summer_r')

plt.show()
df['InvoiceDay'] = df['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, x.day))



act_date = max(df['InvoiceDay'] + dt.timedelta(1))

df['TotalSum'] = df['Quantity'] * df['UnitPrice']

df.drop(['CohortMonth', 'InvoiceMonth', 'CohortIndex'], axis=1, inplace=True)

df.head()
rfm = df.groupby('CustomerID').agg({

    'InvoiceDate' : lambda x: (act_date - x.max()).days,

    'InvoiceNo' : 'count',

    'TotalSum' : 'sum'

})



rfm.rename(columns = {'InvoiceDate' : 'Recency', 

                      'InvoiceNo' : 'Frequency', 

                      'TotalSum' : 'MonetaryValue'}, inplace = True)

rfm.head()
r_labels = range(3, 0, -1)

r_groups = pd.qcut(rfm.Recency, q = 3, labels = r_labels)



f_labels = range(1, 4)

f_groups = pd.qcut(rfm.Frequency, q = 3, labels = f_labels)



m_labels = range(1, 4)

m_groups = pd.qcut(rfm.MonetaryValue, q = 3, labels = m_labels)
rfm['R'] = r_groups.values

rfm['F'] = f_groups.values

rfm['M'] = m_groups.values



rfm = rfm.assign(R=r_groups,F=f_groups,M=m_groups)



rfm['RFM_Segment'] = rfm.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis = 1)

rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis = 1)

rfm.head()
rfm_agg = rfm.groupby('RFM_Score').agg({

    'Recency' : 'mean',

    'Frequency' : 'mean',

    'MonetaryValue' : ['mean', 'count']

})



rfm_agg.rename(columns = {'mean' : 'Mean','count' : 'Count'},

               inplace = True)



rfm_agg.round(2).head()
score_labels = ['Bronze', 'Silver', 'Gold']

score_groups = pd.qcut(rfm.RFM_Score, q = 3, labels = score_labels)

rfm['Robust RFM Level'] = score_groups.values



rfm.head()
rfm.describe()
def neg_to_zero(x):

    if x <= 0:

        return 1

    else:

        return x

rfm['Recency'] = [neg_to_zero(x) for x in rfm.Recency]



rfm_log = rfm[['Recency', 'Frequency', 'MonetaryValue']].apply(np.log, axis = 1)
scaler = StandardScaler()

scaler.fit(rfm_log)



rfm_normalized= scaler.transform(rfm_log)
rfm_scaled = pd.DataFrame(rfm_normalized, index = rfm.index, columns = rfm_log.columns)

rfm.describe()
cont_features = ['Recency', 'Frequency', 'MonetaryValue']





fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(18,18))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cont_features):

    sns.distplot(rfm[feature],  hist=True, color='#e74c3c', ax=ax[i][0])    

    sns.distplot(rfm_scaled[feature], hist=True, color='#e74c3c', ax=ax[i][1])

    

    ax[i][0].set_title(f'Distribution of Unscaled {feature}', size=30, y=1.05)

    ax[i][1].set_title(f'Distribution of Scaled {feature}', size=30, y=1.05)

    

plt.tight_layout()      

plt.show()
plt.figure(figsize=(10,8))



kls = np.arange(1,6,1)

inertias=[]

for k in kls :

    knc = KMeans(n_clusters=k, random_state=42)

    knc.fit(rfm_scaled)

    inertias.append(knc.inertia_)





fig, ax = plt.subplots(figsize=(12, 8))

plt.plot(kls, inertias,'--o', markersize=22, color='#e74c3c')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.xticks(kls)

plt.show()
kc = KMeans(n_clusters=3, random_state=42)

kc.fit(rfm_normalized)

rfm['RFM Cluster'] = kc.labels_



rfm_s=rfm.groupby('RFM Cluster').agg({'Recency': 'mean','Frequency': 'mean',

                                         'MonetaryValue': ['mean', 'count']})





rfm_s.rename(columns = {'mean' : 'Mean','count' : 'Count'},

               inplace = True)







display(rfm_s.style.background_gradient(cmap='summer_r'))
rfm['RFM Cluster']=rfm['RFM Cluster'].map({0: 'K_Bronze', 1: 'K_Gold',2:'K_Silver'})

rfm.sample(10)
rfm_scaled['RFM Cluster'] = kc.labels_

rfm_scaled['Robust RFM Level'] = rfm['Robust RFM Level']

rfm_scaled.reset_index(inplace = True)

rfm_scaled['RFM Cluster']=rfm_scaled['RFM Cluster'].map({0: 'K_Bronze', 1: 'K_Gold',2:'K_Silver'})





rfm_melted = pd.melt(frame= rfm_scaled, id_vars= ['CustomerID', 'Robust RFM Level',

                                                  'RFM Cluster'], var_name = 'Metrics', value_name = 'Value')



rfm_melted.head()
fig, ax = plt.subplots(nrows=2, figsize=(15,8))

plt.subplots_adjust(right=1.5, top=1.25)





sns.lineplot(x = 'Metrics', y = 'Value', hue = 'Robust RFM Level', data = rfm_melted, ax=ax[0])

sns.lineplot(x = 'Metrics', y = 'Value', hue = 'RFM Cluster', data = rfm_melted, ax=ax[1])



ax[0].set_title('Snake Plot of RFM Level', size=25)

ax[1].set_title('Snake Plot of RFM Cluster', size=25)

ax[0].legend(loc='upper right', prop={'size': 12})

ax[1].legend(loc='upper right', prop={'size': 12})



plt.show()
total_avg = rfm.iloc[:, 0:3].mean()



cluster_avg = rfm.groupby('Robust RFM Level').mean().iloc[:, 0:3]

prop_rfm = cluster_avg/total_avg - 1



cluster_avg_K = rfm.groupby('RFM Cluster').mean().iloc[:, 0:3]

prop_rfm_K = cluster_avg_K/total_avg - 1
fig, ax = plt.subplots(ncols=2, figsize=(12,8))

sns.heatmap(prop_rfm, cmap= 'summer_r', fmt= '.2f', annot = True, ax=ax[0])

sns.heatmap(prop_rfm_K, cmap= 'summer_r', fmt= '.2f', annot = True, ax=ax[1])



ax[0].set_title('Heatmap of Robust RFM Level', size=15)

ax[1].set_title('Heatmap of RFM Cluster', size=15)



plt.show()
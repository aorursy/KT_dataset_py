
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from datetime import timedelta


# In[2]:

source = pd.read_csv('../input/consumer_complaints.csv')


# In[47]:

data = source.copy()
#Type casting
data['date_received'] = pd.to_datetime(data['date_received'], format='%m/%d/%Y')
data['date_sent_to_company'] = pd.to_datetime(data['date_sent_to_company'], format='%m/%d/%Y')

#Check size and dtypes
print(data.shape)
print(data.dtypes)


# In[48]:

#CLEANING THE DATA

#Correct dates where the submitted date is less than the received date by making submitted date = received date
data['processing_time'] = data['date_sent_to_company']-data['date_received']
data.loc[data['processing_time']<timedelta(days=0),'date_sent_to_company'] = data['date_received']
data.drop('processing_time',axis=1, inplace=True)


# In[49]:

#Features
data['month_received'] = data['date_received'].dt.month
data['year_received'] = data['date_received'].dt.year
data['month_sent_to_company'] = data['date_sent_to_company'].dt.month
data['year_sent_to_company'] = data['date_sent_to_company'].dt.year
data['processing_time'] = data['date_sent_to_company']-data['date_received']


A = [d.date() for d in data['date_received']]
B = [d.date() for d in data['date_sent_to_company']]
data['processing_time'] = np.busday_count(A, B)


# In[75]:

#More Features
grp = data.groupby('year_received')
days_factor = 365.25/(grp['date_received'].max()-grp['date_received'].min() + timedelta(days=1)).dt.days.astype('int')
days_factor.name = 'days_factor'
cleaned = data.join(days_factor,on='year_received',how='left')
cleaned.head()


# In[76]:

data = cleaned.copy()


# In[77]:

data.head()


# In[8]:

data['product'].unique()


# In[9]:

data['sub_product'].unique()


# In[10]:

data['company_response_to_consumer'].unique()


# In[11]:

grp = data.pivot_table(index='company_response_to_consumer',columns='consumer_disputed?',values='date_received',aggfunc='count')
grp


# In[12]:

grp.plot(kind='barh',stacked=True, title='Customer Dispute by Response')


# In[13]:

pct = grp['Yes']/(grp['No']+grp['Yes'])
pct.plot(kind='barh',title='Customer Dispute Rate by Response')


# In[14]:

grp = data.groupby('submitted_via')
grp.size()


# In[15]:

grp.size().plot(kind='barh',title='Preferred Complaint Methods')


# In[16]:

grp = data.groupby('company')
grp = grp.size().sort_values(ascending=False)
#top10 = grp[10::-1]
top10 = (grp[:10])[::-1]
top10.plot(kind='barh',title='Most Common Companies to Complaint Against')


# In[168]:

grp = data.groupby(['company','year_received'])
grp = grp.size().mul(grp['days_factor'].min()).unstack().fillna(0)
grp['total'] = grp.sum(axis=1)
grp = grp.sort_values('total',ascending=False)
grp = grp.div(grp['total'].max(), axis=0)
grp = grp.drop('total',axis=1)
top10 = (grp[:10])[::-1]
ind = np.arange(10)
series1 = top10.loc[:,2011].values
cumulative = top10.cumsum(axis=1)
color = sns.color_palette("hls", 6)
for i in np.arange(5,-1,-1):
    plt.barh(ind, cumulative.iloc[:,i], color=color[i], label=2011+i)
    
plt.yticks(ind + .85/2., top10.index)
plt.title('Most Common Companies to Complain Against, Annual Trend')
plt.legend(loc='best')


# In[17]:

grp = data.groupby('year_received')
grp.size().plot(kind='bar',title='Volume of Complaints by Year (2011/2016 partial)')


# In[79]:

grp = data.groupby('year_received')
complaint_rate = grp.size().mul(grp['days_factor'].min())
complaint_rate.plot(kind='bar',title='Rate of Complaints/Day by Year (2011/2016 partial)')


# In[19]:

mask = (data['year_received']>2011) & (data['year_received']<2016)
grp = data[mask].groupby('month_received')
grp.size().plot(kind='bar',title='Total Complaints by Month (2012-2015)')


# In[20]:

processing_time_counts = data['processing_time'].value_counts().sort_index()
total = sum(processing_time_counts)
processing_time_counts_CDF = processing_time_counts.cumsum()/total
ax = processing_time_counts_CDF.plot(kind='line',title='Processing Time CDF',xlim=(0,14),ylim=(0,1))


# In[166]:

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
years = np.arange(2011,2017,1)
for y in years:
    processing_time_counts = data[data['year_received']==y]['processing_time'].value_counts().sort_index()
    total = sum(processing_time_counts)
    processing_time_counts_CDF = processing_time_counts.cumsum()/total
    ax.plot(processing_time_counts_CDF.index.values.astype('timedelta64[D]'),processing_time_counts_CDF.values,label=y,color=color[y-2011])
    #ax = processing_time_counts_CDF.plot(kind='line',title='Processing Time CDF',xlim=(0,14),ylim=(0,1))


ax.axis([0,14,0,1])
ax.legend(loc='best')
ax.set_xlabel('Business Days')
fig.suptitle('Procesing Time CDF')


# In[ ]:





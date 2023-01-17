import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/import-data-and-convert-into-table-format/df_train.csv', low_memory=False)
test = pd.read_csv('../input/import-data-and-convert-into-table-format/df_test.csv', low_memory=False)
sub = pd.read_csv('../input/ga-customer-revenue-prediction/sample_submission.csv', low_memory=False)

print('Train data: \nRows: {}\nCols: {}'.format(train.shape[0],train.shape[1]))
print(train.columns)

print('\nTest data: \nRows: {}\nCols: {}'.format(test.shape[0],test.shape[1]))
print(test.columns)

print('\nSubmission format: \nRows: {}\nCols: {}'.format(sub.shape[0],sub.shape[1]))
print(sub.columns)
train.head()
test.head()
sub.head()
print('Unique visitor IDs in training data: {}'.format(train['fullVisitorId'].nunique()))
#print('Unique visitor IDs in test data: {}'.format(test['fullVisitorId'].nunique()))
train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
temp = train.groupby('fullVisitorId')['totals.transactionRevenue'].sum().reset_index()
print('Percentage of visitors with revenue is {:,.2%}'.format(temp[temp['totals.transactionRevenue'] > 0]['fullVisitorId'].count() / train['fullVisitorId'].nunique()))
train['revenue_flag'] = train['totals.transactionRevenue'].apply(lambda x: 1 if x>0 else 0)
mobile_revenue = train.pivot_table(train, index=['device.isMobile'], columns=['revenue_flag'], aggfunc=len).reset_index()[['device.isMobile','totals.transactionRevenue']]
mobile_revenue.columns = ['device.isMobile','No Revenue', 'Revenue']
mobile_revenue['Revenue %'] = mobile_revenue['Revenue'] / (mobile_revenue['Revenue'] + mobile_revenue['No Revenue'])

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x='Revenue %', y='device.isMobile', data=mobile_revenue, color="lightskyblue", orient="h")

for p in ax.patches:
    ax.text(p.get_width() + 0.0006, 
            p.get_y() + (p.get_height()/2), 
            '{:,.1%}'.format(p.get_width()),
            ha="center")
    
ax.set_ylabel('Device is Mobile?', size=14, color="#0D47A1")
ax.set_xlabel('% of transactions with revenue', size=14, color="#0D47A1")
ax.set_title('Percentage of transactions with revenue by device type', size=18, color="#0D47A1")

plt.show()
mobile_revenue = train.pivot_table(train, index=['device.isMobile'], columns=['revenue_flag'], aggfunc=np.mean).reset_index()[['device.isMobile','totals.transactionRevenue']]
mobile_revenue.columns = ['device.isMobile','Avg Revenue']

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x='Avg Revenue', y='device.isMobile', data=mobile_revenue, color="lightskyblue", orient="h")

for p in ax.patches:
    ax.text(p.get_width() + 10000000, 
            p.get_y() + (p.get_height()/2), 
            '${:,.0f}'.format(p.get_width()),
            ha="center")
    
ax.set_ylabel('Device is Mobile?', size=14, color="#0D47A1")
ax.set_xlabel('Average revenue', size=14, color="#0D47A1")
ax.set_title('Average revenue of by device type', size=18, color="#0D47A1")

plt.show()
browser_revenue = train.pivot_table(train, index=['device.browser'], columns=['revenue_flag'], aggfunc=len).reset_index()[['device.browser','totals.transactionRevenue']]
browser_revenue.columns = ['device.browser','No Revenue', 'Revenue']
browser_revenue.fillna(0, inplace=True)
browser_revenue['Transactions'] = browser_revenue['Revenue'] + browser_revenue['No Revenue']
browser_revenue['Revenue %'] = browser_revenue['Revenue'] / browser_revenue['Transactions']
browser_revenue = browser_revenue.sort_values('Transactions',ascending=False).head(15)

fig, ax = plt.subplots(1, 2, figsize=(8,8), sharey=True)
a = sns.barplot(x='Transactions', y='device.browser', data=browser_revenue, color="lightskyblue", orient="h", ax=ax[0])
b = sns.barplot(x='Revenue %', y='device.browser', data=browser_revenue, color="lightskyblue", orient="h", ax=ax[1])

for p in ax[0].patches:
    ax[0].text(p.get_width() + 75000, 
            p.get_y() + (p.get_height()/2), 
            '{:,.0f}'.format(p.get_width()),
            ha="center")
    
for p in ax[1].patches:
    ax[1].text(p.get_width() + 0.001, 
            p.get_y() + (p.get_height()/2), 
            '{:,.1%}'.format(p.get_width()),
            ha="center")
    
ax[0].set_ylabel('Browsers', size=14, color="#0D47A1")
ax[0].set_xlabel('Total Number of Transactions', size=14, color="#0D47A1")

ax[1].set_ylabel('', size=14, color="#0D47A1")
ax[1].set_xlabel('% of Transactions with revenue', size=14, color="#0D47A1")

plt.show()
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('../input/train.csv', parse_dates=['project_submitted_datetime'])
test = pd.read_csv('../input/test.csv', parse_dates=['project_submitted_datetime'])
resources = pd.read_csv('../input/resources.csv')
sample_sub = pd.read_csv('../input/sample_submission.csv')
print('Train shape: {}\nTest shape: {}\nResources shape: {}'.format(train.shape, test.shape, resources.shape))
print('Approved projects: {}%'.format(round(train['project_is_approved'].mean()*100, 2)))
print('Train set - First 5 rows')
train.head()
print('Resources - First 5 rows')
resources.head()
train = pd.merge(train, resources, on='id', how='left', sort=False)
test = pd.merge(test, resources, on='id', how='left', sort=False)
def absolute_value(val):
    return str(np.round(val,2)) + '%'

plt.figure(figsize=(8,8))
plt.pie(train['project_is_approved'].value_counts(), explode = (0.1, 0),
        labels=['Approved Projects', 'Not Approved Projects'], autopct=absolute_value, shadow=True)
plt.title('Target "project_is_approved" distribution over training data', fontsize=12)
plt.legend(fontsize=12)
plt.show()
print('--- Price description ---')
train['price'].describe()
plt.figure(figsize=(12,8))

plt.subplot(121)
p = plt.pie((train['quantity']>1).value_counts(), explode = (0.1, 0),
        labels=['ONE Item', 'More than ONE Item'], autopct=absolute_value, shadow=True, startangle=-150)
p = plt.title('Feature "quantity" distribution over TRAINING data', fontsize=12)
p = plt.legend(fontsize=12)

plt.subplot(122)
p = plt.pie((test['quantity']>1).value_counts(), explode = (0.1, 0),
        labels=['ONE Item', 'More than ONE Item'], autopct=absolute_value, shadow=True, startangle=-150)
p = plt.title('Feature "quantity" distribution over TESTING data', fontsize=12)
p = plt.legend(fontsize=12)

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(211)
p = sns.distplot(np.log1p(train['price']), bins=100, label='Price (ONE item)')
p = plt.title('Projects\' price (just one \'item\')',fontsize=12)
p = plt.ylim(0, 0.5)
p = plt.xlim(0,10)
p = plt.legend(fontsize=12)
p = plt.ylabel('Normalized Frequency', fontsize=12)
p = plt.xlabel('Log1p price', fontsize=12)

plt.subplot(212)
p1 = sns.distplot(np.log1p(train['price'] * train['quantity']), bins=100, label='Price (ALL items)')
p1 = plt.title('Projects\' price (considering \'quantity\')',fontsize=12)
p1 = plt.ylim(0, 0.5)
p1 = plt.xlim(0,10)
p1 = plt.legend(fontsize=12)
p1 = plt.ylabel('Normalized Frequency', fontsize=12)
p1 = plt.xlabel('Log1p price', fontsize=12)

plt.show()
plt.figure(figsize=(12,5))

sns.distplot(np.log1p(train['price']), bins=100, label='Price (ONE item)', hist=False)
sns.distplot(np.log1p(train['price'] * train['quantity']), bins=100, label='Price (ALL items)', hist=False)
plt.title('Projects\' price (just one \'item\' and considering \'quantity\')',fontsize=12)
plt.legend(fontsize=12)
plt.ylim(0, 0.5)
plt.ylabel('Normalized Frequency', fontsize=12)
plt.xlabel('Log1p price', fontsize=12)

plt.show()
plt.figure(figsize=(12,6))

plt.subplot(121)
p = sns.distplot(np.log1p(train[train['project_is_approved']==1]['price']), bins=75, label='Approved Projects (ONE)')
p = sns.distplot(np.log1p(train[train['project_is_approved']==0]['price']), bins=75, label='NOT Approved Projects (ONE)')
p = plt.xlim(0,10)
p = plt.ylim(0,0.5)
p = plt.title('Projects\' price ONE item',fontsize=12)
p = plt.legend(fontsize=12)
p = plt.ylabel('Normalized Frequency', fontsize=12)
p = plt.xlabel('Log1p price', fontsize=12)

plt.subplot(122)
p1 = sns.distplot(np.log1p(train[train['project_is_approved']==1]['price'] * train[train['project_is_approved']==1]['quantity']), bins=75, label='Approved Projects (ALL)')
p1 = sns.distplot(np.log1p(train[train['project_is_approved']==0]['price'] * train[train['project_is_approved']==0]['quantity']), bins=75, label='NOT Approved Projects (ALL)')
p1 = plt.xlim(0,10)
p1 = plt.ylim(0,0.5)
p1 = plt.title('Projects\' price considering ALL items',fontsize=12)
p1 = plt.legend(fontsize=12)
p1 = plt.ylabel('Normalized Frequency', fontsize=12)
p1 = plt.xlabel('Log1p price', fontsize=12)

plt.show()
plt.figure(figsize=(12,18))

train['log1p_price'] = np.log1p(train['price'])
train['log1p_price_x_quantity'] = np.log1p(train['price'] * train['quantity'])

plt.subplot(311)
gb_train_count = train.groupby(['school_state']).count().reset_index()
gb_train_count.sort_values(by='project_is_approved', inplace=True, ascending=False)
p = sns.barplot(x=gb_train_count['school_state'], y=gb_train_count['project_is_approved'])
p = plt.title('Projects barplot of Projects\' school state',fontsize=12)
p = plt.ylabel('Total number of projects proposals', fontsize=12)
p = plt.xlabel('School State', fontsize=12)

plt.subplot(312)
order = train.groupby(['school_state'])['project_is_approved'].count().sort_values()[::-1].index
p = sns.boxplot(x='school_state', y='log1p_price', data=train, order=order)
p = plt.title('Price boxplots of Projects\' school state (ONE item)',fontsize=12)
p = plt.ylabel('Log1p price', fontsize=12)
p = plt.xlabel('School State', fontsize=12)

plt.subplot(313)
order = train.groupby(['school_state'])['project_is_approved'].count().sort_values()[::-1].index
p1 = sns.boxplot(x='school_state', y='log1p_price_x_quantity', data=train, order=order)
p1 = plt.title('Price boxplots of Projects\' school state (ALL items)',fontsize=12)
p1 = plt.ylabel('Log1p price', fontsize=12)
p1 = plt.xlabel('School State', fontsize=12)

plt.show()
plt.figure(figsize=(12,20))

train['log1p_price'] = np.log1p(train['price'])
train['log1p_price_x_quantity'] = np.log1p(train['price'] * train['quantity'])

plt.subplot(121)
order = train.groupby(['school_state'])['log1p_price'].median().fillna(0).sort_values()[::-1].index
p = sns.boxplot(y='school_state', x='log1p_price', hue='project_is_approved', data=train, order=order, orient='h')
p = plt.title('Price boxplots of Projects\' school state (ONE item)',fontsize=12)
p = plt.xlabel('Log1p price', fontsize=12)
p = plt.ylabel('School State', fontsize=12)

plt.subplot(122)
order = train.groupby(['school_state'])['log1p_price_x_quantity'].median().fillna(0).sort_values()[::-1].index
p1 = sns.boxplot(y='school_state', x='log1p_price_x_quantity', hue='project_is_approved', data=train, order=order, orient='h')
p1 = plt.title('Price boxplots of Projects\' school state (ALL items)',fontsize=12)
p1 = plt.xlabel('Log1p price', fontsize=12)
p1 = plt.ylabel('School State', fontsize=12)

plt.show()
plt.figure(figsize=(12,12))

train['log1p_price_x_quantity'] = np.log1p(train['price'] * train['quantity'])

plt.subplot(211)
order = train.groupby(['teacher_prefix'])['log1p_price_x_quantity'].median().fillna(0).sort_values()[::-1].index
p1 = sns.boxplot(x='teacher_prefix', y='log1p_price_x_quantity', data=train, order=order)
p1 = plt.title('Price boxplots of Projects price by teacher\'s prefix (ALL items)',fontsize=12)
p1 = plt.ylabel('Log1p price', fontsize=12)
p1 = plt.xlabel('School State', fontsize=12)

plt.subplot(212)
order = train.groupby(['teacher_prefix'])['log1p_price_x_quantity'].median().fillna(0).sort_values()[::-1].index
p1 = sns.boxplot(x='teacher_prefix', y='log1p_price_x_quantity', hue='project_is_approved', data=train, order=order)
p1 = plt.title('Price boxplots of Projects price by teacher\'s prefix (ALL items)',fontsize=12)
p1 = plt.ylabel('Log1p price', fontsize=12)
p1 = plt.xlabel('School State', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.subplot(121)
p = plt.pie(train[train['project_is_approved']==1].groupby(['teacher_prefix'])['project_is_approved'].value_counts(), explode = (0,0,0,0,0.1),
        labels=[p for p in train['teacher_prefix'].unique() if type(p)==str], autopct=absolute_value)
p = plt.title('Approved projects percentual by teacher\'s prefix over training data', fontsize=12)
p = plt.legend(fontsize=12)

plt.subplot(122)
p1 = plt.pie(train[train['project_is_approved']==0].groupby(['teacher_prefix'])['project_is_approved'].value_counts(), explode = (0,0,0,0,0.1),
        labels=[p for p in train['teacher_prefix'].unique() if type(p)==str], autopct=absolute_value)
p1 = plt.title('NOT Approved projects percentual by teacher\'s prefix over training data', fontsize=12)
p1 = plt.legend(fontsize=12)
p1 = plt.show()
plt.figure(figsize=(12,15))

train['datetime_no_seconds'] = train['project_submitted_datetime'].dt.date
ts_train = train[['datetime_no_seconds','project_is_approved']].groupby('datetime_no_seconds').count()
test['datetime_no_seconds'] = test['project_submitted_datetime'].dt.date
ts_test = test[['datetime_no_seconds', 'teacher_id']].groupby('datetime_no_seconds').count()

plt.subplot(311)
p = plt.plot(ts_train, label='train', linewidth=2)
p = plt.plot(ts_test, label='test', linewidth=2)
p = plt.ylim(0,15000)
p = plt.title('Total project proposals over time',fontsize=12)
p = plt.ylabel('Total Project proposals', fontsize=12)
p = plt.xlabel('Date', fontsize=12)
p = plt.legend(fontsize=12)


plt.subplot(312)
ts_train_sum = train[train['project_is_approved']==1][['datetime_no_seconds','project_is_approved']].groupby('datetime_no_seconds').count()
p1 = plt.plot(ts_train_sum, label='Approved projects', linewidth=2)
ts_train_sum = train[train['project_is_approved']==0][['datetime_no_seconds','project_is_approved']].groupby('datetime_no_seconds').count()
p1 = plt.plot(ts_train_sum, label='NOT Approved projects', linewidth=2)
p1 = plt.ylim(0,15000)
p1 = plt.title('Approved and NOT approved projects over time',fontsize=12)
p1 = plt.ylabel('Total Project proposals', fontsize=12)
p1 = plt.xlabel('Date', fontsize=12)
p1 = plt.legend(fontsize=12)

plt.subplot(313)
ts_train_sum = train[train['project_is_approved']==1][['datetime_no_seconds','log1p_price_x_quantity']].groupby('datetime_no_seconds').mean()
p2 = plt.plot(ts_train_sum, label='Approved projects', linewidth=2)
ts_train_sum = train[train['project_is_approved']==0][['datetime_no_seconds','log1p_price_x_quantity']].groupby('datetime_no_seconds').mean()
p2 = plt.plot(ts_train_sum, label='NOT Approved projects', linewidth=2)
p2 = plt.title('Approved and NOT approved projects\' price over time',fontsize=12)
p2 = plt.ylabel('Mean log1p prices', fontsize=12)
p2 = plt.xlabel('Date', fontsize=12)
p2 = plt.legend(fontsize=12)

plt.show()

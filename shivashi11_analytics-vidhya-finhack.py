import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

data = pd.read_csv('/kaggle/input/train_fwYjLYX.csv')

test = pd.read_csv('/kaggle/input/test_1eLl9Yf.csv')
data.head()
test.head()
data['application_date'] = pd.to_datetime(data['application_date'])

data['year'] = data['application_date'].dt.year

data['month'] = data['application_date'].dt.month

data['day_of_year'] = data['application_date'].dt.dayofyear

data['weekday'] = data['application_date'].dt.weekday

data.head()
ax = sns.countplot(x = 'year', data = data)

plt.title('count of data points given in respective years')

plt.xlabel('years')

plt.ylabel('frequence')

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()/data.shape[0]*100), (p.get_x() + 0.1, p.get_height()))

plt.show()
f, axes = plt.subplots(2,2)

f.set_size_inches(20,20)



sns.barplot(y="case_count", x= "year", data=data,  orient='v' , ax=axes[0,0],)

sns.barplot(y="case_count", x= "month", data=data,  orient='v' , ax=axes[0,1])

sns.barplot(y = 'case_count', x='weekday', data = data, orient = 'v' , ax = axes[1,0])

sns.barplot(y = 'case_count', x='day_of_year', data = data, orient = 'v' , ax = axes[1,1])

plt.show()
ax = sns.barplot(x = ['1', '2'], y = [list(data['segment'].value_counts())[0], list(data['segment'].value_counts())[1]])

plt.title('% of segment')

plt.xlabel('segment')

plt.ylabel('frequency')

for p in ax.patches:

    ax.annotate('%{:.1f}'.format(p.get_height()/data.shape[0]*100), (p.get_x() + 0.1, p.get_height()))

plt.show()
ax = sns.barplot(y="case_count", x= "segment", data=data,  orient='v')

for p in ax.patches:

    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.1, p.get_height()))

plt.show()
data['branch_id'].nunique()
f = plt.figure()

f.set_size_inches(25,10)

sns.barplot(y="case_count", x= "branch_id", data=data,  orient='v' )

plt.show()
# branch_id having highest count_case 

data.groupby('branch_id')['case_count'].mean().sort_values(ascending = False).iloc[0:5]
data.groupby('branch_id')['case_count'].mean().sort_values().iloc[0:2]
# plotting trends
data_new = pd.read_csv('/kaggle/input/train_fwYjLYX.csv')

data_new.index = pd.to_datetime(data_new.application_date)



monthly = data_new.resample('M').mean()

weekly = data_new.resample('W').mean()

dayly = data_new.resample('D').mean()



fig, axs = plt.subplots(3,1) 

#hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0]) 

dayly.case_count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[0]) 

weekly.case_count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[1]) 

monthly.case_count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[2]) 

plt.show()
states = data['state'].unique()
f = plt.figure()

f.set_size_inches(25,10)

g = sns.barplot(y="case_count", x= "state", data=data,  orient='v' , palette= sns.color_palette("muted"), )



plt.title('Count_Case state wise')

plt.xlabel('state')

plt.ylabel('caount_case')



plt.show()
branch_state = []

for state in states:

    branch_state.append(data[data['state'] == state]['branch_id'].nunique())
f = plt.figure()

f.set_size_inches(25,10)

g = sns.barplot(y= branch_state, x= states)



plt.title('branch_id state wise')

plt.xlabel('state')

plt.ylabel(' Number of branch_id')



plt.show()
data['zone'].unique()
for state in states:

    print(state, "occurs in ", data[data['state'] == state]['zone'].nunique(), "zones")
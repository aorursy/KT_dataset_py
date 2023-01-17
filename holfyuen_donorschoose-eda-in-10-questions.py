# Load packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

%matplotlib inline
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
donors = pd.read_csv('../input/Donors.csv')
donations = pd.read_csv('../input/Donations.csv', parse_dates=['Donation Received Date'])
# teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False, parse_dates=['Teacher First Project Posted Date'])
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
# print (resources.shape, schools.shape, donors.shape, donations.shape, teachers.shape, projects.shape)
print (resources.shape, schools.shape, donors.shape, donations.shape, projects.shape)
resources.sample(2)
schools.sample(2)
donors.sample(2)
donors['Donor ID'].nunique()
donations.sample(2)
donations['Donor ID'].nunique()
# Top 3 donations
donations.sort_values(['Donation Amount'], ascending=False).head(3).loc[:,['Project ID','Donation Amount','Donation Received Date']]
print ('Earliest donation: ', donations['Donation Received Date'].min())
print ('Latest donation: ', donations['Donation Received Date'].max())
projects.sample(2)
projects['Project Type'].value_counts()
projects['Project Current Status'].value_counts()
projects['Project Grade Level Category'].value_counts()
projects['Project Resource Category'].value_counts(ascending=True).plot(kind='barh', color='lawngreen', alpha=0.8, figsize=(8,5));
projects['Project Subject Category Tree'].value_counts().reset_index().head(10)
by_donor = donations.groupby(['Donor ID'])['Donation Amount'].agg([np.sum, np.mean, np.size]).reset_index()

plt.figure(figsize=(10,15))
plt.subplot(311)
by_donor.sort_values(['sum'], ascending=False).head(50)['sum'].plot(kind='bar', color='c')
plt.title('Top 50 Donors by Total Donation Amount', fontsize=14)
plt.xticks([])
plt.subplot(312)
by_donor.sort_values(['mean'], ascending=False).head(50)['mean'].plot(kind='bar', color='darkcyan')
plt.title('Top 50 Donors by Mean Donation Amount', fontsize=14)
plt.xticks([])
plt.subplot(313)
by_donor.sort_values(['size'], ascending=False).head(50)['size'].plot(kind='bar', color='cyan')
plt.title('Top 50 Donors by No. of Donations', fontsize=14)
plt.xticks([])
plt.show()
by_donor = by_donor.sort_values(['sum'], ascending=False)
print ('Largest donor by amount: ', by_donor.iloc[0,1].round(0))
print ('2nd largest donor: ', by_donor.iloc[1,1].round(0))
print ('3nd largest donor: ', by_donor.iloc[2,1].round(0))
print ('No. of donors donate above $1,000,000: ', sum(by_donor['sum']>1000000))
print ('No. of donors donate above $100,000: ', sum(by_donor['sum']>100000))
print ('No. of donors donate above $10,000: ', sum(by_donor['sum']>10000))
print ('No. of donors donate above $1,000: ', sum(by_donor['sum']>1000))
print ('No. of donors donate above $100: ', sum(by_donor['sum']>100))
print ('No. of donors donate $100 or below: ', sum(by_donor['sum']<=100))
print ('No. of donors donate $10 or below: ', sum(by_donor['sum']<=10))
by_donor = by_donor.sort_values(['mean'],ascending=False)
print ('Largest donor by mean amount: ', by_donor.iloc[0,2].round(0))
print ('2nd largest donor: ', by_donor.iloc[1,2].round(0))
print ('3nd largest donor: ', by_donor.iloc[2,2].round(0))
print ('No. of donors donate above $10,000 on average: ', sum(by_donor['mean']>10000))
print ('No. of donors donate above $1,000 on average: ', sum(by_donor['mean']>1000))
print ('No. of donors donate above $500 on average: ', sum(by_donor['mean']>500))
print ('No. of donors donate above $100 on average: ', sum(by_donor['mean']>100))
print ('No. of donors donate above $50 on average: ', sum(by_donor['mean']>50))
print ('No. of donors donate above $10 on average: ', sum(by_donor['mean']>10))
print ('No. of donors donate $10 or below: ', sum(by_donor['mean']<=10))
by_donor = by_donor.sort_values(['size'], ascending=False)
print ('Largest donor by no. of donations made: ', by_donor.iloc[0,3])
print ('2nd largest donor: ', by_donor.iloc[1,3])
print ('3nd largest donor: ', by_donor.iloc[2,3])
print ('No. of donors who donate more than 10000 times: ', sum(by_donor['size']>10000))
print ('No. of donors who donate more than 1000 times: ', sum(by_donor['size']>1000))
print ('No. of donors who donate more than 100 times: ', sum(by_donor['size']>100))
print ('No. of donors who donate more than 50 times: ', sum(by_donor['size']>50))
print ('No. of donors who donate more than 10 times: ', sum(by_donor['size']>10))
print ('No. of donors who donate 10 times or below: ', sum(by_donor['size']<=10))
print ('No. of donors who donate only once: ', sum(by_donor['size']==1))
def top_donor(function, position):
    top_donor = donations[donations['Donor ID']==by_donor.sort_values([function], ascending=False).iloc[position-1,0]]
    return top_donor

top_donor('sum', 1)['Donation Amount'].describe()
top_donor('mean', 1)['Donation Amount'].describe()
top_donor('size', 1)['Donation Amount'].describe()
top1 = pd.merge(top_donor('sum', 1), projects, on='Project ID', how='left')
plt.figure(figsize=(10,12))
plt.subplot(211)
top1.groupby('Project Title').sum()['Donation Amount'].sort_values(ascending=False).head(15).plot(kind='barh', color='orange')
plt.title('Top 15 Projects by Largest Donor by Total Amount')
plt.xlabel('Donation Amount')

plt.subplot(212)
_ = top1.groupby('Project Resource Category').sum()['Donation Amount'].sort_values(ascending=False)
_.plot(kind='barh', color='purple')
plt.title('Resources Breakdown by Largest Donor by Total Amount')
plt.xlabel('Donation Amount')

plt.show()
top1 = pd.merge(top_donor('mean', 1), projects, on='Project ID', how='left')
top1[['Project ID','Donation Amount','Project Title', 'Project Need Statement', 'Project Subject Category Tree','Project Resource Category','Project Cost']]
top1 = pd.merge(top_donor('size', 1), projects, on='Project ID', how='left')
plt.figure(figsize=(10,12))

plt.subplot(211)
top1.groupby('Project Title').sum()['Donation Amount'].sort_values(ascending=False).head(15).plot(kind='barh', color='orange')
plt.title('Top 15 Projects by the Most Frequent Donor')
plt.xlabel('Donation Amount')

plt.subplot(212)
_ = top1.groupby('Project Resource Category').sum()['Donation Amount'].sort_values(ascending=False)
_.plot(kind='barh', color='purple')
plt.title('Resources Breakdown by the Most Frequent Donor')
plt.xlabel('Donation Amount')
plt.show()
ts = donations.loc[:,['Donation Received Date', 'Donation Amount']]
ts.set_index('Donation Received Date', inplace=True)
ts = ts[(ts.index>='2013-01-01') & (ts.index<'2018-01-01') ]
ts['month'] = ts.index.month
ts.groupby('month').sum()['Donation Amount'].plot(kind='barh', color='dodgerblue', figsize=(8,5))
plt.xlabel('Donation Amount')
plt.show()
donors = pd.merge(donors, by_donor, on='Donor ID', how='outer')

by_state = donors.groupby(['Donor State'])['sum','size'].agg(np.sum).reset_index()
by_city = donors.groupby(['Donor City'])['sum','size'].agg(np.sum).reset_index()
plt.figure(figsize=(16,6))
plt.subplot(121)
by_state = by_state.sort_values('sum', ascending=False)
sns.barplot(x=by_state['sum'].head(20)/1000000, y=by_state['Donor State'].head(20), palette = 'summer')
plt.title('Top 20 States by Donation Amount', fontsize=13)
plt.xlabel('Amount M')
plt.subplot(122)
by_city = by_city.sort_values('sum', ascending=False)
sns.barplot(x=by_city['sum'].head(20)/1000000, y=by_city['Donor City'].head(20), palette = 'spring')
plt.title('Top 20 Cities by Donation Amount', fontsize=13)
plt.xlabel('Amount M')
plt.show()
plt.figure(figsize=(16,6))
plt.subplot(121)
by_state = by_state.sort_values('size', ascending=False)
sns.barplot(x=by_state['size'].head(20), y=by_state['Donor State'].head(20), palette = 'summer')
plt.title('Top 20 States by Donation Counts', fontsize=13)
plt.xlabel('Count')
plt.subplot(122)
by_city = by_city.sort_values('size', ascending=False)
sns.barplot(x=by_city['size'].head(20), y=by_city['Donor City'].head(20), palette = 'spring')
plt.title('Top 20 Cities by Donation Counts', fontsize=13)
plt.xlabel('Count')
plt.show()
donations = pd.merge(donations, donors.loc[:,['Donor ID','Donor City', 'Donor State']], 
                on='Donor ID', how='left')
donations = donations.merge(projects.loc[:,['Project ID','School ID','Project Grade Level Category','Project Resource Category']], on='Project ID', how='left')
donations = donations.merge(schools.loc[:,['School ID','School State','School City', 'School Name']], on='School ID', how='left')
donations['same_state'] = (donations['Donor State'] == donations['School State'])*1
donations['same_city'] = (donations['Donor City'] == donations['School City'])*1
pd.pivot_table(donations, values='Donation Amount', index='same_state', aggfunc='sum').div(sum(donations['Donation Amount']))*100
_ = donations.loc[donations['same_state']==1,:]
pd.pivot_table(_, values='Donation Amount', index='same_city', aggfunc='sum').div(sum(_['Donation Amount']))*100
pd.options.display.float_format = '{:.2f}'.format

donations = donations.merge(by_donor.loc[:,['Donor ID','size']], on='Donor ID', how='left')
donations = donations.merge(projects.loc[:,['Project ID','Project Cost']], on='Project ID', how='left')
donations['once'] = (donations['size']==1)*1
_ = pd.concat([donations.loc[donations['once']==0,'Donation Amount'].describe(),
           donations.loc[donations['once']==1,'Donation Amount'].describe()],axis=1)
_.columns=['multiple','single']
_
pd.options.display.float_format = '{:.4f}'.format

donations['per_cost']=donations['Donation Amount']/donations['Project Cost']
_ = pd.concat([donations.loc[donations['once']==0,'per_cost'].describe(),
           donations.loc[donations['once']==1,'per_cost'].describe()],axis=1)
_.columns=['multiple','single']
_
pd.options.display.float_format = None
_ = donations.groupby(['Project ID','Donor ID']).size().reset_index()
top10p = _.groupby(['Project ID']).size().sort_values(ascending=False).head(10).to_frame()
top10p_d = projects.merge(top10p, how='right', left_on='Project ID', right_index=True)
top10p_d = top10p_d.rename(columns={0:'donor_no'})
top10p_d = top10p_d.sort_values('donor_no',ascending=False)
top10p_d[['Project Title','Project Resource Category','Project Need Statement','donor_no','Project Cost']]
avg_success = (len(projects[projects['Project Current Status']=='Fully Funded']) - len(projects[projects['Project Current Status']=='Expired'])) / len(projects)

_ = pd.crosstab(projects['Project Resource Category'], projects['Project Current Status'], normalize='index')
_['success'] = _['Fully Funded']-_['Expired']
_['success'].sort_values(ascending=False).plot(kind='barh', color='darkorange', figsize=(8,6))
plt.axvline(x=avg_success)
plt.title('% Fully Funded minus % Expired', fontsize=14)
plt.show()
resources = resources.dropna(axis=0, subset=['Resource Item Name', 'Resource Quantity'], how='any')
resources = resources.merge(projects.loc[:,['Project ID','Project Current Status']], how='left', on='Project ID')
resources['Resource Total Price'] = resources['Resource Quantity'] * resources['Resource Unit Price']
avg_success2 = (resources.groupby('Project Current Status').size()['Fully Funded'] - resources.groupby('Project Current Status').size()['Expired'])/len(resources)
_ = resources[resources['Resource Unit Price']>500]
high_success = (_.groupby('Project Current Status').size()['Fully Funded'] - _.groupby('Project Current Status').size()['Expired'])/len(_)
print ('Success ratio of all resources: {:.4f}'.format(avg_success2))
print ('Success ratio of high unit price resources: {:.4f}'.format(high_success))
d = {}
def create_keys(brand):
    d[brand] = {'No. of Requests': len(_), 
                'Quantity Requested': sum(_['Resource Quantity']), 
                'Total Amount Requested': sum(_['Resource Total Price']), 
                'Total Amount Funded': sum(_.loc[_['Project Current Status']=='Fully Funded','Resource Total Price'])}

cond_p = resources['Resource Unit Price']>100
# Apple
cond1 = resources['Resource Item Name'].str.contains('ipad')
cond2 = resources['Resource Item Name'].str.contains('macbook')
_ = resources[(cond1 | cond2) & cond_p]
create_keys('Apple')
# Google
cond1 = resources['Resource Item Name'].str.contains('chromebook')
_ = resources[cond1 & cond_p]
create_keys('Google')
# Microsoft
cond1 = resources['Resource Item Name'].str.contains('microsoft')
cond2 = resources['Resource Item Name'].str.contains('surface')
_ = resources[cond1 & cond2 & cond_p]
create_keys('Microsoft')
pd.options.display.float_format = '{:.1f}'.format
df = pd.DataFrame.from_dict(d, orient='index')
df['Funded %'] = df['Total Amount Funded'] / df['Total Amount Requested'] * 100
df
def top_x_donor(function, position):
    x_donor_list = by_donor.sort_values([function], ascending=False).iloc[:position,0].tolist()
    top_x_donor = donations[donations['Donor ID'].isin(by_donor.sort_values([function], ascending=False).iloc[:position,0].tolist())]
    return top_x_donor
def donor_table(columns):
    top_10 = top_x_donor('sum', 10)
    df = pd.DataFrame(columns=['Donor ID'])
    df['Donor ID'] = top_10['Donor ID'].unique()
    for column in columns:
        dp = top_10.groupby(['Donor ID', column])['Donation Amount'].sum().reset_index()
        max_col = dp.loc[dp.groupby(["Donor ID"])["Donation Amount"].idxmax(), ['Donor ID', column]]
        df = df.merge(max_col, how='left', on='Donor ID')
        df.rename(columns={column: 'Favorite '+column}, inplace=True)
    return df
df = donor_table(['Project Grade Level Category', 'Project Resource Category', 'School State', 'School City', 'School Name'])
df
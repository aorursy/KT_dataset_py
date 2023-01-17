import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
kiva_loans=pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
print(kiva_loans.shape)
kiva_loans.sample(5)
kiva_mpi_region_locations=pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
print(kiva_mpi_region_locations.shape)
kiva_mpi_region_locations.sample(10)
kiva_mpi_region_locations.isnull().sum()      
mpi = kiva_mpi_region_locations[['country','region', 'MPI']]
mpi = mpi.dropna()
print(mpi.shape)
print(mpi.sample(5))
loans = kiva_loans[['country','region','loan_amount','activity','sector','borrower_genders', 'repayment_interval']]
print(loans.shape)
loans = loans.dropna(subset = ['country','region'])
print(loans.shape)
loans.sample(5)
d= pd.merge(loans, mpi, how='left')
d.count()
d = d.dropna(subset=['MPI'])
d.sample(5)
d1=d.groupby(['country','region','MPI'])['loan_amount'].mean().reset_index(name='Mean Loan Amount')
plt.figure(figsize=(8,6))
sns.regplot(x = d1.MPI, y = d1['Mean Loan Amount'], fit_reg=True)
plt.title("MPI vs. Mean Loan Amount")
plt.show()
d2=d.groupby(['country','region','MPI'])['loan_amount'].sum().reset_index(name='Sum of Loan Amounts')
plt.figure(figsize=(8,6))
sns.regplot(x = d2.MPI, y = d2['Sum of Loan Amounts'], fit_reg=True)
plt.title("MPI vs. Sum of Loan Amounts")
plt.show()
d3=d.groupby(['country','region','MPI'])['loan_amount'].count().reset_index(name='Number of Loans')
plt.figure(figsize=(8,6))
sns.regplot(x = d3.MPI, y = d3['Number of Loans'], fit_reg=False)
plt.title("MPI vs. Number of Loans")
plt.show()
d3.loc[d3['Number of Loans'] == d3['Number of Loans'].max()]
d3['Location']=d3['country'] + ", " + d3['region']
d3 = d3.set_index(['Location'])
d3 = d3.sort_values("Number of Loans", ascending=False)
d3 = d3.loc[d3['Number of Loans']>20]
fig = plt.figure(figsize=(17, 7)) 
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

d3[["MPI"]].plot(kind='bar', color='blue', ax=ax2, width=.4, position=0)
d3[["Number of Loans"]].plot(kind='bar', color='green', ax=ax, width=.4, position=1)

ax.set_ylabel('Number of Loans')
ax2.set_ylabel('MPI')
plt.show()
df_gender = pd.DataFrame(d.borrower_genders.str.split(',').tolist())
#dd = pd.concat([df_gender[0], df_gender[1], df_gender[2], df_gender[3], df_gender[4], df_gender[5]], ignore_index=True).dropna()
d['gender'] = df_gender[0]
# This needs to be done better. Now I'm only taking the first column.
d.groupby(['gender'])['MPI'].mean()
fig = plt.figure(figsize=(17, 7)) 
ax = fig.add_subplot(111) 
sns.distplot(d.loc[d['gender']=='female'].MPI, label='female', ax=ax, color='r', bins=50, kde=True)
sns.distplot(d.loc[d['gender']=='male'].MPI, label='male', ax=ax, color='b', bins=50, kde=True)
plt.legend()
plt.show()
df_sector = d.groupby(['sector'])['MPI'].mean().sort_values(ascending=True).reset_index(name="Average MPI")
plt.figure(figsize=(10,5))
g = sns.barplot(x='sector', y="Average MPI", data=df_sector);
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Average MPI per Sector")
g.set_xlabel("Sector")
plt.show()
df_activity = d.groupby(['activity'])['MPI'].mean().sort_values(ascending=True).reset_index(name="Average MPI")
plt.figure(figsize=(25,5))
g = sns.barplot(x='activity', y="Average MPI", data=df_activity);
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Average MPI per Activity")
g.set_xlabel("Activity")
plt.show()
df_repayment_interval = d.groupby(['repayment_interval'])['MPI'].mean().sort_values(ascending=True).reset_index(name="Average MPI")
plt.figure(figsize=(16,5))

plt.subplot(121)
g = sns.barplot(x='repayment_interval', y="Average MPI", data=df_repayment_interval);
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Average MPI vs Replayment Interval")
g.set_xlabel("Repayment Interval")

plt.subplot(122)
g1 = sns.violinplot(x='repayment_interval', y='MPI', data=d)
g1.set_title("MPI Distribution by Repayment Interval")
g1.set_xlabel("")
g1.set_ylabel("MPI")

plt.show()
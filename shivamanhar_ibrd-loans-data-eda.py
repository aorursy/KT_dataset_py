import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import rcParams

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()

rcParams['figure.figsize'] = 10, 6

import os

print(os.listdir("../input"))

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)
from wand.image import Image as Img

path ='../input/ibrd-statement-of-loans-data/'

Img(filename= path+'Data_Dictionary_-_IBRD_Statement_of_Loans_and_IDA_Statement_of_Credits_and_Grants.pdf', resolution=300)
ibrd_statement = pd.read_csv(path+'ibrd-statement-of-loans-historical-data.csv')

ibrd_statement.head()
ibrd_statement.shape
#ibrd_statement = ibrd_statement.sample(frac=0.1, random_state=54)
del ibrd_statement['Loan Number']

del ibrd_statement['Country Code']

del ibrd_statement['Project ID']

del ibrd_statement['Guarantor Country Code']
ibrd_statement['Loan Status'].unique()
ibrd_statement['Board Approval Date'] = pd.to_datetime(ibrd_statement['Board Approval Date']) 
ibrd_statement = ibrd_statement.set_index(ibrd_statement['Board Approval Date'])
ibrd_statement['Loan Status'] = ibrd_statement['Loan Status'].str.strip()
ibrd_statement['Loan Status'].unique()
ibrd_statement['Loan Type'].unique()
null_values = pd.DataFrame({'columns':pd.isnull(ibrd_statement).sum().index, 'count':pd.isnull(ibrd_statement).sum().values})
print("Total {} cols have a null values".format(null_values.loc[null_values['count']>0,].shape[0]))
null_values.loc[null_values['count']>0,].sort_values(by='count', ascending=False)
ibrd_statement.loc[(ibrd_statement['Loan Status'] == 'Approved'),['Country','Original Principal Amount']].groupby(['Country'], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=False).head(15)
plt.figure(figsize=(25, 10))

sns.barplot(x='Country', y='Original Principal Amount',palette='BuGn_r', data=ibrd_statement.loc[(ibrd_statement['Loan Status'] == 'Approved'),['Country','Original Principal Amount']].groupby(['Country'], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=False).head(15))

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(25, 10))

sns.barplot(x='Country', y='Original Principal Amount', data=ibrd_statement.loc[(ibrd_statement['Loan Status'] == 'Approved'),['Country','Original Principal Amount']].groupby(['Country'], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=True).head(15))

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(25, 10))

sns.barplot(x='Project Name ', y='Original Principal Amount', palette='BuGn_r', data=ibrd_statement.loc[:,['Project Name ','Original Principal Amount']].groupby(['Project Name '], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=False).head(15))

plt.xticks(rotation=45)

plt.show()
ibrd_statement.loc[:,['Country','Original Principal Amount']].groupby([ibrd_statement.index.year]).sum().sort_values(by='Original Principal Amount', ascending=False)
group_df = ibrd_statement.loc[:,['Country','Original Principal Amount']].groupby([ibrd_statement.index.year]).sum().sort_values(by='Original Principal Amount', ascending=True).head(10)

plt.figure(figsize=(25, 10))

sns.barplot(x=group_df.index, y='Original Principal Amount',  data=group_df)

plt.xticks(rotation=45)

plt.show()

india_df = ibrd_statement.loc[ibrd_statement['Country'] =='India',]
india_df.loc[:,['Project Name ','Original Principal Amount']].groupby(['Project Name '], as_index=False).head(5)
plt.figure(figsize=(25, 10))

sns.barplot(x='Project Name ', y='Original Principal Amount', palette='BuGn_r', data=india_df.loc[:,['Project Name ','Original Principal Amount']].groupby(['Project Name '], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=False).head(15))

plt.xticks(rotation=45)

plt.show()
ibrd_statement.loc[:,['Loan Status','Original Principal Amount']].groupby(['Loan Status'], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=True).head(15)
plt.figure(figsize=(25, 10))

sns.barplot(x='Loan Status', y='Original Principal Amount',palette='BuGn_r', data=ibrd_statement.loc[:,['Loan Status','Original Principal Amount']].groupby(['Loan Status'], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=False).head(15))

plt.xticks(rotation=45)

plt.show()
ibrd_statement.head()
ibrd_statement.head(3)
ibrd_statement['First Repayment Date'] = pd.to_datetime(ibrd_statement['First Repayment Date']) 

ibrd_statement['Last Repayment Date'] = pd.to_datetime(ibrd_statement['Last Repayment Date']) 

ibrd_statement['Agreement Signing Date'] = pd.to_datetime(ibrd_statement['Agreement Signing Date']) 

ibrd_statement['Board Approval Date'] = pd.to_datetime(ibrd_statement['Board Approval Date']) 

ibrd_statement['Effective Date (Most Recent)'] = pd.to_datetime(ibrd_statement['Effective Date (Most Recent)']) 

ibrd_statement['Closed Date (Most Recent)'] = pd.to_datetime(ibrd_statement['Closed Date (Most Recent)'])

ibrd_statement['End of Period'] = pd.to_datetime(ibrd_statement['End of Period'])
ibrd_statement.head(3)
ibrd_statement['Interest Rate'].fillna(ibrd_statement['Interest Rate'].min(), inplace=True)
plt.figure(figsize=(25, 6))

sns.distplot(ibrd_statement['Original Principal Amount'] ,color='r')

sns.distplot(ibrd_statement['Disbursed Amount'], color='g')

sns.distplot(ibrd_statement['Repaid to IBRD'], color='b')

plt.show()
plt.plot(ibrd_statement['Loans Held'].unique())

plt.show()
sns.boxplot(ibrd_statement['Interest Rate'])

plt.show()
ibrd_statement.shape
g = sns.FacetGrid(ibrd_statement,  hue="Loan Status", col="Loan Type", margin_titles=True,  col_wrap=4, palette='Paired')

g=g.map(plt.scatter, "Original Principal Amount", "Disbursed Amount",edgecolor="w").add_legend();

plt.show()
ibrd_statement.loc[:,['Loan Type','Original Principal Amount']].groupby(['Loan Type'], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=False).head(10)
sns.barplot(x='Loan Type', y='Original Principal Amount', data=ibrd_statement.loc[:,['Loan Type','Original Principal Amount']].groupby(['Loan Type'], as_index=False).sum().sort_values(by='Original Principal Amount', ascending=False).head(10))

plt.xticks(rotation=45)

plt.show()
india_df = ibrd_statement.loc[(ibrd_statement['Country'] == "India"),]
india_df.head(2)
india_df.loc[:,'Original Principal Amount'].groupby(india_df.index.year, as_index=True).sum().sort_values(ascending=False).head(10).values
group_df = india_df.loc[(india_df['Loan Status'] =='Approved'),'Original Principal Amount']

group_df = group_df.groupby(group_df.index.year, as_index=True).sum().sort_values(ascending=False).head(20)

sns.barplot(x=group_df.index,y=group_df.values)

plt.xticks(rotation=45)

plt.show()
group_df = india_df.loc[(india_df['Loan Status'] =='Disbursed'),'Original Principal Amount']



group_df = group_df.groupby(group_df.index.year, as_index=True).sum().sort_values(ascending=False).head(15)

sns.barplot(x=group_df.index,y=group_df.values)

plt.xticks(rotation=45)

plt.show()
#group_df = india_df.loc[(india_df['Loan Status'] =='Approved'),['Project Name ','Original Principal Amount']]

#group_df[['Project Name ','Original Principal Amount']].sort_values(by='Original Principal Amount', ascending=False).head(10)

#group_df = group_df.groupby(group_df['Project Name ','Original Principal Amount'], as_index=True).sum().sort_values(by=group_df['Original Principal Amount'], ascending=False).head(20)

#group_df.values

#sns.barplot(x=list(group_df.index), y=list(group_df.values))

#plt.xticks(rotation=45)

#plt.show()

#group_df.values
#india_df.loc[(india_df.index.year == 2009),].head()
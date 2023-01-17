# Importing required libraries.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import plotly.express as px
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

import warnings
warnings.filterwarnings('ignore')
#Loading the data into the data frame
df=pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv', parse_dates=['date'], index_col='date')
#Adding the parse_dates=['date'] argument will make the date column to be parsed as a date field.
malawi=df[df['country']=='Malawi']
malawi.head()
malawi.shape
malawi.info()
#list of all sectors in Malawi
pd.unique(malawi['sector'])
malawi.describe()
malawi.columns
# Total number of rows and columns
malawi.shape
# Rows containing duplicate data
malawi_dups = malawi[malawi.duplicated()]
malawi_dups.shape
#count the number of rows for duplicates
malawi_dups.count() 
# Finding the null values per feature
malawi.isnull().any()
#Number of missing data points in each column
malawi.isnull().sum()
# Dropping the missing values
malawi_ds = malawi.dropna() 
malawi_ds.count()
# After dropping the values
malawi_ds.isnull().sum()
# Finding the relations between the variables.
plt.figure(figsize=(20,10))
MW= malawi_ds.corr()
sns.heatmap(MW,cmap='seismic',annot=True)
MW
fig,mw = plt.subplots(figsize=(10,6))
mw.scatter(malawi_ds['funded_amount'], malawi_ds['lender_count'])
mw.set_xlabel('funded_amount')
mw.set_ylabel('lender_count')
plt.show()
Blantyre = malawi_ds[malawi_ds['region'] == 'Blantyre']
Blantyre.head(5)
#picked out a specific region in malawi from its large dataset
#Loan Amount by Activity in Blantyre Region
px.scatter(Blantyre, x='activity', y='loan_amount', title='Activity vs. Loan Amount',
          labels={'activity':'Activities in Malawi','loan_amount':'Allocated Loan Amount to Activity'}, color='sector',
          size='loan_amount')
#Highest loan amount was allocated to fruits & vegetables in Blantyre region of malawi
#Lender Count by Region
plt.figure(figsize=(20,10))
           
plt.title('Lender Count by Region')

plt.xticks(rotation=90)

sns.barplot(x='region',y='lender_count',data=malawi_ds, ci=None);
pie = malawi_ds.groupby('repayment_interval').size()

# Make the plot with pandas
plt.figure()
pie.plot(kind='pie', subplots=True, figsize=(10, 10))
plt.title("Pie Chart of repayment interval")
plt.ylabel("")
# Draw Plot
def plot_df(malawi_ds, x, y, title="", xlabel='funded_time', ylabel='loan_amount', dpi=60):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(malawi_ds,x=malawi_ds.index, y=malawi_ds.loan_amount, title='Loan amount disbursed quartely in Malawi during the funding period (2014 to 2017).')

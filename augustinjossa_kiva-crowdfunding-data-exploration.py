#Import libraries
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

#Import Kiva datasets
loans = pd.read_csv('../input/kiva_loans.csv')

#Crating views
studied_loans = loans.where((loans['funded_amount']>0)).dropna()
small_loans = loans.where((loans['funded_amount']>0)&(loans['funded_amount']<1000)).dropna()
medium_loans = loans.where((loans['funded_amount']>=1000)&(loans['funded_amount']<10000)).dropna()
big_loans = loans.where((loans['funded_amount']>10000)).dropna()
unfunded_loans = loans[loans['funded_amount'].isin([0])]

#Visualization functions
def top10(value,selection,dataset, operation=1, number=10):
    df = dataset[:][[selection,value]]
    if operation == 2: 
        df = df.groupby(selection).mean()
    else:
        df = df.groupby(selection).sum()
    df = df.sort_values(by=value, ascending=False)
    df_x = df.iloc[1:number+1][value]
    df_y = df.iloc[1:number+1].index.values
    plot=sns.barplot(x=df_x, y=df_y) 
    plt.xlabel('')
    if operation == 2: 
        plot.set(title="Mean of " +value + " by " + selection)
    else:
        plot.set(title=value + " by " + selection)
    return
# Understanding loans on Kiva platform
print(len(small_loans)/len(studied_loans))
print(sum(small_loans['funded_amount'])/sum(studied_loans['funded_amount']))
print(len(medium_loans)/len(studied_loans))
print(sum(medium_loans['funded_amount'])/sum(studied_loans['funded_amount']))
print(len(big_loans)/len(studied_loans))
print(sum(big_loans['funded_amount'])/sum(studied_loans['funded_amount']))
small_loans.describe()
plt.figure(figsize=(20,4))
sns.distplot(small_loans[:]['funded_amount'])
plt.show()

plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
top10('id','sector',small_loans, operation=1, number=10)
plt.subplot(1,2,2)
top10('funded_amount','sector',small_loans, operation=1, number=10)
plt.show()

plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
top10('id','country',small_loans, operation=1, number=10)
plt.subplot(1,2,2)
top10('funded_amount','country',small_loans, operation=1, number=10)
plt.show()

small_loans.sort_values(by='funded_amount', ascending=False)[['funded_amount','use']].head(20)
medium_loans.describe()
plt.figure(figsize=(20,4))
sns.distplot(medium_loans[:]['funded_amount'])
plt.show()

plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
top10('id','sector',medium_loans, operation=1, number=10)
plt.subplot(1,2,2)
top10('funded_amount','sector',medium_loans, operation=1, number=10)
plt.show()

plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
top10('id','country',medium_loans, operation=1, number=10)
plt.subplot(1,2,2)
top10('funded_amount','country',medium_loans, operation=1, number=10)
plt.show()

medium_loans.sort_values(by='funded_amount', ascending=False)[['funded_amount','use']].head(20)
print(len(big_loans[big_loans['funded_amount'].isin([100000])]))
print(len(big_loans[big_loans['funded_amount'].isin([50000])]))
plt.figure(figsize=(20,4))
sns.distplot(big_loans[:]['funded_amount'])
plt.show()

plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
top10('id','sector',big_loans, operation=1, number=10)
plt.subplot(1,2,2)
top10('funded_amount','sector',big_loans, operation=1, number=10)
plt.show()

plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
top10('id','country',big_loans, operation=1, number=10)
plt.subplot(1,2,2)
top10('funded_amount','country',big_loans, operation=1, number=10)
plt.show()

big_loans.sort_values(by='funded_amount', ascending=False)[['funded_amount','use', 'country']].head(20)
sum(studied_loans['funded_amount'])/sum(studied_loans['lender_count'])
plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
sns.distplot(big_loans[:]['loan_amount'])

plt.subplot(1,2,2)
sns.distplot(big_loans[:]['lender_count'])
plt.show()

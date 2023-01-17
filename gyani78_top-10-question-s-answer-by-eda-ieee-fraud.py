# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_transaction= pd.read_csv("../input/train_transaction.csv")

train_identity= pd.read_csv("../input/train_identity.csv")
print("No of records on the train_transaction is:",train_transaction.shape[0],"No of columns is",train_transaction.shape[1])

print("No of records on the train_identity is:",train_identity.shape[0],"No of columns is",train_identity.shape[1])
train_transaction.head()
plt.figure(figsize=(20,20))

plt.subplot(3,2,1)

g=sns.countplot(x='isFraud',data=train_transaction)

g.set_xlabel("isFraud ",fontsize =15)

g.set_ylabel("Count Percentage of isFraud variable",fontsize =15)

g.set_title(" % wise target distribution NO Fraud 0 | Fraud 1:",fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height+100,'{:1.2f}%'.format(height/len(train_transaction)*100),ha="center",fontsize = 12)

plt.subplot(3,2,2)

sum_amt= train_transaction.groupby('isFraud')['TransactionAmt'].sum()

total_amt = train_transaction.groupby('isFraud')['TransactionAmt'].sum().sum()

sum_amt  = sum_amt.reset_index()

g=sns.barplot(x='isFraud',y='TransactionAmt',data=sum_amt)

g.set_xlabel("isFraud ?",fontsize =15)

g.set_ylabel("Sum of transactionAmt ",fontsize =15)

g.set_title(" % transactionAmt wise target distribution NO Fraud 0 | Fraud 1:",fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height+100,'{:1.2f}%'.format(height/total_amt*100),ha="center",fontsize = 12)    

plt.show()
train_transaction['TransactionAmt'].quantile([0.100,0.250,0.500,0.750,0.970,0.980,0.990,1.0])
plt.figure(figsize=(16,12))

plt.subplot(2,2,1)

g=sns.distplot(train_transaction[train_transaction['TransactionAmt'] <= 1000]['TransactionAmt'])

g.set_xlabel("values ",fontsize =15)

g.set_ylabel("Probability",fontsize =15)

g.set_title("Transaction amount distribution for less than 1000 value",fontsize=15)



plt.subplot(2,2,2)

g1=sns.distplot(np.log(train_transaction['TransactionAmt']))

g1.set_xlabel("values ",fontsize =15)

g1.set_ylabel("Probability",fontsize =15)

g1.set_title("Transaction amount log distribution",fontsize=15)

plt.figure(figsize=(16,12))

plt.subplot(2,1,2)

g2 = plt.scatter(range(train_transaction[train_transaction['isFraud']==1].shape[0]),np.sort(train_transaction[train_transaction['isFraud']==1]['TransactionAmt'].values),c='red')

g2 = plt.scatter(range(train_transaction[train_transaction['isFraud']==0].shape[0]),np.sort(train_transaction[train_transaction['isFraud']==0]['TransactionAmt'].values),c='blue')

plt.xlabel("Range of all transaction ",fontsize =15)

plt.ylabel("Transaction Amount values",fontsize =15)

plt.title("Transaction vs Transaction amount distribution",fontsize=15)

plt.figure(figsize=(16,12))

plt.subplot(2,2,1)

g3 = plt.scatter(range(train_transaction[train_transaction['isFraud']==1].shape[0]),np.sort(train_transaction[train_transaction['isFraud']==1]['TransactionAmt'].values),c='red')

plt.xlabel("Range of transaction with fraud ",fontsize =15)

plt.ylabel("Transaction Amount values",fontsize =15)

plt.title("Transaction vs Transaction amount distribution for fraud",fontsize=15)

plt.subplot(2,2,2)

g4 = plt.scatter(range(train_transaction[train_transaction['isFraud']==0].shape[0]),np.sort(train_transaction[train_transaction['isFraud']==0]['TransactionAmt'].values),c='blue')

plt.xlabel("Range of transaction with no fraud ",fontsize =15)

plt.ylabel("Transaction Amount values",fontsize =15)

plt.title("Transaction vs Transaction amount distribution for no fraud",fontsize=15)

plt.show()


plt.figure(figsize=(16,12))

plt.suptitle("ProductCD Distribution", fontsize=22)

plt.subplot(2,2,1)

g=sns.countplot(x='ProductCD',hue='isFraud',data=train_transaction)

g.set_xlabel("ProductCD ",fontsize =15)

g.set_ylabel("Count Percentage of isFraud variable",fontsize =15)

g.set_title(" ProductCd distribution of count percent wise groupby isFraud",fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height+100,'{:1.2f}%'.format(height/len(train_transaction)*100),ha="center",fontsize = 12)



plt.subplot(2,2,2)

g1=sns.countplot(x='ProductCD',hue='isFraud',data=train_transaction)

gt = g1.twinx()

gt = sns.pointplot(x='ProductCD', y='isFraud', data=train_transaction, color='black', order=['W', 'H',"C", "S", "R"], legend=False)

gt.set_ylabel("% of Fraud Transactions", fontsize=16)



g.set_xlabel("ProductCD ",fontsize =15)

g.set_ylabel("Count Percentage of isFraud variable",fontsize =15)

g.set_title(" ProductCd distribution of count percent wise groupby isFraud",fontsize=18)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height+100,'{:1.2f}%'.format(height/len(train_transaction)*100),ha="center",fontsize = 12)



plt.figure(figsize=(16,12))

plt.subplot(212)

g=sns.boxplot(x='ProductCD',y='TransactionAmt',hue='isFraud',data=train_transaction[train_transaction['TransactionAmt'] <=1304])

plt.figure(figsize=(16,22))

plt.suptitle("C1 Distributions", fontsize=22)

plt.subplot(411)

g= sns.distplot(train_transaction[train_transaction['isFraud']==0]['card1'],label='No Fraud')

g= sns.distplot(train_transaction[train_transaction['isFraud']==1]['card1'],label='Fraud')

g.legend()

g.set_xlabel("Range of card1 value:",fontsize =15)

g.set_ylabel("Probability",fontsize =15)

g.set_title("card1 distribution on the basis target wise",fontsize=18)



plt.subplot(412)

g= sns.distplot(train_transaction[train_transaction['isFraud']==0]['card2'].dropna(),label='No Fraud')

g= sns.distplot(train_transaction[train_transaction['isFraud']==1]['card2'].dropna(),label='Fraud')

g.legend()

g.set_xlabel("Range of card2 value:",fontsize =15)

g.set_ylabel("Probability",fontsize =15)

g.set_title("card2 distribution on the basis target wise",fontsize=18)



plt.subplot(413)

g= sns.distplot(train_transaction[train_transaction['isFraud']==0]['card3'].dropna(),label='No Fraud')

g= sns.distplot(train_transaction[train_transaction['isFraud']==1]['card3'].dropna(),label='Fraud')

g.legend()

g.set_xlabel("Range of card3 value:",fontsize =15)

g.set_ylabel("Probability",fontsize =15)

g.set_title("card3 distribution on the basis target wise",fontsize=18)



plt.subplot(414)

g= sns.distplot(train_transaction[train_transaction['isFraud']==0]['card5'].dropna(),label='No Fraud')

g= sns.distplot(train_transaction[train_transaction['isFraud']==1]['card5'].dropna(),label='Fraud')

g.legend()

g.set_xlabel("Range of card5 value:",fontsize =15)

g.set_ylabel("Probability",fontsize =15)

g.set_title("card5 distribution on the basis target wise",fontsize=18)



plt.show()
plt.figure(figsize=(16,12))

plt.suptitle("ProductCD Distribution", fontsize=22)

plt.subplot(2,1,1)

g=sns.countplot(x='card4',hue='isFraud',data=train_transaction)

g.set_xlabel("card4 ",fontsize =15)

g.set_ylabel("Count Percentage of isFraud variable",fontsize =15)

g.set_title(" Card4 distribution of count percent wise groupby isFraud",fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height+100,'{:1.2f}%'.format(height/len(train_transaction)*100),ha="center",fontsize = 12)

    

plt.subplot(2,1,2)

g=sns.countplot(x='card6',hue='isFraud',data=train_transaction)

g.set_xlabel("card6 ",fontsize =15)

g.set_ylabel("Count Percentage of isFraud variable",fontsize =15)

g.set_title(" Card6 distribution of count percent wise groupby isFraud",fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,height+100,'{:1.2f}%'.format(height/len(train_transaction)*100),ha="center",fontsize = 12)



pd.set_option('display.max_columns', 500)

train_transaction.head()
print("Card Features Quantiles:")

train_transaction[['addr1','addr2']].quantile([0.500,0.750,0.950,0.990])
train_transaction.loc[train_transaction.addr1.isin(train_transaction.addr1.value_counts()[train_transaction.addr1.value_counts() <= 5000 ].index), 'addr1'] = "Others"

train_transaction.loc[train_transaction.addr2.isin(train_transaction.addr2.value_counts()[train_transaction.addr2.value_counts() <= 50 ].index), 'addr2'] = "Others"
def ploting_cnt_amt(df, col, lim=2000):

    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    

    plt.figure(figsize=(16,14))    

    plt.suptitle(f'{col} Distributions ', fontsize=24)

    

    plt.subplot(211)

    g = sns.countplot( x=col,  data=df, order=list(tmp[col].values))

    gt = g.twinx()

    gt = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),

                       color='black', legend=False, )

    gt.set_ylim(0,tmp['Fraud'].max()*1.1)

    gt.set_ylabel("%Fraud Transactions", fontsize=16)

    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)

    g.set_xlabel(f"{col} Category Names", fontsize=16)

    g.set_ylabel("Count", fontsize=17)

    g.set_xticklabels(g.get_xticklabels(),rotation=45)

    sizes = []

    for p in g.patches:

        height = p.get_height()

        sizes.append(height)

        g.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/len(df)*100),

                ha="center",fontsize=12) 

        

    g.set_ylim(0,max(sizes)*1.15)

    

    #########################################################################

    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() \

                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')

    perc_amt = perc_amt.reset_index()

    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()

    perc_amt = perc_amt.fillna(0)

    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()

    plt.subplot(212)

    g1 = sns.barplot(x=col, y='TransactionAmt', 

                       data=amt, 

                       order=list(tmp[col].values))

    g1t = g1.twinx()

    g1t = sns.pointplot(x=col, y='Fraud', data=perc_amt, 

                        order=list(tmp[col].values),

                       color='black', legend=False, )

    g1t.set_ylim(0,perc_amt['Fraud'].max()*1.1)

    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)

    g.set_xticklabels(g.get_xticklabels(),rotation=45)

    g1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)

    g1.set_xlabel(f"{col} Category Names", fontsize=16)

    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)

    g1.set_xticklabels(g.get_xticklabels(),rotation=45)    

    

    for p in g1.patches:

        height = p.get_height()

        g1.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total_amt*100),

                ha="center",fontsize=12) 

        

    plt.subplots_adjust(hspace=.4, top = 0.9)

    plt.show()



ploting_cnt_amt(train_transaction,'addr1')
ploting_cnt_amt(train_transaction,'addr2')
train_transaction['P_emaildomain'] = train_transaction['P_emaildomain'].apply(lambda x: 'Google' if x=='gmail.com' or x=='gmail' else x)

train_transaction['P_emaildomain'] = train_transaction['P_emaildomain'].apply(lambda x: 'yahoo' if x in ['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk','yahoo.co.jp', 'yahoo.de', 'yahoo.fr','yahoo.es'] else x)

train_transaction['P_emaildomain'] = train_transaction['P_emaildomain'].apply(lambda x: 'microsoft' if x in ['hotmail.com','outlook.com','msn.com', 'live.com.mx','hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr'] else x)
train_transaction.loc[train_transaction.P_emaildomain.isin(train_transaction['P_emaildomain'].value_counts()[train_transaction['P_emaildomain'].value_counts()<500].index),'P_emaildomain'] = 'Others'

ploting_cnt_amt(train_transaction,'P_emaildomain')
train_transaction['R_emaildomain'] = train_transaction['R_emaildomain'].apply(lambda x: 'Google' if x=='gmail.com' or x=='gmail' else x)

train_transaction['R_emaildomain'] = train_transaction['R_emaildomain'].apply(lambda x: 'yahoo' if x in ['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk','yahoo.co.jp', 'yahoo.de', 'yahoo.fr','yahoo.es'] else x)

train_transaction['R_emaildomain'] = train_transaction['R_emaildomain'].apply(lambda x: 'microsoft' if x in ['hotmail.com','outlook.com','msn.com', 'live.com.mx','hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr'] else x)

train_transaction.loc[train_transaction.R_emaildomain.isin(train_transaction['R_emaildomain'].value_counts()[train_transaction['R_emaildomain'].value_counts()<=300].index),'R_emaildomain'] = 'Others'

train_transaction['R_emaildomain'].value_counts()
ploting_cnt_amt(train_transaction,'R_emaildomain')
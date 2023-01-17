# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns 

import matplotlib.pyplot as plt



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

# test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

trainid = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

# testid = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 200)

# pd.set_option('display.height', 1000)
print(train.shape)
train.head()
# test.head()
train.info()
train.describe()
miss=train.isnull().describe()

miss
(miss.loc[:,(miss!=False).all()].loc['freq',:]/590540)[:20]
train.select_dtypes(exclude=['int','float']).describe()
train['P_emaildomain'].value_counts()
cat_columns=['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain',

 'R_emaildomain','M1','M2','M3','M4','M5','M6','M7','M8','M9']
train[cat_columns].select_dtypes(exclude=['object']).nunique()
train[cat_columns].select_dtypes(exclude=['object']).describe()
#train[cat_columns].select_dtypes(exclude=['int','float']).describe()
plt.figure(figsize=(6, 8))

splot=sns.barplot(x='isFraud', y="isFraud", data=train, estimator=lambda x: len(x) / len(train) * 100)

for p in splot.patches:

    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.ylabel('Percentage')
#Dropping Vesta features for now as we have little info about them

cols = [c for c in train.columns if c.lower()[:1] != 'v']

train_rem=train[cols].copy()
cat_columns=['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain',

 'R_emaildomain','M1','M2','M3','M4','M5','M6','M7','M8','M9']
train_rem[cat_columns]=train_rem[cat_columns].astype('category')
a=train_rem.select_dtypes(exclude=['category']).drop(['TransactionID','TransactionDT'],axis=1)
for cols in a.columns:

    data_mean, data_std = np.mean(a[cols]), np.std(a[cols])

    A=a[cols].quantile(0.9)  #Remove outliers and then plot

#     print(cols, A)

#     A=data_mean+data_std * 2

    r=a.loc[a[cols] <= A]

    sns.violinplot(x="isFraud", y=cols, data=r, palette="muted")

#     c_max = a[cols].max()

#     plt.ylim(0,c_max*0.1)

    plt.show()
for cols in a.columns:

    sns.kdeplot(a[(a[cols] <= a[cols].quantile(0.9))&(a['isFraud']==1)][cols],color='red',  shade=True, **{"label": "Fraud",'alpha':0.8})

    sns.kdeplot(a[(a[cols] <= a[cols].quantile(0.9))&(a['isFraud']==0)][cols],color='blue',shade=True,**{"label": "No Fraud",'alpha':0.4})

    plt.ylabel(cols)

    plt.show()
plt.scatter(x=a['C1'].values,y=a['D1'].values,c=a['isFraud'].values)
b=train_rem.select_dtypes(include=['category']).copy()
b.describe()
b['isFraud'] = a['isFraud'].to_numpy()
b['Ones']=np.ones(len(b))
#b.head()
columnt=['ProductCD', 'card4', 'card6', 'M1', 'M2', 'M3',

       'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
for cols in columnt:

    plt.figure(figsize=(12, 8))

    plt.subplot(121)

    cross = pd.crosstab(b[cols], b['isFraud'], normalize='index') * 100

    cross = cross.reset_index()

    cross.rename(columns={0:'No Fraud', 1:'Fraud'}, inplace=True)

    #print(cross)

    lennf=sum(b['isFraud']==0)

    lenf=sum(b['isFraud']==1)





    splot=sns.barplot(x=cols, y="Ones", data=b[b['isFraud']==0], estimator=lambda x: len(x) / lennf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')

    #sns.pointplot(x='card4', y='No Fraud', data=cross, color='black', legend=False,**{'s':20,'marker':'x'})



    plt.subplot(122)

    splot=sns.barplot(x=cols, y="Ones", data=b[b['isFraud']==1], estimator=lambda x: len(x) / lenf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')



    sns.pointplot(x=cols, y='Fraud', data=cross, color='black', legend=False, markers='o')



    plt.show()
#(b['card4'].value_counts(dropna=False)/len(b)*100).plot.bar()
#b['card4'].value_counts(dropna=False)/len(b)*100
# df2['date'] = df1['date'].to_numpy()
#b.columns 
columns=['card1', 'card2', 'card3', 'card5',

       'addr1', 'addr2']

for cols in columns:

    sns.kdeplot(b[(b['isFraud']==1)][cols],color='red',  shade=True, **{"label": "Fraud",'alpha':0.8})

    sns.kdeplot(b[(b['isFraud']==0)][cols],color='blue',shade=True,**{"label": "No Fraud",'alpha':0.4})

    plt.ylabel(cols)

    plt.show()
columnt=['P_emaildomain', 'R_emaildomain']
for cols in columnt:

    plt.figure(figsize=(20, 16))

    plt.subplot(211)

    cross = pd.crosstab(b[cols], b['isFraud'], normalize='index') * 100

    cross = cross.reset_index()

    cross.rename(columns={0:'No Fraud', 1:'Fraud'}, inplace=True)

    #print(cross)

    lennf=sum(b['isFraud']==0)

    lenf=sum(b['isFraud']==1)





    splot=sns.barplot(x=cols, y="Ones", data=b[b['isFraud']==0], estimator=lambda x: len(x) / lennf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')

    plt.xticks(rotation=90)

    #sns.pointplot(x='card4', y='No Fraud', data=cross, color='black', legend=False,**{'s':20,'marker':'x'})



    plt.subplot(212)

    splot=sns.barplot(x=cols, y="Ones", data=b[b['isFraud']==1], estimator=lambda x: len(x) / lenf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')

    plt.xticks(rotation=90)



    sns.pointplot(x=cols, y='Fraud', data=cross, color='black', legend=False, markers='o')



    plt.show()
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 

          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',

          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 

          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',

          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',

          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 

          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 

          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',

          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',

          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 

          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 

          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 

          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 

          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',

          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
for cols in columnt:

    b[cols+'1']=b[cols].map(emails)
columnt=['P_emaildomain1', 'R_emaildomain1']
ed=['google',  

'microsoft',

    'yahoo',

'other',  

'spectrum',  

'aol',  

'apple',  

'att','centurylink']
for cols in columnt:

    plt.figure(figsize=(20, 16))

    plt.subplot(211)

    cross = pd.crosstab(b[cols], b['isFraud'], normalize='index') * 100

    cross = cross.reset_index()

    cross.rename(columns={0:'No Fraud', 1:'Fraud'}, inplace=True)

    #print(cross)

    lennf=sum(b['isFraud']==0)

    lenf=sum(b['isFraud']==1)



    df=b[b['isFraud']==0].set_index(cols).loc[ed]

    splot=sns.barplot(x=df.index, y="Ones", data=df, estimator=lambda x: len(x) / lennf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')

    #positions = (0,1, 2, 3,4,5,6,7,8)

    #labels = ed

    #plt.xticks(positions, labels)

    plt.xticks(rotation=90)

    #sns.pointplot(x='card4', y='No Fraud', data=cross, color='black', legend=False,**{'s':20,'marker':'x'})



    plt.subplot(212)

    df=b[b['isFraud']==1].set_index(cols).loc[ed]

    splot=sns.barplot(x=df.index, y="Ones", data=df, estimator=lambda x: len(x) / lenf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')

    #positions = (0,1, 2, 3,4,5,6,7,8)

    #labels = ed

    #plt.xticks(positions, labels)

    plt.xticks(rotation=90)

    

    plt.figure(figsize=(10, 8))

    sns.pointplot(x=cols, y='Fraud', data=cross, color='black', legend=False, markers='o')



    plt.show()
import datetime



START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

train["Date"] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))



# tra['_Weekdays'] = df_trans['Date'].dt.dayofweek

train['Hour of day'] = train['Date'].dt.hour

train['Day of Month'] = train['Date'].dt.day
b=train[['isFraud', 'Date','Hour of day','Day of Month']]
columnt=['Hour of day','Day of Month']
b['Ones']=np.ones(len(b))
for cols in columnt:

    plt.figure(figsize=(20, 16))

    plt.subplot(211)

    cross = pd.crosstab(b[cols], b['isFraud'], normalize='index') * 100

    cross = cross.reset_index()

    cross.rename(columns={0:'No Fraud', 1:'Fraud'}, inplace=True)

    #print(cross)

    lennf=sum(b['isFraud']==0)

    lenf=sum(b['isFraud']==1)





    splot=sns.barplot(x=cols, y="Ones", data=b[b['isFraud']==0], estimator=lambda x: len(x) / lennf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')

    plt.xticks(rotation=90)

    #sns.pointplot(x='card4', y='No Fraud', data=cross, color='black', legend=False,**{'s':20,'marker':'x'})



    plt.subplot(212)

    splot=sns.barplot(x=cols, y="Ones", data=b[b['isFraud']==1], estimator=lambda x: len(x) / lenf * 100)

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    plt.ylabel('Percentage')

    plt.xticks(rotation=90)



    sns.pointplot(x=cols, y='Fraud', data=cross, color='black', legend=False, markers='o')



    plt.show()
#data.where(data.apply(lambda x: x.map(x.value_counts()))>=2, "other")
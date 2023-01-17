import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt # for plotting

import seaborn as sns 

sns.set(style = 'whitegrid',palette = 'Set3',context = 'talk')

        

        

import warnings

warnings.filterwarnings('ignore')

#load data

df = pd.read_csv('../input/application_train.csv')

#POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')

#bureau_balance = pd.read_csv('../input/bureau_balance.csv')

#previous_application = pd.read_csv('../input/previous_application.csv')

#installments_payments = pd.read_csv('../input/installments_payments.csv')

#credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')

#bureau = pd.read_csv('../input/bureau.csv')

#application_test = pd.read_csv('../input/application_test.csv')
# check column dtypes

print(df.dtypes.value_counts())



# Number of unique classes in each object column

df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# they are category columns
# manually number the cat-columns (just in case )

# of course we could also use (le) Lable Encoding or one-hot encoding 

# we will do this in next kernel : feature engineering



df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].replace({'Cash loans':0,

                                                         'Revolving loans':1})

df['CODE_GENDER'] = df['CODE_GENDER'].replace({'M':1,'F':0})

df['CODE_GENDER'] = df[df['CODE_GENDER'] != 'XNA']  # just 4 rows, and we remove them

df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace({'Y':1,'N':0})

df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace({'Y':1,'N':0})                                                         

# missing value check

def mis_check(x):

    total = x.isnull().sum().sort_values(ascending = False)

    percentage = (x.isnull().sum()/x.isnull().count()*100).sort_values(ascending = False)

    tb = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

    

    display(tb.head(20)) # just show the top 20 rows

mis_check(df)
tmp = df['NAME_CONTRACT_TYPE'].value_counts()

tmp1 = df['TARGET'].value_counts()

plt.subplots(1,2,figsize = (12,6))

plt.subplots_adjust(left = 0.1,wspace = 0.4)

colors = ['lightcoral', 'lightskyblue']



plt.subplot(121)

tmp.plot.pie(autopct='%1.1f%%', shadow=True, startangle=45,explode = (0.2,0),colors = colors)

plt.title('Gender Distribution')

plt.subplot(122)

tmp1.plot.pie(autopct='%1.1f%%', shadow=True, startangle=45,explode = (0.2,0),colors = colors)

plt.title('Default Distribution')





cols = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN']

plt.figure(2 , figsize = ( 24, 6))

n = 0

for c in cols:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(wspace =0.4)

    sns.countplot(df[c] )

    plt.title('Countplot of {}'.format(c))

cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']

plt.figure(4 , figsize = ( 24, 12))

n = 0

for c in cols:

    n += 1

    plt.subplot(2 , 2 , n)

    plt.subplots_adjust(wspace =0.2, hspace =0.4,)

    sns.boxenplot(df[c] )

    plt.title('boxplot of {}'.format(c))
cols =['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']

plt.figure(4 , figsize = ( 24, 16))

n = 0

for c in cols:

    n += 1

    plt.subplot(3 , 2 , n)

    plt.subplots_adjust(wspace =0.5, hspace =0.4,)

    sns.countplot(y = df[c], )

    plt.title('countplot of {}'.format(c))
corr = df.corr()['TARGET'].sort_values()

# Display correlations

print('Top 10 strong Positive Correlations:\n', corr.tail(10))

print('\nTop 10 strong Negative Correlations:\n', corr.head(10))
df['AGE'] = round(abs(df['DAYS_BIRTH'])/365,0)



plt.subplots(2,1,figsize = (20,12))

plt.subplot(211)

df['AGE'].plot(kind= 'hist',bins = 10, figsize = (12,6),color = 'lightblue')

plt.title('Distribution of Age')

plt.subplot(212)

sns.kdeplot(df.loc[df['TARGET'] == 0, 'AGE'], label = 'Not-Default')

sns.kdeplot(df.loc[df['TARGET'] == 1, 'AGE'], label = 'Default')

df['EMPLOYED_YEAR'] = round(abs(df['DAYS_EMPLOYED'])/365,1)

df['EMPLOYED_YEAR'].corr(df['TARGET'])
plt.subplots(2,1,figsize = (20,12))

plt.subplot(211)

df['EMPLOYED_YEAR'].plot(kind= 'hist',bins = 10, figsize = (12,6),color = 'lightcoral')

plt.subplot(212)

sns.kdeplot(df.loc[df['TARGET'] == 0, 'EMPLOYED_YEAR'], label = 'Not-Default')

sns.kdeplot(df.loc[df['TARGET'] == 1, 'EMPLOYED_YEAR'], label = 'Default')
df0 = df[df['EMPLOYED_YEAR'] < 50]



plt.subplots(2,1,figsize = (20,12))

plt.subplot(211)

df0['EMPLOYED_YEAR'].plot(kind= 'hist',bins = 10, figsize = (12,6),color = 'lightcoral')

plt.title('Distribution of working year')

plt.subplot(212)

sns.kdeplot(df0.loc[df0['TARGET'] == 0, 'EMPLOYED_YEAR'], label = 'Not-Default')

sns.kdeplot(df0.loc[df0['TARGET'] == 1, 'EMPLOYED_YEAR'], label = 'Default')

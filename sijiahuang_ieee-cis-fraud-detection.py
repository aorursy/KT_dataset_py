# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Merge identity and transaction for both training and test dataset

train_identity = pd.read_csv('../input/train_identity.csv')

train_transaction = pd.read_csv('../input/train_transaction.csv')

test_identity = pd.read_csv('../input/test_identity.csv')

test_transaction = pd.read_csv('../input/test_transaction.csv')



train = pd.merge( train_transaction,train_identity, on = 'TransactionID', how = 'left')

test = pd.merge(  test_transaction,test_identity, on = 'TransactionID', how = 'left')
#delete unused datasets

del train_identity, train_transaction, test_identity, test_transaction
print("Training data's shape: ", train.shape)

print("Test data's shape: ", test.shape)
train.describe()
train.info()
#Take a look at Missing data: number, percentage for each column

def info_missing( df ):

    missing_num = df.isna().sum()

    missing_perc = missing_num/df.shape[0]*100

    col_dtype = pd.Series([df[col].dtype for col in df.columns], index = df.columns)

    res = pd.concat( [missing_num, missing_perc, col_dtype], axis = 1, keys = ['Missing#', 'Missing%', 'dtype'])

    return res.T



pd.set_option('max_columns', 500)

info_missing(train)
#Take a look at number of unique value for each categorical column: 

def info_unique( df ):

    cols_cat = [c for c in df.columns if df[c].dtype == 'object']

    n_unique = [ df[c].nunique() for c in cols_cat]

    res = pd.Series(n_unique, index = cols_cat).sort_values(ascending = True).transpose()

    return res

info_unique(train)
#Unbalanced data: need to deal with later

print( "Fraud % is  {:1f} % in training set".format(train.isFraud.sum()/train.shape[0]*100)) 
#Divide and conqur 

#id_01 - id_38: id's data

cols = train.columns.tolist()

id_01_index = cols.index('id_01')

id_38_index = cols.index('id_38')

train_id = train.iloc[:,id_01_index: id_38_index+1]



#id's numerical data

train_id_num = train_id.select_dtypes( exclude = ['object'])

train_id_num.describe()
train_id_num.hist(bins = 20, figsize = (20, 15))
#correlation of id's numerical data

import seaborn as sns

train_id_num_corr = train_id_num.corr()

sns.heatmap( train_id_num_corr, cmap="YlGnBu")
#id's categorical data

train_id_cat = train_id.select_dtypes( include = ['object'])
train_id_cat.head()
train_id_cat.describe()
import matplotlib.pyplot as plt

def bar_plot( feature, n, norm = True):

    train_fraud = train[train.isFraud ==1][feature].value_counts(dropna = False, normalize= norm)

    train_notfraud = train[train.isFraud ==0][feature].value_counts(dropna = False, normalize= norm)

    f, ax = plt.subplots(1,2, figsize = (18,4))

    sns.barplot( y = train_fraud.index[:n], x = train_fraud[:n], ax = ax[0]).set_title( feature + ' - Fraud')

    sns.barplot( y = train_notfraud.index[:n], x = train_notfraud[:n], ax = ax[1]).set_title(feature +' - Not Fraud')
#id_30 seems to be system, with Windows 10 being most common, 75 unique values

bar_plot('id_30',10) #Fraud seems to have more systems other than windows
#id_31 seems to be explorer, with chrome being most common, 130 unique values

bar_plot('id_31',10)
#id_33? 260 unique values

bar_plot('id_33',10)
#Device type

bar_plot('DeviceType', 10)
#Device info, somehow overlaps with id_30

bar_plot('DeviceInfo', 10)
#TransactionDT

print( len(train.TransactionDT.unique()))

train.TransactionDT.describe() #TransactionDT seems different for every record
train.TransactionDT.plot( kind = 'hist', figsize = (15, 5), label = 'train', bins = 50, title = 'TransactionDT')

test.TransactionDT.plot( kind = 'hist', label = 'test', bins = 50)

plt.legend()

plt.show()
#TransactionAmt: 

train.TransactionAmt.describe()
#TransactionAmt: some outliers

#train.TransactionAmt[train.TransactionAmt<3000].hist(bins = 50, figsize = (8,6))

ax = train.plot( x = 'TransactionDT', y = 'TransactionAmt', kind = 'scatter', alpha = 0.01, label = 'Train', 

           title = 'Transaction Amount', figsize = (15,5), ylim = (0,5000))

test.plot( x = 'TransactionDT', y = 'TransactionAmt', kind = 'scatter', alpha = 0.01, label = 'Test', ax = ax)

#fraud

train.loc[ train.isFraud ==1].plot( x = 'TransactionDT', y = 'TransactionAmt', kind = 'scatter', alpha = 0.01,color = 'yellow', ax = ax)

plt.legend()

plt.show()
#ProductCD: 5 unique values, not sure what this is

bar_plot( 'ProductCD',5)
#card1 - card6: categorical data

index_card1 = cols.index('card1')

train_card = train.iloc[:, index_card1:index_card1+6]
train_card_num = train_card.select_dtypes( exclude = ['object'])

train_card_num.describe()
train_card_num.hist(bins = 50, figsize = (15,10))
#card4

bar_plot('card4',5)
#card6: fraud has higher share from credit card 

bar_plot('card6',5)
#dist1, dist2

train.loc[:,['dist1','dist2']].describe()
train.loc[:,['dist1','dist2']].hist(bins = 20, figsize= (10,5))
#P_emaildomain and R_emaildomain

train.loc[:,['P_emaildomain', 'R_emaildomain']].nunique()
#P_domain

bar_plot('P_emaildomain', 10)
#R_domain

bar_plot('R_emaildomain', 10)
#C1-C14: all numerical data - float64

index_C1 = cols.index('C1')

train_C = train.iloc[:,index_C1:index_C1+14]
train_C.describe()
train_C.hist(bins=50, figsize=(15,15))
#D1 - D15

index_D1 = cols.index('D1')

train_D = train.iloc[:, index_D1:index_D1+15]

train_D.describe()
train_D.hist(bins = 50, figsize = (15,15))
#M1 - M9: objects; boolean except for M4

index_M1 = cols.index('M1')

train_M = train.iloc[:,index_M1: index_M1+9]

train_M.describe()
#V1-V339: all float values

index_V1 = cols.index('V1')

train_V = pd.DataFrame(train.iloc[:, index_V1:])
train_V.describe()
#train_V.info()

train_V.dtypes.unique()
#Prepare data for machine learning algorithms

#1. delete columns with 1) unique value 2) 90% are different value and 3) more than 90% is missing value

unique_col_train = [col for col in train.columns if train[col].nunique() <= 1]

unique_col_test = [col for col in test.columns if test[col].nunique()<= 1]

various_col_train = [col for col in train.columns if train[col].value_counts(dropna = False, normalize = True).values[0] > 0.9]

various_col_test = [col for col in test.columns if test[col].value_counts(dropna = False, normalize = True).values[0] > 0.9]

missing_col_train = [col for col in train.columns if train[col].isnull().sum()/train.shape[0] > 0.9]

missing_col_test =  [col for col in test.columns if test[col].isnull().sum()/test.shape[0] >0.9 ]
for i in [unique_col_train, unique_col_test, various_col_train, various_col_test, missing_col_train, missing_col_test]:

    print( i, len(i))
col_to_drop = list(set(unique_col_test +unique_col_train + various_col_test +various_col_train + missing_col_test +missing_col_train))

col_to_drop.remove('isFraud')

len(col_to_drop)
train.drop( col_to_drop, axis = 1, inplace = True)

test.drop(col_to_drop, axis = 1, inplace= True)
#2. deal with email domain

#https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin'] = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
#3. deal with categorical data

from sklearn.preprocessing import LabelEncoder

cat_cols = [col for col in train.columns if train[col].dtype is ['object']]

for col in cat_cols:

    le = LabelEncoder()

    le.fit( list(train[col].astype(str).values) + list( test[col].astype(str).values))

    train[col] = le.transform( list(train[col].astype(str).values))

    test[col] = le.transform( list(test[col].astype(str).values))
#drop transaction 
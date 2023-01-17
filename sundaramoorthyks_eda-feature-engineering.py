



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load the train and test data

train_identity=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

train_transaction=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

test_identity=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

test_transaction=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
# reduce your memory by conversion

# convert it to the low memory to fit the RAM

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
#merge both the transaction and identity by left

train=pd.merge(train_transaction,train_identity,how="left",on="TransactionID")

test=pd.merge(test_transaction,test_identity,how="left",on="TransactionID")
#now we should reduce the memory to free the RAM or else we cant fit the model

train=reduce_mem_usage(train)

test=reduce_mem_usage(test)
# delete the 4 variables in order to reduce the memory issue

del train_identity

del test_identity

del train_transaction

del test_transaction
#Try to explore  all the columns in your dataframe

train.head(5)
# category columns

category_column=['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',

            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',

            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',

            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

print("no of categorical column:",len(category_column))
#let us try to check for NAs in each columns

print("Train data")

train.isna().sum()

print("Test data")

test.isna().sum()
#EDA

#If there is more than 90% NA's we can remove that no need of that column it was not going to affect that much on the final column

more_than_90_NA_or_same_value_train=[]

more_than_90_NA_or_same_value_test=[]

many_na_train=[]

many_na_test=[]

for col in train.columns:

    if train[col].isna().sum()/train.shape[0] >=0.90:

        many_na_train.append(col) # full of NAs in train

for col in test.columns:

    if test[col].isna().sum()/test.shape[0]>=0.90:

        many_na_test.append(col) # full of NAs in test

for col in train.columns:

  #  print(col,train[col].value_counts(dropna=False,normalize=True).values[0])

    if train[col].value_counts(dropna=False,normalize=True).values[0] >= 0.90:

      #  print("More than 90% is NA's or same value so we can delete that columns")

        more_than_90_NA_or_same_value_train.append(col) # more unique values in train

for col in test.columns:

    if test[col].value_counts(dropna=False,normalize=True).values[0]>=0.90:

        more_than_90_NA_or_same_value_test.append(col) #more unique values in test
# store the columns to be dropped separately in train and test

cols_drop_at_train=list(set(more_than_90_NA_or_same_value_train+many_na_train))

cols_drop_at_test=list(set(more_than_90_NA_or_same_value_test+many_na_test))

print("Columns to be dropped in train",len(cols_drop_at_train))

print("Columns to be dropped in test",len(cols_drop_at_test))

print("columns are @ train:",cols_drop_at_train)

print("columns are @ test:", cols_drop_at_train)
total_drop_cols=list(set(cols_drop_at_train+cols_drop_at_test))

print("Total no of columns to be deleted to increase your model performance",len(total_drop_cols))

print("They are:",total_drop_cols)
# remove the isFraud

total_drop_cols.remove('isFraud')

print("You can check thta column is removed:",total_drop_cols)
for col in total_drop_cols:

    if col not in train.columns:

        print("missing drop column in train",col)

    if col not in test.columns:

        print("Missing drop columns in test",col)
n=0

print("len",len(total_drop_cols))

for col in train.columns:

    if col in total_drop_cols:

        n+=1

print(n)
#columns after dropping unwanted columns

print("Total no of columns we have now",len(train.columns))

# after dropping columns we need to explore the data distrubtions

# try to plot the distribution to check it

# reference:https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

# we can start to analyze from the  TransactionDT

# timedelta from a given reference datetime (not an actual timestamp)

sns.distplot(train['TransactionDT'], hist=True, kde=True,bins=40) # its shows histogram along with the density plot

sns.distplot(test['TransactionDT'],hist=True,kde=True,bins=40)

plt.title('Density Plot of  TransactionDT  in training data')

plt.xlabel(' TransactionDT')

plt.ylabel('Counts')

   

# so totally the given test is future of the train so be carefull with the split
#TransactionAMT: transaction payment amount in USD

sns.distplot(train['TransactionAmt'], hist=True, kde=True,bins=1) # its shows histogram along with the density plot

plt.title('Density Plot of  TransactionAMT in training data')

plt.xlabel(' TransactionAMT')

plt.ylabel('Counts')

# most of the amount is less than 5000
#ProductCD --  product code, the product for each transaction

#sns.catplot(x="index", y="ProductCD", hue="index", kind="bar", data=feature_count); 

# in the above plot we can arrange it

sns.countplot(x="ProductCD", data=train) # shows the count in each class
# how we can start to analyze more about the cards

#card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.

# categorical variable -ALL the cards

for col in ['card1','card2','card3','card4','card5','card6']:

    print("Feature count of " + str(col))

    feature_count=(train[col].value_counts())

    print(feature_count.head(2)) # its so big so i have plotted only 2 

# card1- some numerical values

#card2- some amount with float values

#card3- same as card 2

#card4 - card type- [visa,mastercard,american express,discover]

#card5- same as card2

#card6- type of the card-[debit,credit,charge card,debit or credit]
# card 6-type of card

sns.countplot(x=train['card6'])
#card4-types of card

sns.countplot(train['card4'])
# how we can start to check how many transaction amount are in each types of card

#for the sum it shows infifnite

print(train.groupby('card4')['TransactionAmt'].mean()) # the discover has highest mean over all
# we can now check for card6

print(train.groupby('card6')['TransactionAmt'].mean()) # the discover has highest mean over all

#fig, ax = plt.subplots()

#train.groupby('card6').plot(x='card6', y='TransactionAmt',ax=ax)

# credit card has more value
a4_dims = (20, 20)

fig, axs = plt.subplots(4,1, figsize=a4_dims, squeeze=False)

card_list=['card1','card2','card3','card5']

co=0



for r in range(0,4):

    for c in range(0, 1): 

        feature_count=train[card_list[co]].value_counts().reset_index()

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        ax=sns.barplot(x='index',y=card_list[co],data=feature_count,errwidth=12,capsize=10,ax=axs[r][c])

        ax.set_xlabel(card_list[co])

        ax.set_ylabel('Number of Occurrences')

        co+=1





print("This column has high number of categoricals to print so it will be very slow")
# addr: address addr1, addr2- categorical variable

a4_dims = (20, 20)

fig, axs = plt.subplots(2,1, figsize=a4_dims, squeeze=False)

addr_list=['addr1','addr2']

co=0



for r in range(0,2):

    for c in range(0, 1): 

        feature_count=train[addr_list[co]].value_counts().reset_index()

       # feature_count= feature_count.sort_values([addr_list[co]])

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        

        ax=sns.barplot(x='index',y=addr_list[co],data=feature_count,errwidth=12,capsize=10,ax=axs[r][c])

        ax.set_xlabel(addr_list[co])

        ax.set_ylabel('Number of Occurrences')

        co+=1
# dist: distance is numerical we can analyze it later

#P_ and (R__) emaildomain: purchaser and recipient email domain its categorical

#print(train['P_emaildomain'].value_counts())

a4_dims = (20, 20)

fig, axs = plt.subplots(2,1, figsize=a4_dims, squeeze=False)

addr_list=['P_emaildomain','R_emaildomain']

co=0



for r in range(0,2):

    for c in range(0, 1): 

        feature_count=train[addr_list[co]].value_counts().reset_index()

       # feature_count= feature_count.sort_values([addr_list[co]])

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        

        ax=sns.barplot(x='index',y=addr_list[co],data=feature_count,errwidth=12,capsize=10,ax=axs[r][c])

        ax.set_xlabel(addr_list[co])

        ax.set_ylabel('Number of Occurrences')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        co+=1



#in both this case the domain name like .eu and .in are different but they should be same try to preprocess it 

# we can split the given one by '.' and take the fisrt part for the correct mail names
m_list=['M1','M2','M3','M4','M5','M6','M7','M8','M9']

# check they are caetgorical or not

for col in m_list:

    print("For the " + str(col))

    print(train[col].value_counts())

# expect M4 all other are T/F
# M1 - M9 categorical variable that need to analyze

# the values are match, such as names on card and address, etc.

a4_dims = (20, 20)

fig, axs = plt.subplots(9,1, figsize=a4_dims, squeeze=False)

co=0

m_list=['M1','M2','M3','M4','M5','M6','M7','M8','M9']

for r in range(0,9):

    for c in range(0, 1): 

        feature_count=train[m_list[co]].value_counts().reset_index()

       # feature_count= feature_count.sort_values([addr_list[co]])

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        

        ax=sns.barplot(x='index',y=m_list[co],data=feature_count,errwidth=12,capsize=100,ax=axs[r][c])

       

        ax.set_xlabel(m_list[co])

        ax.set_ylabel('Number of Occurrences')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        co+=1

plt.subplots_adjust(hspace = 0.2)

plt.tight_layout()

# I think no need of preprocess for this M1-M9 set of features.
#DeviceType

#train['DeviceType'].value_counts() # only two types # we can check where we get more isFraud 

sns.countplot(x='DeviceType',hue='isFraud',data=train)

# we have more isFraud  in desktop
#Deviceinfo

feature_count=train['DeviceInfo'].value_counts().reset_index()

feature_count.sort_values('DeviceInfo')

feature_count=feature_count.iloc[:40,]

#print(feature_count)

ax=sns.barplot(x="index", y="DeviceInfo", data=feature_count,errwidth=12,capsize=100)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# we have more isFraud  in desktop
#id12 - id38 we need to analyze this part its categorical variable

id_list=[]

for i in range(12,39):

    id_list.append('id_'+str(i))

print(id_list)
#iterate the id_list and visualize it

a4_dims = (20, 20)

fig, axs = plt.subplots(5,1, figsize=a4_dims, squeeze=False)



co=0



for r in range(0,5):

    for c in range(0, 1): 

        feature_count=train[id_list[co]].value_counts().reset_index()

       # feature_count= feature_count.sort_values([addr_list[co]])

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        

        ax=sns.barplot(x='index',y=id_list[co],data=feature_count,errwidth=12,capsize=100,ax=axs[r][c])

       

        ax.set_xlabel(id_list[co])

        ax.set_ylabel('Number of Occurrences')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        co+=1



plt.tight_layout()

#id_12 - found/not_found

#id_13- many fields are there

#id_14- many fields are there

#id_15- found/new/unknown

#id_16-found/not_found
#next 5 features

a4_dims = (20, 20)

fig, axs = plt.subplots(5,1, figsize=a4_dims, squeeze=False)



co=5



for r in range(0,5):

    for c in range(0, 1): 

        feature_count=train[id_list[co]].value_counts().reset_index()

       # feature_count= feature_count.sort_values([addr_list[co]])

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        

        ax=sns.barplot(x='index',y=id_list[co],data=feature_count,errwidth=12,capsize=100,ax=axs[r][c])

       

        ax.set_xlabel(id_list[co])

        ax.set_ylabel('Number of Occurrences')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        co+=1



plt.tight_layout()
#next 5 features

a4_dims = (20, 20)

fig, axs = plt.subplots(5,1, figsize=a4_dims, squeeze=False)



co=10



for r in range(0,5):

    for c in range(0, 1): 

        feature_count=train[id_list[co]].value_counts().reset_index()

       # feature_count= feature_count.sort_values([addr_list[co]])

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        

        ax=sns.barplot(x='index',y=id_list[co],data=feature_count,errwidth=12,capsize=100,ax=axs[r][c])

       

        ax.set_xlabel(id_list[co])

        ax.set_ylabel('Number of Occurrences')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        co+=1



plt.tight_layout()

#id_23-proxy-transparent/anonymous/hidden
#next 5 features

a4_dims = (20, 20)

fig, axs = plt.subplots(5,1, figsize=a4_dims, squeeze=False)



co=15



for r in range(0,5):

    for c in range(0, 1): 

        feature_count=train[id_list[co]].value_counts().reset_index()

       # feature_count= feature_count.sort_values([addr_list[co]])

        feature_count=feature_count.iloc[:40,]

        #print(len(feature_count.iloc[:40,]))

        

        ax=sns.barplot(x='index',y=id_list[co],data=feature_count,errwidth=12,capsize=100,ax=axs[r][c])

       

        ax.set_xlabel(id_list[co])

        ax.set_ylabel('Number of Occurrences')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        co+=1



plt.tight_layout()

#id_27/_29 -found/not-found

#id_28-found/new
#next 5 features



#id_34- match-0,1,2

#id_35/_36-True/False
#we need to analyze the some id columns and numeric columns are left now

# try to create new feature with the help of the EDA

# and then try to reduce the dimension by dropping it 

# encode the category data

# missing value treatment for numeric and category columns
# you can check the some of the important parameters here

someFeature_list=['id_36','id_35','id_34','id_28','id_29','id_12','id_15','id_16']

a4_dims = (20, 20)

co=0

ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
ax=sns.countplot(x=someFeature_list[co],hue='isFraud',data=train)

ax.set_xlabel(someFeature_list[co])

ax.set_ylabel('Number of Occurrences')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

co+=1

plt.tight_layout()
print(addr_list)

# when you do one hot encoding please add both test and train both may have different one

for col in addr_list:

    train[col]=(train[col].str.split(".",expand=True)[0])

    test[col]=(test[col].str.split(".",expand=True)[0])

# now we are done with mails so we do some feature engineering
#Feature Engineering 

# first we can try to use card features

for col in ['card1','card2','card3','card4','card5','card6']:

    # we are just taking a mean for each group and diving it with the each group Transaction amount to get more information

    # and also std for each group 

    train['Transactionamt_mean_'+str(col)]=(train['TransactionAmt']/train.groupby(col)['TransactionAmt'].transform('mean'))

    train['Transactionamt_std_'+str(col)]=(train['TransactionAmt']/train.groupby(col)['TransactionAmt'].transform('std'))

    test['Transactionamt_mean_'+str(col)]=(test['TransactionAmt']/test.groupby(col)['TransactionAmt'].transform('mean'))

    test['Transactionamt_std_'+str(col)]=(test['TransactionAmt']/test.groupby(col)['TransactionAmt'].transform('std'))

#feature Engineering only for Cards alone

# we also need to check device info and device type,id_30,id_31
#Let's try to do feature Enginnering based on ProductCD because it has only 4 levels

# and also for P_emaildomain ,R_emaildomain,DeviceType

#DeviceInfo,id_15,id_23,id_30,id_31,id_34

#you guys can do according to your understanding

for col in ['ProductCD', 'P_emaildomain','R_emaildomain','DeviceType','DeviceInfo','id_15','id_23','id_30','id_31','id_34']:

    train['Transactionamt_mean_'+str(col)]=(train['TransactionAmt']/train.groupby(col)['TransactionAmt'].transform('mean'))

    train['Transactionamt_std_'+str(col)]=(train['TransactionAmt']/train.groupby(col)['TransactionAmt'].transform('std'))

    test['Transactionamt_mean_'+str(col)]=(test['TransactionAmt']/test.groupby(col)['TransactionAmt'].transform('mean'))

    test['Transactionamt_std_'+str(col)]=(test['TransactionAmt']/test.groupby(col)['TransactionAmt'].transform('std'))

# there will be lot of NAN in our columns 
# now we can preprocess our data

print("Total number of columns after Feture Engineering:",len(train.columns)) #466

# now we want to drop the unwanted columns

print(total_drop_cols)

#train=train.drop(drop)

for col in total_drop_cols:

    del train[col]

    del test[col]

print("Final number of columns after Feature Engineering:",len(train.columns)) # 384



#now we can do label encoding for categorical variable

# we can do one hot encoding but it will increase our dimension so its problem

# so we can try label encoding or any other encoding like frequency encoding .etc

# i am going to try label encoding

from sklearn  import preprocessing

for col in train.columns:

    if train[col].dtype=='object' :

      #  print("label encoding",col)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[col].values) + list(test[col].values))

        train[col] =lbl.transform(list(train[col].values))

        test[col]=lbl.transform(list(test[col].values))

def clean_inf_nan(df):

    return df.replace([np.inf, -np.inf], np.nan)   



# Cleaning infinite values to NaN

train = clean_inf_nan(train)

test = clean_inf_nan(test ) # replace all nan,inf,-inf to nan so it will be easy to replace

for i in train.columns:

    train[i].fillna(train[i].median(),inplace=True) # fill with median because mean may be affect by outliers.

#X.isna().sum().sum()

for i in test.columns:

    test[i].fillna(test[i].median(),inplace=True)



print("Number of Na's in train",train.isna().sum().sum())

print("Number of Na's in test",test.isna().sum().sum())

# now we an split the data and train our model

X = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

y = train['isFraud']

#X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

#del train

test = test[['TransactionID']]

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



#train and test split

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)
n_fold = 5

folds = TimeSeriesSplit(n_splits=n_fold)

folds = KFold(n_splits=5)
submission=pd.DataFrame()

submission['TransactionID']=test['TransactionID']

submission['isFraud'] = 0

import xgboost as xgb

from sklearn.metrics import roc_auc_score

print(submission.head(5))
import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics



from itertools import product



import altair as alt

from altair.vega import v5

def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual', n_jobs=-1):

    """

    A function to train a variety of classification models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns is None else columns

    n_splits = folds.n_splits if splits is None else n_folds

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    

    result_dict = {}

    if averaging == 'usual':

        # out-of-fold predictions on train data

        oof = np.zeros((len(X), 1))



        # averaged predictions on train data

        prediction = np.zeros((len(X_test), 1))

        

    elif averaging == 'rank':

        # out-of-fold predictions on train data

        oof = np.zeros((len(X), 1))



        # averaged predictions on train data

        prediction = np.zeros((len(X_test), 1))



    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = n_jobs)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)[:, 1]

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=Logloss)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        if averaging == 'usual':

            

            oof[valid_index] = y_pred_valid.reshape(-1, 1)

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

            

            prediction += y_pred.reshape(-1, 1)



        elif averaging == 'rank':

                                  

            oof[valid_index] = y_pred_valid.reshape(-1, 1)

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

                                  

            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

            result_dict['top_columns'] = cols

        

    return result_dict

def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True





def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()

    
params = {'num_leaves': 256,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': 13,

          'learning_rate': 0.03,

          "boosting_type": "gbdt",

          "subsample_freq": 3,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3,

          'reg_lambda': 0.3,

          'colsample_bytree': 0.9,

          #'categorical_feature': cat_cols

         }

result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,

                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual', n_jobs=-1)
submission['isFraud']=result_dict_lgb['prediction']
submission.head(5)
submission.to_csv('Submission_v2.csv')
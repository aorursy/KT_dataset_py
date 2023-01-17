%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd #pandas is short for 'panel data'

import seaborn as sns

import scipy.stats as stats



#fraud = 1, notfraud = 0

df_train_trans = pd.read_csv('../input/train_transaction.csv')

df_train_ident = pd.read_csv('../input/train_identity.csv')



df_test_trans = pd.read_csv('../input/test_transaction.csv')

df_test_ident = pd.read_csv('../input/test_identity.csv')



print("finished loading imports and methods")
#Shapes

print("training transactions shape: ",np.shape(df_train_trans))

print("training indentity shape: ",np.shape(df_train_ident))

print("testing transactions shape: ",np.shape(df_test_trans))

print("testing indentity shape: ",np.shape(df_test_ident))
def missingDataCount(df):

    missing_values_count = df.isnull().sum()

    missing_values_count.sort_values(ascending=False,inplace=True)

    print (missing_values_count[0:10])

    total_cells = np.product(df.shape)

    total_missing = missing_values_count.sum()

    print ("% of NaNs = ",(total_missing/total_cells) * 100)



missingDataCount(df_train_trans)

missingDataCount(df_train_ident)

print("--------")

missingDataCount(df_test_trans)

missingDataCount(df_test_ident)
#the large majority of the data is not fraud, will cause problems with model over fitting

y = df_train_trans['isFraud'].value_counts()

plt.bar(['Not Fraud','Fraud'],y)

plt.show()
def timeInDays(seconds):

    return (seconds / 60 / 60 / 24)

def timeInYears(days):

    return (days / 365)



min_dt_train = df_train_trans['TransactionDT'].min()

max_dt_train= df_train_trans['TransactionDT'].max()

min_dt_test = df_test_trans['TransactionDT'].min()

max_dt_test = df_test_trans['TransactionDT'].max()



train_seconds = max_dt_train - min_dt_train

test_seconds = max_dt_test - min_dt_test

total_seconds = max_dt_test - min_dt_train



train_in_days = timeInDays(train_seconds)

test_in_days = timeInDays(test_seconds)

total_in_days = timeInDays(total_seconds)



print("Days training data elapses: ",train_in_days)

print("Days test data elapses: ",test_in_days)

print("Days total data elapses: ",total_in_days)

print("Years (from total days): ",timeInYears(total_in_days))

print("years total (from days train + test)",timeInYears((train_in_days + test_in_days)))
#no reason, just for reference while i'm writing code and thinking

df_train_trans.head()
#this is a very high amount, but not classified as fraud. possibly an outlier to remove

df_train_trans.loc[df_train_trans['TransactionAmt'].idxmax()]
#plot a graph for a day's worth of transaction data

#would be more useful if there was a way to 

#statistically detect a pattern to see if these frauds occur at

#the same kind of times etc...would help our thinking on models and features

def plotDayTrans(df):

    plt.scatter(df['TransactionAmt'].index,df['TransactionAmt'])

    plt.xlabel("Transaction time (early to late)")

    plt.ylabel("Transaction Amount (usd $)")



DAY = 86400

day_start = 7

day_end = 8

plotDayTrans(df_train_trans.loc[(df_train_trans['TransactionDT']\

                                 >= (day_start*DAY)) & (df_train_trans['TransactionDT'] <= (day_end*DAY))])

#overlay the fraudulent ones

plotDayTrans(df_train_trans.loc[(df_train_trans['TransactionDT']\

                                 >= (day_start*DAY)) & (df_train_trans['TransactionDT'] <= (day_end*DAY))\

                                & df_train_trans['isFraud'] == True])
print("transaction quantitative/qualitative breakdown")

quantitative = [f for f in df_train_trans.columns if df_train_trans.dtypes[f] != 'object']

#quantitative.remove('')

qualitative = [f for f in df_train_trans.columns if df_train_trans.dtypes[f] == 'object']

for numeric in quantitative:

    print(numeric)

print('-------')

for category in qualitative:

    print(category)
print("transaction quantitative/qualitative breakdown")

quantitative = [f for f in df_train_trans.columns if df_train_trans.dtypes[f] != 'object']

qualitative = [f for f in df_train_trans.columns if df_train_trans.dtypes[f] == 'object']

print(np.shape(df_train_trans))

#most of the columns are quantiative

print(len(quantitative),len(qualitative))
print(np.sum(df_train_trans.index.isin(df_train_ident.index.unique())) / len(df_train_trans))

print(np.sum(df_test_trans.index.isin(df_train_ident.index.unique())) / len(df_test_trans))
#take numpy.log of the times since they are very large integers in most cases

fig, ax = plt.subplots(1, 2, figsize=(18,4))



time_val = df_train_trans.loc[df_train_trans['isFraud'] == 1]['TransactionDT'].values



sns.distplot(np.log(time_val), ax=ax[0], color='r')

ax[0].set_title('Distribution of LOG TransactionDT, isFraud=1', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])



time_val = df_train_trans.loc[df_train_trans['isFraud'] == 0]['TransactionDT'].values



sns.distplot(np.log(time_val), ax=ax[1], color='b')

ax[1].set_title('Distribution of LOG TransactionDT, isFraud=0', fontsize=14)

ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])



plt.show()
#is there a correlation between email domain and fraud occuring?



fraud_domains = df_train_trans.loc[df_train_trans['isFraud'] == 1]['P_emaildomain'].unique().tolist()

no_fraud_domains = df_train_trans['P_emaildomain'].unique().tolist()



for domain in fraud_domains:

    no_fraud_domains.remove(domain)



#print(len(df_train_trans.isin(fraud_domains)['P_emaildomain']))

print(no_fraud_domains)
#V columns (V1 to V339)

missing_percent = 0.47 * len(df_train_trans)

missing_list = []



for i in range(1,340):

    if df_train_trans['V'+str(i)].isnull().sum() > missing_percent:

        missing_list.append(i)

        

print(len(missing_list))

print(missing_list)
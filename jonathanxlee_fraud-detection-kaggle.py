import pandas as pd 

import numpy as np 



import matplotlib.pyplot as plt 



from scipy import stats

import seaborn as sns 



from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.preprocessing import LabelEncoder





%matplotlib inline
# From Kaggle Kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage and https://www.kaggle.com/wti200/data-preparation-outlier-analysis

def reduce_mem_usage2(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

train_transaction = pd.read_csv('input/train_transaction.csv.zip')

train_transaction = reduce_mem_usage2(train_transaction)

test_transaction = pd.read_csv('input/test_transaction.csv.zip')

test_transaction = reduce_mem_usage2(test_transaction)
train_identity = pd.read_csv('train_identity.csv.zip')

train_identity = reduce_mem_usage2(train_identity)

test_identity = pd.read_csv('test_identity.csv.zip')

test_identity = reduce_mem_usage2(test_identity)
train_transaction[train_transaction['isFraud']==1].head(n=10)
def bclass_count_graph(df, ax, title):





    totals = df.groupby(by=train_transaction['isFraud']).size()



    sum_not = totals[0]

    sum_fraud = totals[1]



    perc_fraud = sum_fraud/totals.sum()

    perc_not = sum_not/totals.sum()



    ax.bar([('Not Fraud (%.2f%s)' %(perc_not*100,'%')),('Fraud  (%.2f%s)' %(perc_fraud*100,'%'))], [sum_not, sum_fraud], color=['green','red'])

    ax.set_ylabel('Percent ot Total Observations')

    ax.set_title(title)

    

    

class_counts = plt.figure(figsize=(16,6))



class_counts.add_axes()

train_ax = class_counts.add_subplot(121)

bclass_count_graph(train_transaction , train_ax,'Distribution of Observations for Train Data')



test_ax = class_counts.add_subplot(122)

bclass_count_graph(test_transaction , test_ax,'Distribution of Observations for Test Data')



dist_trans, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 6))



train_transaction[train_transaction['isFraud'] == 1]['TransactionAmt'].apply(np.log).plot(kind='hist',

                                                                            bins=100,

                                                                            ax=ax1,              

                                                                            color='red',

                                                                            title='Log Transformed Distribution of Fraud Transaction Amounts')



train_transaction[train_transaction['isFraud'] == 0]['TransactionAmt'].apply(np.log).plot(kind='hist',

                                                                            bins=100,

                                                                            ax=ax2,

                                                                            title='Log Transformed Distribution of Legal Transaction Amounts')



train_transaction[train_transaction['isFraud'] == 1]['TransactionAmt'].plot(kind='hist',

                                                                            bins=100,

                                                                            ax=ax3,

                                                                            color='red',

                                                                            title='Distribution of Fraud Transaction Amounts')



train_transaction[train_transaction['isFraud'] == 0]['TransactionAmt'].plot(kind='hist',

                                                                            bins=100,

                                                                            ax=ax4,

                                                                            title='Distribution of Legal Transaction Amounts')

df_train = pd.merge(train_transaction,train_identity,how='left',on='TransactionID')

df_test = pd.merge(test_transaction,test_identity,how='left',on='TransactionID')
dist_trans,((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16, 6))



df_train[df_train['isFraud'] == 0]['DeviceType'].value_counts(dropna=False).plot(kind='pie', 

                                                                     ax=ax1,

                                                                     title='Observations that are not Fraud')



df_train[df_train['isFraud'] == 1]['DeviceType'].value_counts(dropna=False).plot(kind='pie', 

                                                                     ax=ax2,

                                                                     title='Observations of Fraud')

df_train[df_train['isFraud'] == 0]['DeviceInfo'].value_counts(dropna=False).head().plot(kind='pie', 

                                                                     ax=ax3)

df_train[df_train['isFraud'] == 1]['DeviceInfo'].value_counts(dropna=False).head().plot(kind='pie', 

                                                                     ax=ax4)
dist_trans,(ax1, ax2) = plt.subplots(1,2, figsize=(16, 6))



df_train[df_train['isFraud'] == 0]['ProductCD'].value_counts(dropna=False).plot(kind='bar', 

                                                                     ax=ax1,

                                                                     title='Observations that are not Fraud')



df_train[df_train['isFraud'] == 1]['ProductCD'].value_counts(dropna=False).plot(kind='bar', 

                                                                     ax=ax2,

                                                                     title='Observations of Fraud')
dist_trans,(ax1, ax2) = plt.subplots(1,2, figsize=(16, 6))

df_train[df_train['isFraud'] == 0]['card4'].value_counts(dropna=False).plot(kind='bar', 

                                                                     ax=ax1,

                                                                     title='Observations that are not Fraud')



df_train[df_train['isFraud'] == 1]['card4'].value_counts(dropna=False).plot(kind='bar', 

                                                                     ax=ax2,

                                                                     title='Observations of Fraud')
df_train[['card3', 'card5']].describe()
# Label Encoding https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt

for f in df_train.columns:

    if  df_train[f].dtype == 'object': 

        lbl = LabelEncoder()

        lbl.fit(list(df_train[f].values) + list(df_test[f].values))

        df_train[f] = lbl.transform(list(df_train[f].values))

        df_test[f] = lbl.transform(list(df_test[f].values)) 

        

df_train.reset_index()

df_test.reset_index()



y = df_train['isFraud']

x = df_train.drop(['isFraud'],axis=1)



# fit model no training data

model = XGBClassifier(objective='binary:logistic')

model.fit(x, y)



print(model)

pred = model.predict_proba(df_test)



print(pred)
predictions = [value[1] for value in pred]

print(predictions)
import csv



with open('submission.csv', mode='w') as submissiont_file:

    submission_writer = csv.writer(submissiont_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    submission_writer.writerow(['TransactionID' , 'isFraud'])

    for i in range(len(pred)): 

        submission_writer.writerow([df_test.TransactionID.loc[i], predictions[i]])
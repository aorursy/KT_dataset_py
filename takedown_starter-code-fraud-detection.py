# import packages

import pandas as pd

import numpy as np

from scipy import stats

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

import os 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_recall_curve

from sklearn.metrics import recall_score, classification_report, auc, roc_curve

from sklearn.metrics import precision_recall_fscore_support, f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from pylab import rcParams

from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers



#set random seed

RANDOM_SEED = 314 



rcParams['figure.figsize'] = 14, 8.7 # Golden Mean

LABELS = ["Normal","Fraud"]

sns.set()
## Function to reduce the DF size

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
path = '../input'

name=[]



for i in os.listdir(path):

        name.append(i)

print(name)    

for j in name:

        new_j = j.replace('.csv','')

        print('Creating {} DataFrame'.format(new_j))

        new_j=exec('{} = pd.read_csv(os.path.join(path,j))'.format(new_j))

        print('Done creating.') #format(new_j)
train_transaction = reduce_mem_usage(train_transaction)
train_transaction.info()
train_transaction.shape
train_transaction.head()
obj_cols = train_transaction.dtypes

obj_cols[obj_cols=='object']
train_transaction.isna().sum()#check to see if any values are null, which there are not
pd.set_option('precision', 3)

train_transaction.describe()
#visualizations of time and amount

plt.figure(figsize=(15,7))

plt.title('Distribution of Time Feature')

sns.distplot(train_transaction.TransactionDT)
plt.figure(figsize=(15,7))

plt.title('Distribution of Monetary Value Feature')

sns.distplot(train_transaction.TransactionAmt)
count = train_transaction.isFraud.value_counts()

regular = count[0]

frauds = count[1]

total= frauds + regular 

perc_reg = (regular/total)*100

perc_frauds = (frauds/total)*100

print('There were {} non-fraudulent transactions ({:.3f}%) and {} fraudulent transactions ({:.3f}%).'.format(regular, perc_reg, frauds, perc_frauds))
plt.figure(figsize=(15,7))

sns.countplot(x='isFraud',data=train_transaction)

plt.title('CountPlot Frauds 1 = Positive , 0 = Negative')
#Macro on correlations 

corr = train_transaction.corr()

corr
#heatmap

plt.figure(figsize=(15,7))

sns.heatmap(corr)

plt.title('Heatmap correlations Train_data')
frauds = train_transaction[train_transaction['isFraud']==1]
notfrauds= train_transaction[train_transaction['isFraud']==0]
frauds.TransactionAmt.describe()
notfrauds.TransactionAmt.describe()
#plot of high value transactions

plt.figure(figsize=(15,7))

bins = np.linspace(200, 2500, 100)

plt.hist(notfrauds.TransactionAmt, bins, alpha=1, normed=True, label='Normal')

plt.hist(frauds.TransactionAmt, bins, alpha=0.6, normed=True, label='Fraud')

plt.legend(loc='upper right')

plt.title("Amount by percentage of transactions (transactions \$200+)")

plt.xlabel("Transaction amount (USD)")

plt.ylabel("Percentage of transactions (%)");

plt.show()
#train_x, test_x = train_test_split(XX, test_size=0.2, random_state=RANDOM_SEED) xx == Merged dataframe

#train_x = train_x[train_x.isFraud == 0] #where normal transactions

#train_x = train_x.drop(['isFraud'], axis=1) #drop the class column





#test_y = test_x['isFraud'] #save the class column for the test set

#test_x = test_x.drop(['isFraud'], axis=1) #drop the class column



#train_x = train_x.values #transform to ndarray

#test_x = test_x.values
#The magics

feats= ['TransactionID',

 'C14',

 'C13',

 'C12',

 'C11',

 'C10',

 'C8',

 'C7',

 'C6',

 'C5',

 'C4',

 'C3',

 'C2',

 'C1',

 'C9',

 'isFraud',

 'TransactionDT',

 'TransactionAmt',

 'ProductCD',

 'card1']
sns.set()

plt.figure(figsize=(15,7))

train_transaction[feats].isna().sum().sort_values(ascending=False).plot(kind='barh')
corr_matrix = train_transaction[feats].corr()

corr_matrix.isFraud.sort_values(ascending=False)
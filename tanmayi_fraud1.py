import pandas as pd

import numpy as np

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity="all"
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
warnings.simplefilter('ignore')

files = ['../input/ieee-fraud-detection/test_identity.csv', 

         '../input/ieee-fraud-detection/test_transaction.csv',

         '../input/ieee-fraud-detection/train_identity.csv',

         '../input/ieee-fraud-detection/train_transaction.csv']



def load_data(file):

    return reduce_mem_usage(pd.read_csv(file))



with multiprocessing.Pool() as pool:

    test1, test2, train1, train2 = pool.map(load_data, files)
train2['TransactionAmt'] = train2['TransactionAmt'].astype(float)

total = len(train2)

total_amt = train2.groupby(['isFraud'])['TransactionAmt'].sum().sum()



plt.figure(figsize=(12,5))

plt.subplot(121)



plot_tr = sns.countplot(x='isFraud', data=train2)

plot_tr.set_title("Fraud Transactions Distribution \n 0: No Fraud | 1: Fraud", fontsize=18)

plot_tr.set_xlabel("Is fraud?", fontsize=16)

plot_tr.set_ylabel('Count', fontsize=16)

for p in plot_tr.patches:

    height = p.get_height()

    plot_tr.text(p.get_x()+p.get_width()/2.,height + 3, '{:1.2f}%'.format(height/total*100),ha="center", fontsize=15) 

    

        

plt.subplot(122)

percent_amt = (train2.groupby(['isFraud'])['TransactionAmt'].sum())

percent_amt = percent_amt.reset_index()

plot_tr_2 = sns.barplot(x='isFraud', y='TransactionAmt',  dodge=True, data=percent_amt)

plot_tr_2.set_title("% Total Amount in Transaction Amt \n 0: No Fraud | 1: Fraud", fontsize=18)

plot_tr_2.set_xlabel("Is fraud?", fontsize=16)

plot_tr_2.set_ylabel('Total Transaction Amount Scalar', fontsize=16)

for p in plot_tr_2.patches:

    height = p.get_height()

    plot_tr_2.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}%'.format(height/total_amt * 100),ha="center", fontsize=15) 
tmp = pd.crosstab(train2['ProductCD'], train2['isFraud'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)



plt.figure(figsize=(14,10))

plt.suptitle('ProductCD Distributions', fontsize=22)



plt.subplot(221)

plot_1 = sns.countplot(x='ProductCD', data=train2)

# plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])



plot_1.set_title("ProductCD Distribution", fontsize=18)

plot_1.set_xlabel("ProductCD Name", fontsize=16)

plot_1.set_ylabel("Count", fontsize=17)

plot_1.set_ylim(0,500000)

for p in plot_1.patches:

    height = p.get_height()

    plot_1.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}%'.format(height/total*100),ha="center", fontsize=14) 



plt.subplot(222)

plot_2 = sns.countplot(x='ProductCD', hue='isFraud', data=train2)

plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])

plot_2_2 = plot_2.twinx()

plot_2_2 = sns.pointplot(x='ProductCD', y='Fraud', data=tmp, color='black', order=['W', 'H',"C", "S", "R"], legend=False)

plot_2_2.set_ylabel("% of Fraud Transactions", fontsize=16)



plot_2.set_title("Product CD by Target(isFraud)", fontsize=18)

plot_2.set_xlabel("ProductCD Name", fontsize=16)

plot_2.set_ylabel("Count", fontsize=16)



plt.subplot(212)

plot_3 = sns.boxenplot(x='ProductCD', y='TransactionAmt', hue='isFraud', 

              data=train2[train2['TransactionAmt'] <= 2000] )

plot_3.set_title("Transaction Amount Distribuition by ProductCD and Target", fontsize=18)

plot_3.set_xlabel("ProductCD Name", fontsize=16)

plot_3.set_ylabel("Transaction Values", fontsize=16)



plt.subplots_adjust(hspace = 0.6, top = 0.85)



plt.show();
def cards(column):

    plt.figure(figsize=(14,10))

    plt.suptitle('Card 4 Distributions', fontsize=22)



    plt.subplot(221)

    plot_1 = sns.countplot(x=column, data=train2)

    plot_1.set_title(f'{column} Distributions ', fontsize=19)

    plot_1.set_ylim(0,420000)

    plot_1.set_xlabel(f"{column} Category Names", fontsize=15)

    plot_1.set_ylabel("Count", fontsize=15)

    for p in plot_1.patches:

        height = p.get_height()

        plot_1.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}%'.format(height/total*100),ha="center",fontsize=14)

    

    

    

    plt.subplot(222)

    plot_2 = sns.countplot(x=column, hue='isFraud', data=train2)

    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])

    plot_2.set_title(f"{column}by Target(isFraud?)", fontsize=19)

    plot_2.set_xlabel(f"{column}Category Names", fontsize=15)

    plot_2.set_ylabel("Count", fontsize=15);

    

    

    plt.subplot(212)

    plot_3 = sns.boxenplot(x=column, y='TransactionAmt', hue='isFraud', 

              data=train2[train2['TransactionAmt'] <= 2000] )

    plot_3.set_title("Transaction Amount Distribuition by ProductCD and Target", fontsize=18)

    plot_3.set_xlabel(f"{column}", fontsize=16)

    plot_3.set_ylabel("Transaction Values", fontsize=16)

    

    plt.subplots_adjust(hspace = 0.6, top = 0.85)



    plt.show();
cards('card4')
cards('card6')
card=['card1','card2','card3','card5']

sns.pairplot(train2[card])
def ploting_dist_ratio(DataFile, Column, lim=2000):

    tmp = pd.crosstab(DataFile[Column], DataFile['isFraud'], normalize='index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)



    plt.figure(figsize=(20,5))

    plt.suptitle(f'{Column} Distributions ', fontsize=22)



    plt.subplot(121)

    plot_1 = sns.countplot(x=Column, data=DataFile)

    plot_1.set_title(f"{Column} Distribution\nCound and %Fraud by each category", fontsize=18)

    plot_1.set_ylim(0,400000)

    plot_1.set_xlabel(f"{Column} Category Names", fontsize=16)

    plot_1.set_ylabel("Count", fontsize=17)

    for p in plot_1.patches:

        height = p.get_height()

        plot_1.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}%'.format(height/total*100),ha="center",fontsize=14) 

        

    

    plt.subplot(122)

    plot_2 = sns.boxenplot(x=Column, y='TransactionAmt', hue='isFraud', data=DataFile[DataFile['TransactionAmt'] <= lim])

    plot_2.set_title(f"{Column} by Transactions dist", fontsize=18)

    plot_2.set_xlabel(f"{Column} Category Names", fontsize=16)

    plot_2.set_ylabel("Transaction Amount(U$)", fontsize=16)

    #plt.close(12)

    

    plt.subplots_adjust(hspace=.4, wspace = 0.35, top = 0.80)

    plt.show();
for columns in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:

    ploting_dist_ratio(train2, columns, lim=2000)
def h5(df):

  c1=[]

  n=df.iloc[:,17:31].columns

  for i in n:

     c1.append(i)

  n1=df.iloc[:,17:31].isnull().sum()

  print(n1)

  return c1

c1=h5(train2)
def ploting_cnt_amt(DataFile, Column, lim=3500):

   

    plt.figure(figsize=(10,5))    

     

    plot_2 = sns.lineplot(x=Column,y='TransactionAmt',hue='isFraud', palette="Set2",data=DataFile[DataFile['TransactionAmt'] <= lim])

  

    plot_2.set_xticklabels(plot_2.get_xticklabels(),rotation=45)

    plot_2.set_title(f"{Column} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)

    plot_2.set_xlabel(f"{Column} Category Names", fontsize=16)

    plot_2.set_ylabel("Transaction Amount(U$)", fontsize=16)

    plot_2.set_xticklabels(plot_2.get_xticklabels(),rotation=45)  

    

    plt.show()
for columns in c1:

    ploting_cnt_amt(train2, columns)
for columns in ['D1','D2','D3','D4','D5','D6','D7','D8','D9']:

    ploting_cnt_amt(train2, columns)
def ploting_cnt_amt(DataFile, Column, lim=3500):

   

    plt.figure(figsize=(30,10))    

     

    plot_2 = sns.boxenplot(x=Column,y='TransactionAmt',hue='isFraud',data=DataFile[DataFile['TransactionAmt'] <= lim])

  

    plot_2.set_xticklabels(plot_2.get_xticklabels(),rotation=45)

    plot_2.set_title(f"{Column} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)

    plot_2.set_xlabel(f"{Column} Category Names", fontsize=16)

    plot_2.set_ylabel("Transaction Amount(U$)", fontsize=16)

    plot_2.set_xticklabels(plot_2.get_xticklabels(),rotation=45)  

    

    plt.show()
ploting_cnt_amt(train2, 'R_emaildomain')
ploting_cnt_amt(train2, 'P_emaildomain')
m=pd.merge(train1, train2,how='outer', on='TransactionID')

train=pd.DataFrame(m)

m1=pd.merge(test1, test2,how='outer', on='TransactionID')

test=pd.DataFrame(m1)
train.shape

test.shape
train.info()

test.info()
def find_null(data):

    nulval = data.isna().sum()/data.shape[0]*100

    null_cols = np.array(nulval[nulval>15].index)

    data=data.drop(null_cols,axis=1)

    return data

train=find_null(train)

test=find_null(test)
def dty(data,obj):

  d=data.select_dtypes(include=[obj]).dtypes

  h2=[]

  for i in d.index:

     h2.append(i)

  return h2

objcols=dty(train,'object')

objcols1=dty(test,'object')
def dty1(data,obj):

  d=data.select_dtypes(exclude=[obj]).dtypes

  h2=[]

  for i in d.index:

     h2.append(i)

  return h2

flintcl=dty1(train,'object')

flintcl1=dty1(test,'object')
for i in objcols:

    train[i] = train[i].replace(np.nan, 'unknown')

for j in flintcl:

      train[j] = train[j].replace(np.nan, train[j].min())
for i in objcols1:

    test[i] = test[i].replace(np.nan, 'unknown')

for j in flintcl1:

      test[j] = test[j].replace(np.nan, test[j].min())
train.shape

test.shape
from sklearn.preprocessing import LabelEncoder

def labele(catcols,data):

    cat_col =catcols

    le = LabelEncoder()

    for i in cat_col:

        data[i] = le.fit_transform(data[i])

labele(objcols,train)

labele(objcols1,test)

        
def null_col(obj):

  null_columns=train.columns[train.isnull().any()]

  null=train[null_columns].isnull().sum()

  n=[]

  for i in null_columns:

    n.append(i)

  dty=train[n].select_dtypes(include=[obj]).dtypes

  return dty

null_col('float64')
s=[]

for i in test.columns:

    if i not in train.columns:

        s.append(i)

test=test.drop(s,axis=1)
y = train['isFraud'] # Y_test

x=train.drop('isFraud',axis=1) # X_test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=101)



x_train.head()

y_train.head()

y_test.head()

x_test.head()
#import

from sklearn.preprocessing import StandardScaler



# Initialize

scaler=StandardScaler()



#Apply

scaler.fit(x_train)

# Apply transform to both the training set and the test set.

train_std = scaler.transform(x_train)

test_std = scaler.transform(x_test)
scaler1=StandardScaler()

#Apply

scaler1.fit(test)

# Apply transform to both the training set and the test set.

r_test_std = scaler1.transform(test)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report,accuracy_score
gb_clf2 = GradientBoostingClassifier(n_estimators=70,max_features=100, learning_rate=0.3, max_depth=9, random_state=0)

gb_clf2.fit(train_std, y_train)

predictions = gb_clf2.predict(test_std)



print("Confusion Matrix:")

print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
FPR,TPR,_=roc_curve(y_test,predictions)

roc_auc=auc(FPR,TPR)

print('ROC AUC: %0.3f' % roc_auc)
plt.figure(figsize=(10,10))

plt.plot(FPR,TPR,label='Roc curve (area=%0.3f)'% roc_auc)

plt.plot([0,1],[0,1],'k--')

plt.xlim([-0.05,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('FP rate')

plt.ylabel('TP rate')

plt.title('ROC curve')

plt.legend(loc="lower right")

plt.show();
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y_test,predictions)
y_pred=gb_clf2.predict(r_test_std)
Submission=pd.DataFrame({'TransactionId': test['TransactionID'],'isFraud':y_pred})

Submission=Submission[['TransactionId','isFraud']]

Submission.to_csv("Submission.csv",Index=False)
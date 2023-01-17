####################################################################################

# import Packages

####################################################################################

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import gc

import time

import warnings

import os



####################################################################################        

# pre-setting

####################################################################################

%matplotlib inline

warnings.filterwarnings('ignore')

np.random.seed(seed=777)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

pd.set_option("display.max_colwidth", 80)

pd.options.display.precision = 3

gc.enable()



####################################################################################        

# Read Dataset and make train, test data

####################################################################################

train_identity   = pd.read_csv("/kaggle/input/train_identity.csv"   )

train_transaction= pd.read_csv("/kaggle/input/train_transaction.csv")

test_identity    = pd.read_csv("/kaggle/input/test_identity.csv"    )

test_transaction = pd.read_csv("/kaggle/input/test_transaction.csv" )

print("Read csv files")



train = train_transaction.merge(train_identity, how='outer', on='TransactionID',sort ='True')

test  = test_transaction.merge (test_identity , how='outer', on='TransactionID',sort ='True')

del train_identity,train_transaction,test_identity, test_transaction

print("Make train & test data")



####################################################################################        

# write data frame to fearher

####################################################################################

train.to_feather("train.feather")

test.to_feather("test.feather")

print("Write feather data")
####################################################################################   

# Read DataFrame from feather

####################################################################################   

train = pd.read_feather("train.feather")

test  = pd.read_feather("test.feather")
####################################################################################   

# Define Functions

####################################################################################  



####################################################################################   

# make graph 

#  s:start column number, e:end column, max : max data number for object

#  xl:set x log scale  yl:set y log scale 

def make_graph(df,s,e,xl=0,yl=0,m=0):

    for col in df.iloc[:,s:e]:

        if(df[col].dtype != 'object'):

            #Data

            print("column:",col)

            print("dtype:{}  number:{}  unique:{}  NaN:{}".format(df[col].dtype, df[col].count().sum(),

                                                                  df[col].nunique(), df[col].isna().sum()))

            print("min:{}  max:{}  ave:{:.3f}".format(df[col].min(),df[col].max(),df[col].mean()))

            #Graph

            fig, ax = plt.subplots(1, 3, figsize=(16,2))

                        

            if(xl==1):

                ax[0].set_xscale("log")

                ax[1].set_xscale("log")

                ax[2].set_xscale("log")

                

            if(yl==1):

                ax[0].set_yscale("log")

                ax[1].set_yscale("log")

                ax[2].set_yscale("log")

                

        

            sns.distplot(df[col], ax = ax[0], bins=50, kde = False ,color="b")

            ax[0].set_title("Hist : " + col,fontsize=14)

        

            sns.distplot(df.loc[df['isFraud']==1, col], ax = ax[1], bins=50, kde = False ,color="r")

            ax[1].set_title("Hist : " + col + " (Fraud:1)",fontsize=14)

        

            sns.distplot(df.loc[df['isFraud']==0, col], ax = ax[2], bins=50, kde = False ,color="g")

            ax[2].set_title("Hist : " + col + " (Fraud:0)",fontsize=14)

        

            plt.show()

    

        else:

            #Data

            print("column:",col)

            print("dtype:{}  unique:{}  NaN:{}".format(df[col].dtype, df[col].nunique(), df[col].isna().sum()))

        

            #Graph

            fig, ax = plt.subplots(1, 3, figsize=(16,3))

        

            for i in range(3):

                labels = ax[i].get_xticklabels()        

                plt.setp(labels, rotation=45, fontsize=10)   

            

            if(m == 0):

                col_tmp0 = df[col].value_counts()

                col_tmp1 = df.loc[df['isFraud'] == 1 , col].value_counts()

                col_tmp2 = df.loc[df['isFraud'] == 0 , col].value_counts()

            else:

                col_tmp0 = df[col].value_counts().head(m)

                col_tmp1 = df.loc[df['isFraud'] == 1 , col].value_counts().head(m)

                col_tmp2 = df.loc[df['isFraud'] == 0 , col].value_counts().head(m)



            if(xl==1):

                ax[0].set_xscale("log")

                ax[1].set_xscale("log")

                ax[2].set_xscale("log")

                

            if(yl==1):

                ax[0].set_yscale("log")

                ax[1].set_yscale("log")

                ax[2].set_yscale("log")   

                            

            sns.barplot(ax = ax[0], x = col_tmp0.index, y = col_tmp0)

            sns.barplot(ax = ax[1], x = col_tmp1.index, y = col_tmp1)

            sns.barplot(ax = ax[2], x = col_tmp2.index, y = col_tmp2)

            

            ax[0].set_title("Hist : " + col,fontsize=14)

            ax[1].set_title("Hist : " + col + " (Fraud:1)",fontsize=14)

            ax[2].set_title("Hist : " + col + " (Fraud:0)",fontsize=14)

                   

            plt.show()

            continue

    return

#################################################################################### 

#Check data of Train

#################################################################################### 

train.head(15)
make_graph(train,1,4,0,0,0)
make_graph(train,5,15,0,0,0)
print('_'*80)

print("P_emaildomain")

print(train['P_emaildomain'].value_counts())

print('_'*80)

print("R_emaildomain")

print(train['R_emaildomain'].value_counts())
make_graph(train,15,17,0,0,10)
make_graph(train,17,31,0,1,0)
make_graph(train,31,46,0,1,0)
make_graph(train,46,55,0,1,0)
make_graph(train,394,423,0,0,10)
make_graph(train,423,425,0,0,10)

make_graph(train,426,427,0,0,10)
make_graph(train,425,426,0,0,0)

make_graph(train,427,432,0,0,0)
print('_'*80)

print("Device Info")

print(train['DeviceInfo'].value_counts())
make_graph(train,432,434,0,0,10)
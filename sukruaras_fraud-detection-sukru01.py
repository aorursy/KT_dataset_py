import numpy as np

import pandas as pd, os, gc

gc.collect()
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)



# to display all rowss:

pd.set_option('display.max_rows', None)



from sklearn.model_selection import train_test_split, GridSearchCV



# %% [markdown]

# # Libraries



# %% [code]

# data analysis libraries:

import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

import numpy as np, pandas as pd, os, gc

from sklearn.model_selection import GroupKFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

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

import gc

import time

from contextlib import contextmanager
# Read train and test data with pd.read_csv():



train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

#  combine the data and work with the whole dataset



train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left',left_index=True, right_index=True)

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left', left_index=True, right_index=True)
# export dataset into computer as excel



#df.to_excel(r'Path where you want to store the exported excel file\File Name.xlsx', sheet_name='Your sheet name', index = False)

# df.to_excel (export_file_path, index = False, header=True)

#train.to_excel(r"E:\DATA SCIENCE\KAGGLE PROJECTS\fraudetection\train.xlsx", index = False )
train.head(2)
test.head(2)
del train_identity, train_transaction, test_identity, test_transaction; gc.collect()
train.shape, test.shape
train.isnull().any().sum()
train.info()
test.info()
train.isnull().all()
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
## REducing memory

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.count()
train.isnull().count()
rowsperct=(train.count()/train.isnull().count())*100
rowsperct, gc.collect()
train.shape, test.shape
correlat1=train.corrwith( train['isFraud'], method= 'spearman')
cortable=pd.DataFrame(correlat1)
cortable
# FEATURE SELECTION

# correlat=train.corrwith( train['isFraud'], method= 'spearman')

#  correlat1=pd.DataFrame(correlat,columns=["corr"])

# correlat1.reset_index()

#  cor_features=correlat1.loc[(correlat1.loc[:,"corr"] > 0.11)|(correlat1.loc[:,"corr"]<-0.11)]
gc.collect()

# TARGET 

yiftrain =train['isFraud'].copy(); gc.collect()
yiftrain.shape
# ELIMINATION UNRELIABLE VARIABLES FROM TRAIN DATASET



train1= train.drop(['V1', 'V2', 'V4', 'V6', 'V7', 'V8','V9', 'V10', 'V12', 'V13', 'V14', 'V15','V16', 'V17', 'V18','V19','V20', 'V21', 'V22', 'V24', 'V25', 'V26','V27', 'V28', 'V29', 'V30', 

             'V31', 'V32', 'V35', 'V36','V37','V38', 'V39', 'V40', 'V41', 'V42', 'V43','V45', 'V46', 'V47', 'V49', 'V50', 'V51', 'V53','V54', 'V55', 'V56', 'V57', 'V58', 'V59','V60',

             'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67','V68', 'V69', 'V71', 'V72', 'V73', 'V75', 'V76','V77', 'V78', 'V79', 'V80', 'V81', 'V82','V83','V84', 'V85', 'V86', 'V88', 'V89',

             'V91', 'V92', 'V93', 'V95', 'V96', 'V97', 'V98', 'V100','V101', 'V104', 'V105', 'V106', 'V107', 'V108','V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115','V116', 'V117',

             'V118','V119','V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126','V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136','V137','V138', 'V139', 'V141', 'V142',

             'V143', 'V144','V145', 'V146','V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153','V154', 'V155', 'V157','V160','V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167',

             'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176','V177', 'V178', 'V179', 'V180', 'V181', 'V182','V183','V184', 'V185', 'V186', 'V187', 'V188', 'V189','V190','V191', 

             'V192', 'V193', 'V194','V195', 'V196', 'V197', 'V198','V199', 'V200','V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208','V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215',

            'V216', 'V217', 'V218','V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226','V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236','V237','V238', 'V239',

            'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247','V248', 'V249', 'V250', 'V251', 'V252', 'V253','V254', 'V255', 'V256', 'V257', 'V259','V260', 'V261', 'V262', 'V263', 'V265',

            'V266', 'V267','V268', 'V269', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276','V277', 'V278', 'V279', 'V280', 'V281', 'V282','V284', 'V286', 'V287', 'V288', 'V289', 'V291', 'V292', 'V293',

             'V295', 'V296', 'V297','V298','V299', 'V300', 'V301', 'V302', 'V304', 'V305', 'V306', 'V307', 'V308','V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318','V319',

            'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326','V327', 'V328', 'V329', 'V330', 'V331', 'V333', 'V334', 'V335', 'V336','V337','V338', 'V339', 'id_33'], axis=1);x = gc.collect()
train1.head(2)
# ELIMINATION UNRELIABLE VARIABLES FROM TEST DATASET



test1= test.drop(['V1', 'V2', 'V4', 'V6', 'V7', 'V8','V9', 'V10', 'V12', 'V13', 'V14', 'V15','V16', 'V17', 'V18','V19','V20', 'V21', 'V22', 'V24', 'V25', 'V26','V27', 'V28', 'V29', 'V30', 

             'V31', 'V32', 'V35', 'V36','V37','V38', 'V39', 'V40', 'V41', 'V42', 'V43','V45', 'V46', 'V47', 'V49', 'V50', 'V51', 'V53','V54', 'V55', 'V56', 'V57', 'V58', 'V59','V60',

             'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67','V68', 'V69', 'V71', 'V72', 'V73', 'V75', 'V76','V77', 'V78', 'V79', 'V80', 'V81', 'V82','V83','V84', 'V85', 'V86', 'V88', 'V89',

             'V91', 'V92', 'V93', 'V95', 'V96', 'V97', 'V98', 'V100','V101', 'V104', 'V105', 'V106', 'V107', 'V108','V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115','V116', 'V117',

             'V118','V119','V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126','V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136','V137','V138', 'V139', 'V141', 'V142',

             'V143', 'V144','V145', 'V146','V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153','V154', 'V155', 'V157','V160','V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167',

             'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176','V177', 'V178', 'V179', 'V180', 'V181', 'V182','V183','V184', 'V185', 'V186', 'V187', 'V188', 'V189','V190','V191', 

             'V192', 'V193', 'V194','V195', 'V196', 'V197', 'V198','V199', 'V200','V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208','V209', 'V210', 'V211', 'V212', 'V213', 'V214', 'V215',

            'V216', 'V217', 'V218','V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226','V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236','V237','V238', 'V239',

            'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247','V248', 'V249', 'V250', 'V251', 'V252', 'V253','V254', 'V255', 'V256', 'V257', 'V259','V260', 'V261', 'V262', 'V263', 'V265',

            'V266', 'V267','V268', 'V269', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276','V277', 'V278', 'V279', 'V280', 'V281', 'V282','V284', 'V286', 'V287', 'V288', 'V289', 'V291', 'V292', 'V293',

             'V295', 'V296', 'V297','V298','V299', 'V300', 'V301', 'V302', 'V304', 'V305', 'V306', 'V307', 'V308','V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318','V319',

            'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326','V327', 'V328', 'V329', 'V330', 'V331', 'V333', 'V334', 'V335', 'V336','V337','V338', 'V339', 'id-33'], axis=1);x = gc.collect()
test1.head(2)
train1.shape, test1.shape
del train, test ; gc.collect()
# del train1['isFraud']; gc.collect()
# PLOT ORIGINAL D

plt.figure(figsize=(15,5))

plt.scatter(train1.TransactionDT,train1.D15)

plt.title('Original D15')

plt.xlabel('Time')

plt.ylabel('D15')

plt.show()
train1.shape, test1.shape
# NORMALIZE D COLUMNS

for i in range(1,16):

    if i in [1,2,3,5,9]: continue

    train1['D'+str(i)] =  train1['D'+str(i)] - train1.TransactionDT/np.float32(24*60*60)

    test1['D'+str(i)] = test1['D'+str(i)] - test1.TransactionDT/np.float32(24*60*60) 
# PLOT TRANSFORMED D

plt.figure(figsize=(15,5))

plt.scatter(train1.TransactionDT,train1.D15)

plt.title('Transformed D15')

plt.xlabel('Time')

plt.ylabel('D15n')

plt.show()
train1.shape, test1.shape
yiftrain.shape
train1= train1.drop(['C1', 'C2', 'C3', 'C6', 'C8','C9', 'C10', 'C11', 'C13', 'C14', 'C14', 'D4', 'D5', 'D6', 'D7','D8', 'D9', 'D10', 'D11', 'D12', 'D13','D14',

                    'M2', 'M3', 'M5', 'M6','M7', 'M8', 'M9'], axis=1)
test1= test1.drop(['C1', 'C2', 'C3', 'C6', 'C8','C9', 'C10', 'C11', 'C13', 'C14', 'C14', 'D4', 'D5', 'D6', 'D7','D8', 'D9', 'D10', 'D11', 'D12', 'D13','D14',

                    'M2', 'M3', 'M5', 'M6','M7', 'M8', 'M9'], axis=1)
train1.shape, test1.shape
#   Mapping emails



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



us_emails = ['gmail', 'net', 'edu']



# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654

for c in ['P_emaildomain', 'R_emaildomain']:

    train1[c + '_bin'] = train1[c].map(emails)

    test1[c + '_bin'] = test1[c].map(emails)

    

    train1[c + '_suffix'] = train1[c].map(lambda x: str(x).split('.')[-1])

    test1[c + '_suffix'] = test1[c].map(lambda x: str(x).split('.')[-1])

    

    train1[c + '_suffix'] = train1[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test1[c + '_suffix'] = test1[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
train1.head(2)
train1.shape, test1.shape
# FREQUENCY ENCODE TOGETHER

def encode_FE(df1, df2, cols):

    for col in cols:

        df = pd.concat([df1[col],df2[col]])

        vc = df.value_counts(dropna=True, normalize=True).to_dict()

        vc[-1] = -1

        nm = col+'_FE'

        df1[nm] = df1[col].map(vc)

        df1[nm] = df1[nm].astype('float32')

        df2[nm] = df2[col].map(vc)

        df2[nm] = df2[nm].astype('float32')

        print(nm,', ',end='')

        

# LABEL ENCODE

def encode_LE(col,train1=train1,test1=test1,verbose=True):

    df_comb = pd.concat([train1[col],test1[col]],axis=0)

    df_comb,_ = df_comb.factorize(sort=True)

    nm = col

    if df_comb.max()>32000: 

        train1[nm] = df_comb[:len(train1)].astype('int32')

        test1[nm] = df_comb[len(train1):].astype('int32')

    else:

        train1[nm] = df_comb[:len(train1)].astype('int16')

        test1[nm] = df_comb[len(train1):].astype('int16')

    del df_comb; x=gc.collect()

    if verbose: print(nm,', ',end='')

        

# GROUP AGGREGATION MEAN AND STD

# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda

def encode_AG(main_columns, uids, aggregations=['mean'], train_df=train1, test_df=test1, 

              fillna=True, usena=False):

    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS

    for main_column in main_columns:  

        for col in uids:

            for agg_type in aggregations:

                new_col_name = main_column+'_'+col+'_'+agg_type

                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])

                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan

                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(

                                                        columns={agg_type: new_col_name})



                temp_df.index = list(temp_df[col])

                temp_df = temp_df[new_col_name].to_dict()   



                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')

                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')

                

                if fillna:

                    train_df[new_col_name].fillna(-1,inplace=True)

                    test_df[new_col_name].fillna(-1,inplace=True)

                

                print("'"+new_col_name+"'",', ',end='')        

# COMBINE FEATURES

def encode_CB(col1,col2,df1=train1,df2=test1):

    nm = col1+'_'+col2

    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)

    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 

    encode_LE(nm,verbose=False)

# GROUP AGGREGATION NUNIQUE

def encode_AG2(main_columns, uids, train_df=train1, test_df=test1):

    for main_column in main_columns:  

        for col in uids:

            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)

            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()

            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')

            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')

            print(col+'_'+main_column+'_ct, ',end='')

encode_CB('card1','addr1')
#  encode_CB('card3','addr1')
train1.shape, test1.shape,
na_low=[]

for col in train1.loc[:,'TransactionAmt':].columns: 

       if sum(train1[col].isnull())/float(len(train1.index)) < 0.30:

            na_low.append(col)

            print("'"+col+"'",', ',end='')
na_low=['TransactionAmt' , 'ProductCD' , 'card1' , 'card2' , 'card3' , 'card4' , 'card5' , 'card6' , 'addr1' ,

        'addr2' , 'P_emaildomain' , 'C4' , 'C5' , 'C7' , 'C12' ]
numeric=train1[na_low]._get_numeric_data().columns

numeric
train1.shape, test1.shape,
train1.head(1)
test1.head(1)
# Creating DAY FEATURE IN TRAIN



train1['day'] = train1.TransactionDT / (24*60*60)

train1['uid'] = train1.card1_addr1.astype(str)+'_'+np.floor(train1.day-train1.D1).astype(str)



train1.head(1)
test1['day'] = test1.TransactionDT / (24*60*60)

test1['uid'] = test1.card1_addr1.astype(str)+'_'+np.floor(test1.day-test1.D1).astype(str)
train1.shape, test1.shape
encode_FE(train1,test1,['addr1','card1','card2','card3','P_emaildomain'])
train1.shape, test1.shape
train1.head(1)
test1.head(1)
encode_FE(train1,test1,['uid'])

encode_AG( numeric, ['uid'],['mean',"std"], train1, test1, fillna=True, usena=False)
train1= train1.drop(['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',

       'id_30', 'id_31', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'], axis=1)
test1= test1.drop([ 'id-12', 'id-15', 'id-16', 'id-23', 'id-27', 'id-28', 'id-29',

       'id-30', 'id-31', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38'], axis=1)
categorical_columns=test1.columns.drop(test1._get_numeric_data().columns)

categorical_columns=categorical_columns.drop('uid')

categorical_columns
train1.shape, test1.shape
test1.head(1)
train1.head(1)
encode_AG2(categorical_columns, ['uid'], train1, test1)
# FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN

encode_FE(train1,test1,['addr1','card1','card2','card3','P_emaildomain'])

# COMBINE COLUMNS CARD1+ADDR1+P_EMAILDOMAIN

encode_CB('card1_addr1','P_emaildomain')

# FREQUENCY ENOCDE

encode_FE(train1,test1,['card1_addr1'])

# GROUP AGGREGATE

encode_AG(['TransactionAmt'],['card1','card1_addr1'],['mean','std'],usena=True)
gc.collect()
train1.shape, test1.shape
del train1['uid'], test1['uid']
train1.shape, test1.shape
%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
categorical_columns=test1.columns.drop(test1._get_numeric_data().columns)

categorical_columns
from sklearn import preprocessing

for i in categorical_columns: 

    lbe=preprocessing.LabelEncoder()

    train1[i]=lbe.fit_transform(train1[i].astype(str))
for i in categorical_columns:    

    test1[i]=lbe.fit_transform(test1[i].astype(str))
train_columns=train1.columns

train_columns=train_columns.drop("isFraud")

test1.columns=train_columns

test1.columns
train1.shape, test1.shape
for i in categorical_columns:

    if (test1[i].max()== train1[i].max())&(train1[i].max()<8):

            test1 = pd.get_dummies(test1, columns = [i])

            train1=pd.get_dummies(train1, columns = [i])
train1.shape, test1.shape
train_TransactionID= train1["TransactionID"]

test_TransactionID=test1["TransactionID"]

TransactionDT=train1["TransactionDT"]

X= train1.drop([ 'TransactionDT', 'TransactionID'], axis=1)

y = train1['isFraud']

test1 = test1.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

del train1
#X=X.drop(feature_drop,axis=1)

#test=test.drop(feature_drop,axis=1)
X=X.drop("isFraud", axis=1)
X.head(1)
y.head()
X.shape, test1.shape
params = {'num_leaves': 491,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47

          

         }
# datetime object containing current date and time



import datetime

datetime.datetime.now()

from time import time

time()
folds = TimeSeriesSplit(n_splits=5)



aucs = list()

feature_importances = pd.DataFrame()

feature_importances['feature'] = X.columns



training_start_time = time()

for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

    start_time = time()

    print('Training on fold {}'.format(fold + 1))

    

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)

    

    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()

    aucs.append(clf.best_score['valid_1']['auc'])

    

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

print('-' * 30)

print('Training has finished.')

print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))

print('Mean AUC:', np.mean(aucs))

print('-' * 30)
feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances_TimeFold.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));
#clf right now is the last model, trained with 80% of data and validated with 20%

best_iter = clf.best_iteration
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)

clf.fit(X, y)
#set the output as a dataframe and convert to csv file named submission.csv

predictions = clf.predict_proba(test1)[:, 1]

output = pd.DataFrame({ "TransactionID" : test_TransactionID, "isFraud": predictions })

output.to_csv('submission_lgbm.csv', index=False)
output
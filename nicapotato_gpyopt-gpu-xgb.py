# Latest Pandas version

!pip install -q 'pandas==0.25' --force-reinstall

# Install Gpyopt

!pip install GPyOpt
import xgboost as xgb

import pandas as pd

import GPyOpt

from GPyOpt.methods import BayesianOptimization

print("GPyOpt version:", GPyOpt.__version__)

print("LGBM version:", xgb.__version__)

print("Pandas version:", pd.__version__)
import time

notebookstart = time.time()



import os

from contextlib import contextmanager

import gc; gc.enable()

import pprint



import datetime

import csv

import random



import numpy as np

from pandas.io.json import json_normalize



# Viz

import matplotlib.pyplot as plt

import seaborn as sns



# Modeling

from sklearn.model_selection import KFold, train_test_split

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from scipy import interp

import itertools

import warnings



warnings.filterwarnings("ignore")



seed = 24

np.random.seed(seed)



pd.set_option('display.max_columns', 500)

pd.options.display.max_rows = 999

pd.set_option('max_colwidth', 500)
print("Define DF Schema..")



target_var = 'isFraud'



schema = {

    "TransactionDT":       "int32",

    "TransactionAmt":    "float32",

    "ProductCD":          "object",

    "card1":               "int16",

    "card2":             "float32",

    "card3":             "float32",

    "card4":              "object",

    "card5":             "float32",

    "card6":              "object",

    "addr1":             "float32",

    "addr2":             "float32",

    "dist1":             "float32",

    "dist2":             "float32",

    "P_emaildomain":      "object",

    "R_emaildomain":      "object",

    "C1":                "float32",

    "C2":                "float32",

    "C3":                "float32",

    "C4":                "float32",

    "C5":                "float32",

    "C6":                "float32",

    "C7":                "float32",

    "C8":                "float32",

    "C9":                "float32",

    "C10":               "float32",

    "C11":               "float32",

    "C12":               "float32",

    "C13":               "float32",

    "C14":               "float32",

    "D1":                "float32",

    "D2":                "float32",

    "D3":                "float32",

    "D4":                "float32",

    "D5":                "float32",

    "D6":                "float32",

    "D7":                "float32",

    "D8":                "float32",

    "D9":                "float32",

    "D10":               "float32",

    "D11":               "float32",

    "D12":               "float32",

    "D13":               "float32",

    "D14":               "float32",

    "D15":               "float32",

    "M1":                 "object",

    "M2":                 "object",

    "M3":                 "object",

    "M4":                 "object",

    "M5":                 "object",

    "M6":                 "object",

    "M7":                 "object",

    "M8":                 "object",

    "M9":                 "object",

    "V1":                "float32",

    "V2":                "float32",

    "V3":                "float32",

    "V4":                "float32",

    "V5":                "float32",

    "V6":                "float32",

    "V7":                "float32",

    "V8":                "float32",

    "V9":                "float32",

    "V10":               "float32",

    "V11":               "float32",

    "V12":               "float32",

    "V13":               "float32",

    "V14":               "float32",

    "V15":               "float32",

    "V16":               "float32",

    "V17":               "float32",

    "V18":               "float32",

    "V19":               "float32",

    "V20":               "float32",

    "V21":               "float32",

    "V22":               "float32",

    "V23":               "float32",

    "V24":               "float32",

    "V25":               "float32",

    "V26":               "float32",

    "V27":               "float32",

    "V28":               "float32",

    "V29":               "float32",

    "V30":               "float32",

    "V31":               "float32",

    "V32":               "float32",

    "V33":               "float32",

    "V34":               "float32",

    "V35":               "float32",

    "V36":               "float32",

    "V37":               "float32",

    "V38":               "float32",

    "V39":               "float32",

    "V40":               "float32",

    "V41":               "float32",

    "V42":               "float32",

    "V43":               "float32",

    "V44":               "float32",

    "V45":               "float32",

    "V46":               "float32",

    "V47":               "float32",

    "V48":               "float32",

    "V49":               "float32",

    "V50":               "float32",

    "V51":               "float32",

    "V52":               "float32",

    "V53":               "float32",

    "V54":               "float32",

    "V55":               "float32",

    "V56":               "float32",

    "V57":               "float32",

    "V58":               "float32",

    "V59":               "float32",

    "V60":               "float32",

    "V61":               "float32",

    "V62":               "float32",

    "V63":               "float32",

    "V64":               "float32",

    "V65":               "float32",

    "V66":               "float32",

    "V67":               "float32",

    "V68":               "float32",

    "V69":               "float32",

    "V70":               "float32",

    "V71":               "float32",

    "V72":               "float32",

    "V73":               "float32",

    "V74":               "float32",

    "V75":               "float32",

    "V76":               "float32",

    "V77":               "float32",

    "V78":               "float32",

    "V79":               "float32",

    "V80":               "float32",

    "V81":               "float32",

    "V82":               "float32",

    "V83":               "float32",

    "V84":               "float32",

    "V85":               "float32",

    "V86":               "float32",

    "V87":               "float32",

    "V88":               "float32",

    "V89":               "float32",

    "V90":               "float32",

    "V91":               "float32",

    "V92":               "float32",

    "V93":               "float32",

    "V94":               "float32",

    "V95":               "float32",

    "V96":               "float32",

    "V97":               "float32",

    "V98":               "float32",

    "V99":               "float32",

    "V100":              "float32",

    "V101":              "float32",

    "V102":              "float32",

    "V103":              "float32",

    "V104":              "float32",

    "V105":              "float32",

    "V106":              "float32",

    "V107":              "float32",

    "V108":              "float32",

    "V109":              "float32",

    "V110":              "float32",

    "V111":              "float32",

    "V112":              "float32",

    "V113":              "float32",

    "V114":              "float32",

    "V115":              "float32",

    "V116":              "float32",

    "V117":              "float32",

    "V118":              "float32",

    "V119":              "float32",

    "V120":              "float32",

    "V121":              "float32",

    "V122":              "float32",

    "V123":              "float32",

    "V124":              "float32",

    "V125":              "float32",

    "V126":              "float32",

    "V127":              "float32",

    "V128":              "float32",

    "V129":              "float32",

    "V130":              "float32",

    "V131":              "float32",

    "V132":              "float32",

    "V133":              "float32",

    "V134":              "float32",

    "V135":              "float32",

    "V136":              "float32",

    "V137":              "float32",

    "V138":              "float32",

    "V139":              "float32",

    "V140":              "float32",

    "V141":              "float32",

    "V142":              "float32",

    "V143":              "float32",

    "V144":              "float32",

    "V145":              "float32",

    "V146":              "float32",

    "V147":              "float32",

    "V148":              "float32",

    "V149":              "float32",

    "V150":              "float32",

    "V151":              "float32",

    "V152":              "float32",

    "V153":              "float32",

    "V154":              "float32",

    "V155":              "float32",

    "V156":              "float32",

    "V157":              "float32",

    "V158":              "float32",

    "V159":              "float32",

    "V160":              "float32",

    "V161":              "float32",

    "V162":              "float32",

    "V163":              "float32",

    "V164":              "float32",

    "V165":              "float32",

    "V166":              "float32",

    "V167":              "float32",

    "V168":              "float32",

    "V169":              "float32",

    "V170":              "float32",

    "V171":              "float32",

    "V172":              "float32",

    "V173":              "float32",

    "V174":              "float32",

    "V175":              "float32",

    "V176":              "float32",

    "V177":              "float32",

    "V178":              "float32",

    "V179":              "float32",

    "V180":              "float32",

    "V181":              "float32",

    "V182":              "float32",

    "V183":              "float32",

    "V184":              "float32",

    "V185":              "float32",

    "V186":              "float32",

    "V187":              "float32",

    "V188":              "float32",

    "V189":              "float32",

    "V190":              "float32",

    "V191":              "float32",

    "V192":              "float32",

    "V193":              "float32",

    "V194":              "float32",

    "V195":              "float32",

    "V196":              "float32",

    "V197":              "float32",

    "V198":              "float32",

    "V199":              "float32",

    "V200":              "float32",

    "V201":              "float32",

    "V202":              "float32",

    "V203":              "float32",

    "V204":              "float32",

    "V205":              "float32",

    "V206":              "float32",

    "V207":              "float32",

    "V208":              "float32",

    "V209":              "float32",

    "V210":              "float32",

    "V211":              "float32",

    "V212":              "float32",

    "V213":              "float32",

    "V214":              "float32",

    "V215":              "float32",

    "V216":              "float32",

    "V217":              "float32",

    "V218":              "float32",

    "V219":              "float32",

    "V220":              "float32",

    "V221":              "float32",

    "V222":              "float32",

    "V223":              "float32",

    "V224":              "float32",

    "V225":              "float32",

    "V226":              "float32",

    "V227":              "float32",

    "V228":              "float32",

    "V229":              "float32",

    "V230":              "float32",

    "V231":              "float32",

    "V232":              "float32",

    "V233":              "float32",

    "V234":              "float32",

    "V235":              "float32",

    "V236":              "float32",

    "V237":              "float32",

    "V238":              "float32",

    "V239":              "float32",

    "V240":              "float32",

    "V241":              "float32",

    "V242":              "float32",

    "V243":              "float32",

    "V244":              "float32",

    "V245":              "float32",

    "V246":              "float32",

    "V247":              "float32",

    "V248":              "float32",

    "V249":              "float32",

    "V250":              "float32",

    "V251":              "float32",

    "V252":              "float32",

    "V253":              "float32",

    "V254":              "float32",

    "V255":              "float32",

    "V256":              "float32",

    "V257":              "float32",

    "V258":              "float32",

    "V259":              "float32",

    "V260":              "float32",

    "V261":              "float32",

    "V262":              "float32",

    "V263":              "float32",

    "V264":              "float32",

    "V265":              "float32",

    "V266":              "float32",

    "V267":              "float32",

    "V268":              "float32",

    "V269":              "float32",

    "V270":              "float32",

    "V271":              "float32",

    "V272":              "float32",

    "V273":              "float32",

    "V274":              "float32",

    "V275":              "float32",

    "V276":              "float32",

    "V277":              "float32",

    "V278":              "float32",

    "V279":              "float32",

    "V280":              "float32",

    "V281":              "float32",

    "V282":              "float32",

    "V283":              "float32",

    "V284":              "float32",

    "V285":              "float32",

    "V286":              "float32",

    "V287":              "float32",

    "V288":              "float32",

    "V289":              "float32",

    "V290":              "float32",

    "V291":              "float32",

    "V292":              "float32",

    "V293":              "float32",

    "V294":              "float32",

    "V295":              "float32",

    "V296":              "float32",

    "V297":              "float32",

    "V298":              "float32",

    "V299":              "float32",

    "V300":              "float32",

    "V301":              "float32",

    "V302":              "float32",

    "V303":              "float32",

    "V304":              "float32",

    "V305":              "float32",

    "V306":              "float32",

    "V307":              "float32",

    "V308":              "float32",

    "V309":              "float32",

    "V310":              "float32",

    "V311":              "float32",

    "V312":              "float32",

    "V313":              "float32",

    "V314":              "float32",

    "V315":              "float32",

    "V316":              "float32",

    "V317":              "float32",

    "V318":              "float32",

    "V319":              "float32",

    "V320":              "float32",

    "V321":              "float32",

    "V322":              "float32",

    "V323":              "float32",

    "V324":              "float32",

    "V325":              "float32",

    "V326":              "float32",

    "V327":              "float32",

    "V328":              "float32",

    "V329":              "float32",

    "V330":              "float32",

    "V331":              "float32",

    "V332":              "float32",

    "V333":              "float32",

    "V334":              "float32",

    "V335":              "float32",

    "V336":              "float32",

    "V337":              "float32",

    "V338":              "float32",

    "V339":              "float32",

    "id_01":             "float32",

    "id_02":             "float32",

    "id_03":             "float32",

    "id_04":             "float32",

    "id_05":             "float32",

    "id_06":             "float32",

    "id_07":             "float32",

    "id_08":             "float32",

    "id_09":             "float32",

    "id_10":             "float32",

    "id_11":             "float32",

    "id_12":              "object",

    "id_13":             "float32",

    "id_14":             "float32",

    "id_15":              "object",

    "id_16":              "object",

    "id_17":             "float32",

    "id_18":             "float32",

    "id_19":             "float32",

    "id_20":             "float32",

    "id_21":             "float32",

    "id_22":             "float32",

    "id_23":              "object",

    "id_24":             "float32",

    "id_25":             "float32",

    "id_26":             "float32",

    "id_27":              "object",

    "id_28":              "object",

    "id_29":              "object",

    "id_30":              "object",

    "id_31":              "object",

    "id_32":             "float32",

    "id_33":              "object",

    "id_34":              "object",

    "id_35":              "object",

    "id_36":              "object",

    "id_37":              "object",

    "id_38":              "object",

    "DeviceType":         "object",

    "DeviceInfo":         "object",

    "is_fraud":			  "int8"

}



emails = {'gmail': 'google',

'att.net': 'att',

'twc.com': 'spectrum',

'scranton.edu': 'other',

'optonline.net': 'other',

'hotmail.co.uk': 'microsoft',

'comcast.net': 'other',

'yahoo.com.mx': 'yahoo',

'yahoo.fr': 'yahoo',

'yahoo.es': 'yahoo',

'charter.net': 'spectrum',

'live.com': 'microsoft',

'aim.com': 'aol',

'hotmail.de': 'microsoft',

'centurylink.net': 'centurylink',

'gmail.com': 'google',

'me.com': 'apple',

'earthlink.net': 'other',

'gmx.de': 'other',

'web.de': 'other',

'cfl.rr.com': 'other',

'hotmail.com': 'microsoft',

'protonmail.com': 'other',

'hotmail.fr': 'microsoft',

'windstream.net': 'other',

'outlook.es': 'microsoft',

'yahoo.co.jp': 'yahoo',

'yahoo.de': 'yahoo',

'servicios-ta.com': 'other',

'netzero.net': 'other',

'suddenlink.net': 'other',

'roadrunner.com': 'other',

'sc.rr.com': 'other',

'live.fr': 'microsoft',

'verizon.net': 'yahoo',

'msn.com': 'microsoft',

'q.com': 'centurylink',

'prodigy.net.mx': 'att',

'frontier.com': 'yahoo',

'anonymous.com': 'other',

'rocketmail.com': 'yahoo',

'sbcglobal.net': 'att',

'frontiernet.net': 'yahoo',

'ymail.com': 'yahoo',

'outlook.com': 'microsoft',

'mail.com': 'other',

'bellsouth.net': 'other',

'embarqmail.com': 'centurylink',

'cableone.net': 'other',

'hotmail.es': 'microsoft',

'mac.com': 'apple',

'yahoo.co.uk': 'yahoo',

'netzero.com': 'other',

'yahoo.com': 'yahoo',

'live.com.mx': 'microsoft',

'ptd.net': 'other',

'cox.net': 'other',

'aol.com': 'aol',

'juno.com': 'other',

'icloud.com': 'apple'}





us_emails = ['gmail', 'net', 'edu']
@contextmanager

def timer(name):

    """

    Time Each Process

    """

    t0 = time.time()

    yield

    print('\n[{}] done in {} Minutes\n'.format(name, round((time.time() - t0)/60,2)))



# Device Features

def id_split(dataframe):

    # https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu

    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]

    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]



    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]

    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]



    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]

    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]



    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]

    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]



    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]

    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]



    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    dataframe['had_id'] = 1

    gc.collect()

    

    return dataframe

    

def fraud_preprocessing(debug = None):

    print("Starting Pre-Processing..")

    with timer("Load Tables"):

        train_transaction = pd.read_csv('../input/train_transaction.csv',

                                        index_col='TransactionID', nrows= debug, dtype = schema)

        test_transaction = pd.read_csv('../input/test_transaction.csv',

                                       index_col='TransactionID', nrows= debug, dtype = schema)



        train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

        test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



        sample_submission = pd.read_csv('../input/sample_submission.csv',

                                        index_col='TransactionID',

                                        nrows= debug)



    with timer("Merge Tables"):

        train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

        test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



        print("Train Shape: {} Rows, {} Columns".format(*train.shape))

        print("Test Shape: {} Rows, {} Columns".format(*test.shape))



        y = train[target_var].copy()

        del train_transaction, train_identity, test_transaction, test_identity



        traindex = train.index

        testdex = test.index

        

    with timer("Train/Test Split Feature Engineering"):

        # Credit https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-w-gpu

        train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')

        train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')

        train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')

        train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')



        test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')

        test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

        test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')

        test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')



        train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

        train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

        train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

        train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



        test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

        test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

        test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

        test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



        train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')

        train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

        train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')

        train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



        test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')

        test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

        test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')

        test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')



        train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')

        train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

        train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')

        train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



        test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

        test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

        test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')

        test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

        

        # New feature - log of transaction amount. ()

        train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])

        test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

        

        # Encoding - count encoding for both train and test

        for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:

            train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

            test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))



        # Encoding - count encoding separately for train and test

        for feature in ['id_01', 'id_31', 'id_33', 'id_36']:

            train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

            test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))

            

        # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499

        for c in ['P_emaildomain', 'R_emaildomain']:

            train[c + '_bin'] = train[c].map(emails)

            test[c + '_bin'] = test[c].map(emails)



            train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

            test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])



            train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

            test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

            

        # Extract Device Information

        train = id_split(train)

        test = id_split(test)

        

        # Combine

        df = pd.concat([train.drop(target_var,axis=1),test],axis = 0)

        del train, test

        

    with timer("Whole Feature Engineering"):

        START_DATE = '2017-12-01'

        startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')    

        df = df.assign(

                # New feature - decimal part of the transaction amount

                TransactionAmt_decimal = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int),



                # Count encoding for card1 feature. 

                # Explained in this kernel: https://www.kaggle.com/nroman/eda-for-cis-fraud-detection

                card1_count_full = df['card1'].map(df['card1'].value_counts(dropna=False)),



                # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

                Transaction_day_of_week = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7),

                Transaction_hour = np.floor(df['TransactionDT'] / 3600) % 24,



                TransactionDT = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x))),

            )

        df = df.assign(

                # Time of Day

                year = df['TransactionDT'].dt.year,

                month = df['TransactionDT'].dt.month,

                dow = df['TransactionDT'].dt.dayofweek,

                quarter = df['TransactionDT'].dt.quarter,

                hour = df['TransactionDT'].dt.hour,

                day = df['TransactionDT'].dt.day,

        

                # All NaN

                all_group_nan_sum = df.isnull().sum(axis=1) / df.shape[1],

                all_group_0_count = (df == 0).astype(int).sum(axis=1) / (df.shape[1] - df.isnull().sum(axis=1))

        )

        

        # Create Features based on anonymised prefix groups

        prefix = ['C','D','Device','M','Transaction','V','addr','card','dist','id']

        for i, p in enumerate(prefix):

            column_set = [x for x in df.columns.tolist() if x.startswith(prefix[i])]



            # Take NA count

            df[p + "group_nan_sum"] = df[column_set].isnull().sum(axis=1) / df[column_set].shape[1]



            # Take SUM/Mean if numeric

            numeric_cols = [x for x in column_set if df[x].dtype != object]

            if numeric_cols:

                df[p + "group_sum"] = df[column_set].sum(axis=1)

                df[p + "group_mean"] = df[column_set].mean(axis=1)

                # Zero Count

                df[p + "group_0_count"] = (df[column_set] == 0).astype(int).sum(axis=1) / (df[column_set].shape[1] - df[p + "group_nan_sum"])



    with timer("Label Encode"):

        categorical_cols = []

        # Label Encoding

        for f in df.columns:

            if df[f].dtype=='object': 

                categorical_cols += [f]

                lbl = preprocessing.LabelEncoder()

                df[f] = lbl.fit_transform(df[f].astype(str))

    print("Total Shape: {} Rows, {} Columns".format(*df.shape))

    return df, y, traindex, testdex, categorical_cols, sample_submission
DEBUG = None # None for no debug, else number of rows



df, y, traindex, testdex, cat_cols, sample_submission = fraud_preprocessing(debug = DEBUG)

df.fillna(-999, inplace=True)

X = df.loc[traindex,:]

feature_subset = [x for x in X.columns.tolist() if x not in ['TransactionDT','Fraud', 'traintest', 'yrmth']]

test = df.loc[testdex,:]

# xgtest = xgb.DMatrix(test.loc[:,feature_subset], feature_names = feature_subset)



del df; gc.collect();
print("None Fraud: {}%, Fraud: {}%".format(*y.value_counts(normalize=True)))

print("Randomness Score AUC: {}".format(

    metrics.roc_auc_score(y,np.array([y.value_counts(normalize=True)[0]]*y.shape[0]))))
# Optimization objective 

def XGB_score(para):

    parameters = para[0]

#     num_leaves = 2**parameters[0] if 2**parameters[0] < 4095 else 4095

#     parameters[0] = -1 if parameters[0] == 45 else parameters[0]

#     num_leaves = int(num_leaves * parameters[1])

    params = {

        # Static Variables

        'objective': 'binary:logistic',

        'eval_metric': metric,

        'learning_rate': 0.1, # Multiplication performed on each boosting iteration.

        'tree_method': 'gpu_hist', # GPU usage.

        'importance_type': 'weight',

        'missing': -999,



        # Dynamic Variables

        # https://xgboost.readthedocs.io/en/latest/parameter.html

        # https://sites.google.com/view/lauraepp/parameters



        # Bushi-ness Parameters

        'max_depth': int(parameters[0]),  # -1 means no tree depth limit

#         'num_leaves': int(parameters[1]), # we should let it be smaller than 2^(max_depth)



        # Tree Depth Regularization

#         'subsample_for_bin': int(parameters[2]), # Number of samples for constructing bin

#         'min_data_in_leaf': int(parameters[0]), # Minimum number of data need in a child(min_data_in_leaf) - Must be motified when using a smaller dataset

    #     'min_gain_to_split': [0], # Prune by minimum loss requirement.

#         'min_sum_hessian_in_leaf': parameters[3], # Prune by minimum hessian requirement - Minimum sum of instance weight(hessian) needed in a child(leaf)



        # Regularization L1/L2

        'reg_alpha': parameters[1], # L1 regularization term on weights (0 is no regular)

        'reg_lambda': parameters[2], # L2 regularization term on weights

    #     'max_bin': list(range(70, 300, 30)),  # Number of bucketed bin for feature values



        # Row/Column Sampling

        'subsample': parameters[3], # Subsample ratio of the training instance.

    #     'subsample_freq': 0, # frequence of subsample, <=0 means no enable

#         'bagging_fraction': 1.0,# Percentage of rows used per iteration frequency.

#         'bagging_freq': 5,# Iteration frequency to update the selected rows.

        'colsample_bytree': parameters[4], # Percentage of columns used per iteration.

        

#         'nthread': -1, # Multi-threading

#         'verbose': -1, # Logging Iteration Progression

        'seed': seed # Seed for row sampling RNG.

    }

    

    experiment_clf = xgb.XGBClassifier(**params,

                                  n_estimators = n_estimators)

    

    modelstart= time.time()

    experiment_clf.fit(X_train,

                       y_train,

                       eval_set=[(X_valid,y_valid)],

                       early_stopping_rounds=100,

                       verbose=0)

    runtime = (time.time() - modelstart)/60



    rounds = experiment_clf.best_ntree_limit

    val_pred = experiment_clf.predict_proba(X_valid)[:,1]



    # Get Metrics

    score = experiment_clf.best_score

    loss = metrics.log_loss(y_valid, val_pred)

    params['num_boost_round'] = rounds



    gpyopt_output.append(

        [

         loss,

         score,

         rounds,

         params,

         runtime,

         experiment_clf.feature_importances_

        ]

    )

    

    return score
split_size = 0.4

n_estimators = 10000

metric = 'auc'

ESR = 175



X_train, X_valid, y_train, y_valid = train_test_split(

    X[feature_subset], y, test_size=split_size,

    random_state=seed, shuffle=True,stratify=y)



bds = [ {'name': 'max_depth', 'type': 'discrete', 'domain': (5, 15)},

        {'name': 'reg_alpha', 'type': 'continuous', 'domain': (.05, 0.8)},

        {'name': 'reg_lambda', 'type': 'continuous', 'domain': (.05, 0.8)},

        {'name': 'subsample', 'type': 'continuous', 'domain': (0.1, 0.80)},

        {'name': 'colsample_bytree', 'type': 'continuous', 'domain': (0.1, 0.80)},

#         {'name': 'num_leaves', 'type': 'discrete', 'domain': (200, 4000)},

#         {'name': 'subsample_for_bin', 'type': 'discrete', 'domain': (1000, 5000)},

#         {'name': 'min_sum_hessian_in_leaf', 'type': 'continuous', 'domain': (0.1, 15)},

      ]



sb_cat_cols = [x for x in cat_cols if x in feature_subset]





max_iter = 30

initial_iter = 5



gpyopt_output = []

gpy_importance = pd.DataFrame()

optimizer = BayesianOptimization(f=XGB_score, 

                                 domain=bds,

                                 model_type='GP',

                                 optimize_restarts = 1,

                                 initial_design_numdata = initial_iter,

                                 acquisition_type ='EI',

                                 acquisition_jitter = 0.05,

                                 exact_feval=True, 

                                 maximize=True)



with timer("Bayesian Optimisation - {} Iterations".format(max_iter + initial_iter)):

    optimizer.run_optimization(max_iter=max_iter)

    

# Output

results = pd.DataFrame(gpyopt_output,

        columns = ['logloss','valid_auc',

                   'boosting_rounds','parameters', 'runtime', 'imp']

                      )

gpyimp = results.imp.apply(lambda x: pd.Series(x)).T

gpyimp.index = feature_subset

og_cols = gpyimp.columns

gpyimp['mean'] = gpyimp[og_cols].mean(axis = 1)

gpyimp['std'] = gpyimp[og_cols].std(axis = 1)



del gpyopt_output; results.drop('imp', axis =1, inplace=True)

results.to_csv("gpyopt_iterations_output.csv")

best_params = results['parameters'].iloc[np.argmax(results.valid_auc)]



# Visualize Convergence

optimizer.plot_convergence()



# Trees and Runtime

results['TPM'] = results['boosting_rounds'] / results['runtime'] # tree_per_minutes



plt.plot(results['TPM'], '-rx')

plt.xlabel("Bayesian Search Iteration")

plt.ylabel("TPM")

plt.title("Tree per Minute through Bayesian Search")

plt.show()



print("Best AUC: {}".format(optimizer.fx_opt))

print("Best Parameters")

pprint.pprint(best_params)



# Json to DataFrame

results = pd.concat([results.drop('parameters',axis=1).reset_index(drop=True),

                     json_normalize(results['parameters']).reset_index(drop=True)

                    ], axis = 1)



del X_train, X_valid, y_train, y_valid
importance_cutoff = .95



# Decide which features to preserve

cumu_imp = np.cumsum(gpyimp.sort_values(by="mean", ascending=False)['mean'])

keep_features = cumu_imp[cumu_imp < importance_cutoff].index[:200]



# Plot Them

f, ax = plt.subplots(1,2, figsize = [10,5])

sns.distplot(gpyimp['mean'], ax =ax[0])

ax[0].set_title("Feature Importance Distribution")

ax[0].set_xlabel("Importance")



ax[1].plot(cumu_imp.reset_index(drop=True), color = 'r')

ax[1].set_title("Cumulative Feature Importance")

ax[1].set_xlabel("Number of Features")

ax[1].axvline(len(keep_features), color = 'black')



plt.show()
display(results.head())
t_r,t_c = 2, 3

f, axes = plt.subplots(t_r, t_c, figsize = [15,8],

                       sharex=False, sharey=True)

row,col = 0,0

paras = [x['name'] for x in bds] 



for var in paras:

    if col == t_c:

        col = 0

        row += 1

    sns.regplot(x=var, y = "valid_auc", data = results,

                x_estimator=np.mean, logx=True,

                truncate=True, ax = axes[row,col])

    axes[row,col].set_title('{} vs AUC'.format(var.title()))

    axes[row,col].grid(True, lw = 2, ls = '--', c = '.75')

    axes[row,col].set_ylim(0.8,1)

    if var == 'min_data_in_leaf':

        axes[row,col].set_xlim(0,100)

    col+=1

plt.tight_layout(pad=0)

plt.show()
del results
feature_subset = keep_features

drop_cols = [x for x in X.columns if x not in keep_features]



X.drop(drop_cols, axis = 1, inplace= True)

test.drop(drop_cols, axis = 1, inplace= True)

gc.collect()

print("Train Shape: {} Rows, {} Cols\n".format(*X[feature_subset].shape))



allmodelstart= time.time()

EPOCHS = 3

best_params['learning_rate'] = 0.1

kf = KFold(n_splits = EPOCHS, shuffle = True)

y_preds = np.zeros(sample_submission.shape[0])

y_oof = np.zeros(X.shape[0])

f,ax = plt.subplots(1,3,figsize = [15,6])

sb_cat_cols = [x for x in cat_cols if x in feature_subset]

all_feature_importance_df  = pd.DataFrame()



# Vincent Lugat - https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt

mean_fpr = np.linspace(0,1,100)

cms, tprs, aucs, y_real, y_proba,recalls, roc_aucs,f1_scores, accuracies, precisions = [],[],[],[],[],[],[],[],[],[]



# Run Out of Fold

for i, (tr_idx, val_idx) in enumerate(kf.split(X, y)):

    i += 1

    modelstart = time.time()

    sb_cat_cols = [x for x in cat_cols if x in feature_subset]        

    clf = xgb.XGBClassifier(**best_params,

                    n_estimators = n_estimators)

    

    modelstart= time.time()

    clf.fit(X.iloc[tr_idx, :][feature_subset],

               y.iloc[tr_idx],

               eval_set=[(X.iloc[val_idx, :][feature_subset], y.iloc[val_idx])],

               early_stopping_rounds=ESR,

               verbose=500)

    runtime = (time.time() - modelstart)/60

    rounds = clf.best_ntree_limit

    

    # Model Evaluation

    y_oof[val_idx] = clf.predict_proba(X.iloc[val_idx, :][feature_subset])[:,1]

    y_preds += clf.predict_proba(test[feature_subset])[:,1] / EPOCHS

    

    # Convergence

    label = 'valid_{}'.format(i)

    ax[0].plot(clf.evals_result_['validation_0']['auc'], label = label)

    

    # Feature Importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = feature_subset

    fold_importance_df["importance"] = clf.feature_importances_

    all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df.sort_values(by = 'feature')], axis=0)

    print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

    

    del clf

    

    # Scores 

    roc_aucs.append(roc_auc_score(y.iloc[val_idx].values,y_oof[val_idx]))

    accuracies.append(accuracy_score(y.iloc[val_idx].values,y_oof[val_idx].round()))

    recalls.append(recall_score(y.iloc[val_idx].values,y_oof[val_idx].round()))

    precisions.append(precision_score(y.iloc[val_idx].values,y_oof[val_idx].round()))

    f1_scores.append(f1_score(y.iloc[val_idx].values,y_oof[val_idx].round()))

    

    # Roc curve by folds

    fpr, tpr, t = roc_curve(y.iloc[val_idx].values,y_oof[val_idx])

    tprs.append(interp(mean_fpr, fpr, tpr))

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    ax[1].plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))

    

    # Precion recall by folds

    precision, recall, _ = precision_recall_curve(y.iloc[val_idx].values,y_oof[val_idx])

    y_real.append(y.iloc[val_idx].values)

    y_proba.append(y_oof[val_idx])

    ax[2].plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))

    

    # Confusion matrix by folds

    cms.append(confusion_matrix(y.iloc[val_idx].values,y_oof[val_idx].round()))



# Convergence

ax[0].set_title("GPU XGB Metric Convergence over {} Folds".format(EPOCHS)) 

ax[0].set_ylabel("AUC")

ax[0].set_xlabel("Boosting Rounds")



#ROC 

ax[1].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

mean_tpr = np.mean(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

ax[1].plot(mean_fpr, mean_tpr, color='blue',

         label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)



ax[1].set_xlabel('False Positive Rate')

ax[1].set_ylabel('True Positive Rate')

ax[1].set_title('LGB ROC curve by folds')

ax[1].legend(loc="lower right")



# PR plt

ax[2].plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')

y_real = np.concatenate(y_real)

y_proba = np.concatenate(y_proba)

precision, recall, _ = precision_recall_curve(y_real, y_proba)

ax[2].plot(recall, precision, color='blue',

         label=r'Mean P|R')

ax[2].set_xlabel('Recall')

ax[2].set_ylabel('Precision')

ax[2].set_title('P|R curve by folds')

ax[2].legend(loc="lower left")



plt.tight_layout(pad=0)

plt.savefig('model_eval.png')

plt.show()



# Metrics

print(

'CV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),

'\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),

'\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),

'\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),

'\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))

)
# Plot Importance

cols = all_feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(

    by="importance", ascending=False)[:50].index

best_features = all_feature_importance_df.loc[all_feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(8,10))

sns.barplot(x="importance", y="feature", 

            data=best_features.sort_values(by="importance", ascending=False))

plt.title('XGB Weight (split) Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')

print("All Model Runtime: %0.2f Minutes"%((time.time() - allmodelstart)/60))



# Confusion matrix 

def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix"',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')

 

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Confusion maxtrix & metrics

plt.rcParams["axes.grid"] = False



cm = np.average(cms, axis=0).round(1)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes=class_names, 

                      title= 'XGB Confusion matrix [averaged/folds]')

plt.show()



cm = np.std(cms, axis=0).round(2)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes=class_names, 

                      title= 'XGB Confusion matrix [STD/folds]')

plt.show()
cols = ['card1_count_full', 'card1','card2','card2_count_full']

plot_df = pd.concat([X.loc[:,cols], y], axis =1 )



t_r,t_c = 2, 2

f, axes = plt.subplots(t_r,t_c, figsize = [12,8],sharex=False, sharey=False)

row,col = 0,0

for c in cols:

    if col == t_c:

        col = 0

        row += 1

    sns.kdeplot(plot_df.loc[plot_df.isFraud == 0, c], shade = True, alpha = 0.6, color = 'black', ax = axes[row,col], label = 'Not Fraud')

    sns.kdeplot(plot_df.loc[plot_df.isFraud == 1, c], shade = True, alpha = 0.6, color = 'lime', ax = axes[row,col], label = 'Fraud')

    axes[row,col].set_title('{} and Fraud Distribution'.format(c.title()))

    col+=1

    

plt.tight_layout(pad=0)

plt.show()

del plot_df
# When doing feature selection, make sure you use the same subset on test set.

# LGBM will not break, but it will give you broken predictions.. -_-

assert X[feature_subset].shape[1] == test[feature_subset].shape[1]



sample_submission['isFraud'] = y_preds

sample_submission.to_csv('{}_feats_{}fold_lgbm_gpu.csv'.format(len(feature_subset),EPOCHS))
print("Notebook Runtime: %0.2f Hours"%((time.time() - notebookstart)/60/60))
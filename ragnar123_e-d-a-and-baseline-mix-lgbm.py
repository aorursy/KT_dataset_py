import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

pd.set_option('display.max_columns', 1000)

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import lightgbm as lgb

import gc

import datetime

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sns.set()

%matplotlib inline
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col = 'TransactionID')

print('Successfully loaded train_identity')



train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col = 'TransactionID')

print('Successfully loaded train_transaction')



test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col = 'TransactionID')

print('Successfully loaded test_identity')



test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col = 'TransactionID')

print('Successfully loaded test_transaction')



sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

print('Successfully loaded sample_submisssion')



print('Data was successfully loades!')



print('Merging data....')

train = train_transaction.merge(train_identity, how = 'left', left_index = True, right_index = True)

test = test_transaction.merge(test_identity, how = 'left', left_index = True, right_index = True)



print('Data was successfully merged!')



del train_identity, train_transaction, test_identity, test_transaction



print('Train dataset has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

print('Test dataset has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
# target variable



train['TransactionAmt'] = train['TransactionAmt'].astype(float)

total = len(train)

total_amt = train.groupby(['isFraud'])['TransactionAmt'].sum().sum()

plt.figure(figsize=(12,5))



plt.subplot(121)

plot_tr = sns.countplot(x='isFraud', data=train)

plot_tr.set_title("Fraud Transactions Distribution \n 0: No Fraud | 1: Fraud", fontsize=18)

plot_tr.set_xlabel("Is fraud?", fontsize=16)

plot_tr.set_ylabel('Count', fontsize=16)

for p in plot_tr.patches:

    height = p.get_height()

    plot_tr.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=15) 

    

percent_amt = (train.groupby(['isFraud'])['TransactionAmt'].sum())

percent_amt = percent_amt.reset_index()

plt.subplot(122)

plot_tr_2 = sns.barplot(x='isFraud', y='TransactionAmt',  dodge=True, data=percent_amt)

plot_tr_2.set_title("% Total Amount in Transaction Amt \n 0: No Fraud | 1: Fraud", fontsize=18)

plot_tr_2.set_xlabel("Is fraud?", fontsize=16)

plot_tr_2.set_ylabel('Total Transaction Amount Scalar', fontsize=16)

for p in plot_tr_2.patches:

    height = p.get_height()

    plot_tr_2.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total_amt * 100),

            ha="center", fontsize=15)
def missing_values(df):

    df1 = pd.DataFrame(df.isnull().sum()).reset_index()

    df1.columns = ['features', 'freq']

    df1['percentage'] = df1['freq']/df.shape[0]

    df1.sort_values('percentage', ascending = False, inplace = True)

    return df1



missing_train = missing_values(train)

missing_train.columns = ['features', 'freq_tr', 'percentage_tr']

missing_train
missing_test = missing_values(test)

missing_test.columns = ['features', 'freq_te', 'percentage_te']

missing_test
missing = missing_train.merge(missing_test, on = 'features')

missing.head(10)
train['id_24'].value_counts(normalize = True, dropna = False)
test['id_24'].value_counts(normalize = True, dropna = False)
drop_features = []

drop_features.append('id_24')
for i in ['id_24', 'id_25', 'id_08', 'id_07', 'id_21', 'id_26', 'id_27', 'id_23', 'id_22']:

    drop_features.append(i)

drop_features
train['dist2'].value_counts(normalize = True, dropna = False).head()
test['dist2'].value_counts(normalize = True, dropna = False).head()
def plot_feature(train, test, feature, log = False):

    df1_0 = train[train['isFraud']==0]

    df1_1 = train[train['isFraud']==1]

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(13,9))

    if log == True:

        sns.kdeplot(np.log(df1_0[feature]), shade = True, label = 'Not Fraud', ax = ax1)

        sns.kdeplot(np.log(df1_1[feature]), shade = True, label = 'Fraud', ax = ax1)

    else:

        sns.kdeplot(df1_0[feature], shade = True, label = 'Not Fraud', ax = ax1)

        sns.kdeplot(df1_1[feature], shade = True, label = 'Fraud', ax = ax1)

        

    

    if log == True:

        sns.kdeplot(np.log(train[feature]), shade = True, label = 'Train', ax = ax2)

        sns.kdeplot(np.log(test[feature]), shade = True, label = 'Test', ax = ax2)

    else:

        sns.kdeplot(train[feature], shade = True, label = 'Train', ax = ax2)

        sns.kdeplot(test[feature], shade = True, label = 'Test', ax = ax2)

        

plot_feature(train, test, 'dist2', True)
check_features = ['dist2']
missing[missing['features']=='D7']
plot_feature(train, test, 'D7', True)
check_features.append('D7')
missing[missing['features']=='id_18']
plot_feature(train, test, 'id_18', True)
drop_features.append('id_18')
missing[missing['features']=='D13']
plot_feature(train, test, 'D13', True)
missing[missing['features']=='D14']
plot_feature(train, test, 'D14', True)
missing[missing['features']=='D12']
plot_feature(train, test, 'D12', True)
missing[missing['features']=='id_04']
def plot_c_feature(train, feature):

    tmp = pd.crosstab(train[feature], train['isFraud'], normalize = 'index') * 100

    tmp = tmp.reset_index()

    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(13,9))

    plot_1 = sns.countplot(x = feature, hue = 'isFraud', data = train)

    plt.legend(title = 'Fraud', loc = 'best', labels = ['No', 'Yes'])

    plot_1_1 = plot_1.twinx()

    plot_1_1 = sns.pointplot(x = feature, y = 'Fraud', data = tmp, color = 'black', 

                         order = list(tmp[feature].values), legend = False)

    plot_1_1.set_ylabel('% of Fraud Transactions', fontsize = 16)

    plot_1.set_ylabel("Count", fontsize=16)

    

    

plot_c_feature(train, 'id_04')
missing[missing['features']=='id_03']
plot_c_feature(train, 'id_03')
missing[missing['features']=='D6']
plot_feature(train, test, 'D6', True)
missing[missing['features']=='id_33']
plot_c_feature(train, 'id_33')
missing[missing['features']=='id_09']
plot_c_feature(train, 'id_09')
missing[missing['features']=='id_10']
plot_c_feature(train, 'id_10')
check_features.append('id_10')
missing[missing['features']=='D9']
plot_feature(train, test, 'D9', True)
missing[missing['features']=='D8']
plot_feature(train, test, 'D8', True)
missing[missing['features']=='id_30']
plot_c_feature(train, 'id_30')
missing[missing['features']=='id_32']
plot_c_feature(train, 'id_32')
missing[missing['features']=='id_34']
plot_c_feature(train, 'id_34')
drop_features.append('id_34')
missing[missing['features']=='id_14']
plot_c_feature(train, 'id_14')
missing[missing['features']=='V141']
plot_feature(train, test, 'V141', True)
drop_features.append('V141')
missing[missing['features']=='V157']
plot_feature(train, test, 'V157', True)
check_features.append('V157')
missing[missing['features']=='V162']
plot_feature(train, test, 'V162', True)
missing[missing['features']=='V161']
plot_feature(train, test, 'V161', True)
missing[missing['features']=='V158']
plot_feature(train, test, 'V158', True)
check_features.append('V158')
missing[missing['features']=='V156']
plot_feature(train, test, 'V156', True)
check_features.append('V156')
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(10,5,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(10,5,i)

        sns.kdeplot(np.log(df1[feature]), bw=0.5,label=label1)

        sns.kdeplot(np.log(df2[feature]), bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
V = ['V142', 'V155', 'V154', 'V140', 'V149', 'V148', 'V147', 'V146', 'V153', 'V163', 'V139', 'V138', 'V151', 'V152', 

     'V145','V144', 'V143', 'V160', 'V159', 'V164', 'V165', 'V166', 'V150', 'V337', 'V333', 'V336', 'V335', 'V334', 'V338', 

     'V339', 'V325', 'V332', 'V324', 'V330', 'V329', 'V328', 'V327', 'V326', 'V322', 'V323', 'V331', 'V278', 'V277', 'V252', 

     'V253', 'V254', 'V257', 'V258', 'V260', 'V243', 'V262', 'V263', 'V264', 'V249', 'V266', 'V267', 'V268', 'V269', 'V273', 

     'V274', 'V275', 'V276', 'V265', 'V261', 'V247', 'V246', 'V241', 'V240', 'V237', 'V236', 'V235', 'V233', 'V232', 

     'V231', 'V230', 'V229', 'V228', 'V226', 'V225', 'V224', 'V223', 'V219', 'V218', 'V217', 'V244', 'V248', 'V242', 'V211', 

     'V214', 'V213', 'V212', 'V196', 'V205', 'V183', 'V216', 'V206', 'V186', 'V187', 'V192', 'V207', 'V215', 'V182', 'V191',

     'V181', 'V167', 'V168', 'V199', 'V193', 'V172', 'V173', 'V202', 'V203', 'V176', 'V177', 'V178', 'V179', 'V204', 'V190',

     'V194', 'V201', 'V189', 'V188', 'V185', 'V184', 'V180', 'V175', 'V174', 'V171', 'V170', 'V169', 'V195', 'V200', 'V197', 

     'V198', 'V208', 'V210', 'V209', 'V272', 'V234', 'V222', 'V238', 'V239', 'V227', 'V251', 'V250', 'V271', 'V270', 'V221',

     'V220', 'V255', 'V256', 'V259', 'V245', 'V3', 'V1', 'V2', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V46', 'V42',

     'V44', 'V43', 'V47', 'V41', 'V40', 'V39', 'V38', 'V37', 'V36', 'V35', 'V52', 'V51', 'V50', 'V49', 'V48', 'V45', 'V88', 

     'V93', 'V85', 'V87', 'V84', 'V83', 'V82', 'V81', 'V90', 'V91', 'V92', 'V94', 'V80', 'V79', 'V78', 'V77', 'V76',

     'V75', 'V86', 'V53', 'V74', 'V73', 'V72', 'V66', 'V54', 'V67', 'V64', 'V63', 'V62', 'V61', 'V71', 'V69', 'V55',

     'V60', 'V59', 'V58', 'V57', 'V65', 'V56', 'V70', 'V22', 'V23', 'V24', 'V34', 'V33', 'V32', 'V31', 'V30', 'V29',

     'V26', 'V25', 'V15', 'V21', 'V14', 'V16', 'V17', 'V18', 'V19', 'V12', 'V20', 'V13', 'V282', 'V301', 'V300', 'V296',

     'V289', 'V288', 'V283', 'V281', 'V315', 'V314', 'V313', 'V114', 'V110', 'V105', 'V104', 'V103', 'V102', 'V101', 'V100',

     'V95', 'V99', 'V98', 'V97', 'V107', 'V111', 'V96', 'V112', 'V106', 'V113', 'V137', 'V136', 'V108', 'V135', 'V134', 'V133', 

     'V132', 'V131', 'V130', 'V129', 'V128', 'V127', 'V126', 'V125', 'V124', 'V123', 'V122', 'V121', 'V120', 'V119', 'V118',

     'V117', 'V116', 'V115', 'V109', 'V312', 'V321', 'V294', 'V306', 'V305', 'V304', 'V303', 'V302', 'V299', 'V298', 'V297', 

     'V295', 'V293', 'V308', 'V292', 'V291', 'V290', 'V287', 'V286', 'V285', 'V284', 'V280', 'V279', 'V320', 'V307', 'V309',

     'V316', 'V310', 'V318', 'V317', 'V319', 'V311']
t0 = train[train['isFraud']==0]

t1 = train[train['isFraud']==1]

first = V[0:50]

plot_feature_distribution(t0, t1, '0', '1', first)
V1drop = ['V142', 'V146', 'V138', 'V151', 'V152', 'V333', 'V338', 'V339', 'V325', 'V332', 'V324', 'V330', 'V329', 'V322', 

          'V323', 'V278', 'V277', 'V252', 'V253', 'V254', 'V260']



for i in V1drop:

    drop_features.append(i)
second = V[50:100]

plot_feature_distribution(t0, t1, '0', '1', second)
V2drop = ['V263', 'V249', 'V266', 'V267', 'V268', 'V273', 'V276', 'V275', 'V247', 'V241', 'V240', 'V237', 'V235', 'V225', 

          'V224', 'V224', 'V248', 'V211', 'V213', 'V196', 'V205', 'V183', 'V206', 'V192']

for i in V2drop:

    drop_features.append(i)
third = V[100:150]

plot_feature_distribution(t0, t1, '0', '1', third)
V3drop = ['V191', 'V181', 'V193', 'V172', 'V173', 'V202', 'V203', 'V177', 'V179', 'V194', 'V185', 'V184', 'V175', 'V174', 

          'V195', 'V197', 'V198', 'V208', 'V210', 'V227', 'V251', 'V250', 'V271', 'V270', 'V225']

for i in V3drop:

    drop_features.append(i)
forth = V[150:200]

plot_feature_distribution(t0, t1, '0', '1', forth)
V4drop = ['V89', 'V256', 'V3', 'V1', 'V2', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V46', 'V42', 'V43', 'V47', 

         'V41', 'V39', 'V36', 'V35', 'V51', 'V50', 'V49', 'V48', 'V88', 'V93', 'V85', 'V84', 'V83', 'V81', 'V90', 'V91', 

         'V92', 'V94', 'V80', 'V79', 'V75', 'V75']

for i in V4drop:

    drop_features.append(i)
fifth = V[200:250]

plot_feature_distribution(t0, t1, '0', '1', fifth)
V5drop = ['V68', 'V27', 'V28', 'V53', 'V74', 'V73', 'V72', 'V66', 'V54', 'V67', 'V64', 'V63', 'V62', 'V61', 'V71', 'V69', 

          'V55', 'V60', 'V59', 'V58', 'V57', 'V65', 'V56', 'V70', 'V22', 'V23', 'V24', 'V34', 'V33', 'V32', 'V31', 'V30', 

          'V29', 'V26', 'V25', 'V15', 'V21', 'V14', 'V16', 'V17', 'V18', 'V19', 'V12', 'V20', 'V13', 'V301', 'V300', 'V296', 

          'V289', 'V288']

for i in V5drop:

    drop_features.append(i)
sixth = V[250:300]

plot_feature_distribution(t0, t1, '0', '1', sixth)
V6drop = ['V114', 'V110', 'V105', 'V104', 'V103', 'V102', 'V101', 'V100', 'V95', 'V99', 'V98', 'V107', 'V111', 'V112', 'V106',

         'V113', 'V108', 'V134', 'V133', 'V135', 'V132', 'V131', 'V130', 'V129', 'V126', 'V125', 'V124', 'V123', 'V122', 

          'V121', 'V120', 'V119', 'V118', 'V117', 'V116', 'V115', 'V109', 'V294']

for i in V6drop:

    drop_features.append(i)
seventh = V[300:350]

plot_feature_distribution(t0, t1, '0', '1', seventh)
V7drop = ['V305', 'V304', 'V303', 'V302', 'V299', 'V298', 'V297', 'V295', 'V293', 'V292', 'V291', 'V290', 'V287', 'V286',

         'V285', 'V289', 'V279', 'V309', 'V316', 'V318', 'V319']

for i in V7drop:

    drop_features.append(i)
D_done = ['D7', 'D13', 'D14', 'D12', 'D6', 'D9', 'D8']

D_not_done = missing['features'].apply(lambda x: x if x[0]=='D' else 0)

D_not_done = pd.DataFrame(D_not_done)

D_not_done = D_not_done[D_not_done['features']!=0]

D_not_done = D_not_done[~D_not_done['features'].isin(D_done)]

D_not_done = D_not_done[~D_not_done['features'].isin(['DeviceInfo', 'DeviceType'])]

D = list(D_not_done['features'])

D
plot_feature(train, test, 'D5', True)
plot_feature(train, test, 'D2', True)
plot_feature(train, test, 'D11', True)
plot_feature(train, test, 'D3', True)
plot_feature(train, test, 'D4', True)
plot_feature(train, test, 'D15', True)
plot_feature(train, test, 'D10', True)
plot_feature(train, test, 'D1', True)
C_not_done = missing['features'].apply(lambda x: x if x[0]=='C' else 0)

C_not_done = pd.DataFrame(C_not_done)

C_not_done = C_not_done[C_not_done['features']!=0]

C_not_done = list(C_not_done['features'])

C_not_done
plot_feature(train, test, 'C1', True)
plot_feature(train, test, 'C2', True)
plot_feature(train, test, 'C3', False)
drop_features.append('C3')
plot_feature(train, test, 'C4', True)
plot_feature(train, test, 'C5', True)
plot_feature(train, test, 'C6', True)
plot_feature(train, test, 'C7', True)
plot_feature(train, test, 'C8', True)
plot_feature(train, test, 'C9', True)
plot_feature(train, test, 'C10', True)
plot_feature(train, test, 'C11', True)
plot_feature(train, test, 'C12', True)
plot_feature(train, test, 'C13', True)
plot_feature(train, test, 'C14', True)
id_done = ['id_24', 'id_25', 'id_08', 'id_07', 'id_21', 'id_26', 'id_27', 'id_23', 'id_22', 'id_18', 'id_04', 'id_03', 

           'id_33', 'id_09', 'id_30', 'id_32', 'id_32', 'id_34', 'id_14']

id_not_done = missing['features'].apply(lambda x: x if x[0]=='i' else 0)

id_not_done = pd.DataFrame(id_not_done)

id_not_done = id_not_done[id_not_done['features']!=0]

id_not_done = id_not_done[~id_not_done['features'].isin(id_done)]

id_not_done
plot_c_feature(train, 'id_10')
check_features.append('id_10')
plot_c_feature(train, 'id_13')
plot_c_feature(train, 'id_16')
check_features.append('id_16')
plot_c_feature(train, 'id_05')
plot_c_feature(train, 'id_06')
plot_c_feature(train, 'id_20')
plot_c_feature(train, 'id_19')
plot_c_feature(train, 'id_17')
plot_c_feature(train, 'id_31')
plot_feature(train, test, 'id_02', False)
plot_c_feature(train, 'id_11')
plot_c_feature(train, 'id_28')
check_features.append('id_28')
plot_c_feature(train, 'id_29')
check_features.append('id_29')
plot_c_feature(train, 'id_38')
check_features.append('id_38')
plot_c_feature(train, 'id_37')
check_features.append('id_37')
plot_c_feature(train, 'id_36')
check_features.append('id_36')
plot_c_feature(train, 'id_35')
check_features.append('id_35')
plot_c_feature(train, 'id_15')
plot_feature(train, test, 'id_01', False)
plot_c_feature(train, 'id_12')
c_not_done = missing['features'].apply(lambda x: x if x[0]=='c' else 0)

c_not_done = pd.DataFrame(c_not_done)

c_not_done = c_not_done[c_not_done['features']!=0]

list(c_not_done['features'])
plot_feature(train, test, 'card1', True)
plot_feature(train, test, 'card2', False)
plot_feature(train, test, 'card3', True)
plot_c_feature(train, 'card4')
check_features.append('card4')
plot_feature(train, test, 'card5', True)
plot_c_feature(train, 'card6')
check_features.append('card6')
M_not_done = missing['features'].apply(lambda x: x if x[0]=='M' else 0)

M_not_done = pd.DataFrame(M_not_done)

M_not_done = M_not_done[M_not_done['features']!=0]

list(M_not_done['features'])
plot_c_feature(train, 'M1')
drop_features.append('M1')
plot_c_feature(train, 'M2')
check_features.append('M2')
plot_c_feature(train, 'M3')
check_features.append('M3')
plot_c_feature(train, 'M4')
plot_c_feature(train, 'M5')
check_features.append('M5')
plot_c_feature(train, 'M6')
check_features.append('M6')
train_copy = train.copy()

test_copy = test.copy()
def id_split(dataframe):

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

    

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = 'Others'

    gc.collect()

    return dataframe
# split some features and replace values

train = id_split(train)

test = id_split(test)
# filter usefull features with the e.d.a

usefull_features = [col for col in train.columns if col not in drop_features]

train = train[usefull_features]

usefull_features.remove('isFraud')

test = test[usefull_features]
# New feature - log of transaction amount. ()

train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])

test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])



# New feature - decimal part of the transaction amount.

train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 

          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 

          'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 

          'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 

          'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 

          'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 

          'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 

          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 

          'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 

          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 

          'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 

          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 

          'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 

          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 

          'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 

          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 

          'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 

          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']
for c in ['P_emaildomain', 'R_emaildomain']:

    train[c + '_bin'] = train[c].map(emails)

    test[c + '_bin'] = test[c].map(emails)

    

    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

    

    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

def setTime(df):

    df['TransactionDT'] = df['TransactionDT'].fillna(df['TransactionDT'].median())

    # Temporary

    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month

    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear

    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear

    

    df['DT_hour'] = df['DT'].dt.hour

    df['DT_day_week'] = df['DT'].dt.dayofweek

    df['DT_day'] = df['DT'].dt.day

    

    # Lets transform D8 and D9 column

    # As we almost sure it has connection with hours

    df['D9_not_na'] = np.where(df['D9'].isna(),0,1)

    df['D8_not_same_day'] = np.where(df['D8']>=1,1,0)

    df['D8_D9_decimal_dist'] = df['D8'].fillna(0)-df['D8'].fillna(0).astype(int)

    df['D8_D9_decimal_dist'] = ((df['D8_D9_decimal_dist']-df['D9'])**2)**0.5

    df['D8'] = df['D8'].fillna(-1).astype(int)



    return df

    

train=setTime(train)

test=setTime(test)
def addNewFeatures(data): 

    data['uid'] = data['card1'].astype(str)+'_'+data['card2'].astype(str)



    data['uid2'] = data['uid'].astype(str)+'_'+data['card3'].astype(str)+'_'+data['card5'].astype(str)



    data['uid3'] = data['uid2'].astype(str)+'_'+data['addr1'].astype(str)+'_'+data['addr2'].astype(str)

    

    return data
train = addNewFeatures(train)

test = addNewFeatures(test)
i_cols = ['card2','card3','card5','uid','uid2','uid3']



for col in i_cols:

    for agg_type in ['mean','std']:

        new_col_name = col+'_TransactionAmt_'+agg_type

        temp_df = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])

        #temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)

        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(

                                                columns={agg_type: new_col_name})



        temp_df.index = list(temp_df[col])

        temp_df = temp_df[new_col_name].to_dict()   



        train[new_col_name] = train[col].map(temp_df)

        test[new_col_name]  = test[col].map(temp_df)
train = train.replace(np.inf,999)

test = test.replace(np.inf,999)
i_cols = ['card1','card2','card3','card5',

          'C1','C2','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',

          'D1','D2','D3','D4','D5','D6','D7','D8',

          'addr1','addr2',

          'dist1','dist2',

          'P_emaildomain', 'R_emaildomain',

          'DeviceInfo','device_name',

          'id_30','id_33',

          'uid','uid2','uid3',

         ]



for col in i_cols:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   

    train[col+'_fq_enc'] = train[col].map(fq_encode)

    test[col+'_fq_enc']  = test[col].map(fq_encode)





for col in ['DT_M','DT_W','DT_D']:

    temp_df = pd.concat([train[[col]], test[[col]]])

    fq_encode = temp_df[col].value_counts().to_dict()

            

    train[col+'_total'] = train[col].map(fq_encode)

    test[col+'_total']  = test[col].map(fq_encode)



periods = ['DT_M','DT_W','DT_D']

i_cols = ['uid']

for period in periods:

    for col in i_cols:

        new_column = col + '_' + period

            

        temp_df = pd.concat([train[[col,period]], test[[col,period]]])

        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)

        fq_encode = temp_df[new_column].value_counts().to_dict()

            

        train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)

        test[new_column]  = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)

        

        train[new_column] /= train[period+'_total']

        test[new_column]  /= test[period+'_total']
# drop noisy columns    

train.drop(['TransactionDT', 'uid','uid2','uid3', 'DT','DT_M','DT_W','DT_D', 'DT_hour','DT_day_week','DT_day',

            'DT_D_total','DT_W_total','DT_M_total', 'id_30','id_31','id_33', 'D1', 'D2', 'D9'], axis = 1, inplace = True)

test.drop(['TransactionDT', 'uid','uid2','uid3', 'DT','DT_M','DT_W','DT_D', 'DT_hour','DT_day_week','DT_day',

            'DT_D_total','DT_W_total','DT_M_total', 'id_30','id_31','id_33', 'D1', 'D2', 'D9'], axis = 1, inplace = True)
for col in train.columns:

    if train[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values))
def agg_features(df):

    columns_a = ['TransactionAmt', 'id_02', 'D15']

    columns_b = ['card1', 'card4', 'addr1']

    for col_a in columns_a:

        for col_b in columns_b:

            df[f'{col_a}_to_mean_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')

            df[f'{col_a}_to_std_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('std')

    return df



test = agg_features(test)

train = agg_features(train)
def reduce_mem_usage(df):

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
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
X = train.drop(['isFraud'], axis = 1)

y = train['isFraud']



print('Our train set have {} columns'.format(train.shape[1]))

print('Our test set have {} columns'.format(test.shape[1]))



gc.collect()
params = {

                    'objective':'binary',

                    'boosting_type':'gbdt',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.005,

                    'num_leaves': 2**8,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.7,

                    'subsample_freq':1,

                    'subsample':0.7,

                    'n_estimators':100000,

                    'max_bin':255,

                    'verbose':-1,

                    'random_state': 47,

                    'early_stopping_rounds':100, 

                }
NFOLDS = 10

folds = KFold(n_splits=NFOLDS)





splits = folds.split(X, y)

y_preds = np.zeros(test.shape[0])

y_oof = np.zeros(X.shape[0])

score = 0



for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label = y_train)

    dvalid = lgb.Dataset(X_valid, label = y_valid)

    

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], 

                    verbose_eval = 200, early_stopping_rounds = 500)

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(test) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print("Mean AUC: ", score)

print("Out of folds AUC: ", roc_auc_score(y, y_oof))
sub['isFraud'] = y_preds

sub.to_csv('submission_v1.csv', index = False)
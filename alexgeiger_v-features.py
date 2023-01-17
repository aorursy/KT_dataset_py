import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import roc_auc_score

import random





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.

# Helper function for column value details

def column_value_freq(sel_col,cum_per):

    dfpercount = pd.DataFrame(columns=['col_name','num_values_'+str(round(cum_per,2))])

    for col in sel_col:

        col_value = train_transaction[col].value_counts(normalize=True)

        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})

        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()

        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:

            num_col_99 = len(colpercount.loc[colpercount['per_count'] > (1- cum_per),])

        else:

            num_col_99 = len(colpercount.loc[colpercount['cum_per_count']< cum_per,] )

        dfpercount=dfpercount.append({'col_name': col,'num_values_'+str(round(cum_per,2)): num_col_99},ignore_index = True)

    dfpercount['unique_values'] = train_transaction[sel_col].nunique().values

    dfpercount['unique_value_to_num_values'+str(round(cum_per,2))+'_ratio'] = 100 * (dfpercount['num_values_'+str(round(cum_per,2))]/dfpercount.unique_values)

    dfpercount['percent_missing'] = percent_na(train_transaction[sel_col])['percent_missing'].round(3).values

    return dfpercount



def column_value_details(sel_col,cum_per):

    dfpercount = pd.DataFrame(columns=['col_name','values_'+str(round(cum_per,2)),'values_'+str(round(1-cum_per,2))])

    for col in sel_col:

        col_value = train_transaction[col].value_counts(normalize=True)

        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})

        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()

        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:

            values_freq = colpercount.loc[colpercount['per_count'] > (1- cum_per),'value'].tolist()

        else:

            values_freq = colpercount.loc[colpercount['cum_per_count']< cum_per,'value'].tolist() 

        values_less_freq =  [item for item in colpercount['value'] if item not in values_freq]

        dfpercount=dfpercount.append({'col_name': col,'values_'+str(round(cum_per,2)) : values_freq ,'values_'+str(round(1-cum_per,2)): values_less_freq},ignore_index = True)

    num_values_per =[]

    for i in range(len(dfpercount)):

        num_values_per.append(len(dfpercount['values_'+str(round(cum_per,2))][i]))

    dfpercount['num_values_per'] = num_values_per

    return dfpercount



# Helper functions

# 1. For calculating % na values in  columns

def percent_na(df):

    percent_missing = df.isnull().sum() * 100 / len(df)

    missing_value_df = pd.DataFrame({'column_groups': percent_missing.index,

                                 'percent_missing': percent_missing.values})

    return missing_value_df

# 2. For plotting grouped histograms 

def sephist(col):

    yes = train_transaction[train_transaction['isFraud'] == 1][col]

    no = train_transaction[train_transaction['isFraud'] == 0][col]

    return yes, no



def plot_feature_distribution(df,features,label1='1',label2='0',target_field = 'isFraud'):

    df1 = df.loc[df[target_field] == 1].copy();

    df2 = df.loc[df[target_field] == 0].copy();

    i = 0;

    sns.set_style('whitegrid');

    #plt.figure(figsize=((10,8)));

    

    fig,ax = plt.subplots(1,3,figsize=[17,20]);

    

    for feature in features:

        i += 1

        plt.subplot(3,3,i);

        sns.distplot(df1[feature],label=label1);

        sns.distplot(df2[feature],label=label2);

        plt.xlabel(feature,fontsize=9);

        locs, labels = plt.xticks();

        plt.tick_params(axis='x',which='major',labelsize=8,pad=1);

        plt.tick_params(axis='y',which='major',labelsize=8);

    plt.show();

    

def get_score(df_temp,KEYS,feature,flip=True):

    

    #df_temp[feature+'mu'] = df_temp[KEYS].min(axis=1)

    df_temp[feature] = df_temp[KEYS].skew(axis=1)

    #df_temp.loc[df_temp[feature+'mu'] == 0,feature] = np.nan

    dff = df_temp[[feature,'isFraud']].copy().dropna()

    

    the_score = roc_auc_score(dff['isFraud'],dff[feature])

    if the_score < .5:

        df_temp[feature] = -df_temp[feature] + df_temp[feature].min()

        dff = df_temp[[feature,'isFraud']].copy().dropna()

        return roc_auc_score(dff['isFraud'],dff[feature])

    

    return the_score





def check_score(score,CHANGED,KEY_LIST,new_score,KEY_LIST_TEMP,what='remove'):

    if new_score > score - OFFSET:

        BREAK   = False

        CHANGED = True

        if (random.random()  < .5) or (score < new_score):

            print(FEATURE,' ',what,' new high score: ',new_score)# ,KEY_LIST)

            score    = new_score

            KEY_LIST = KEY_LIST_TEMP

            BREAK   = True

    return score,CHANGED,KEY_LIST,BREAK
# setup data

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_transaction  = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

transaction       = pd.concat([train_transaction,test_transaction],sort=False)



# find patterns

pd.options.display.max_colwidth = 300

Vcols=train_transaction.columns[train_transaction.columns.str.startswith('V')]

train_transaction_vcol_na = percent_na(train_transaction[Vcols])

train_transaction_vcol_na_group= train_transaction_vcol_na.groupby('percent_missing')['column_groups'].unique().reset_index()

num_values_per =[]

for i in range(len(train_transaction_vcol_na_group)):

    num_values_per.append(len(train_transaction_vcol_na_group['column_groups'][i]))

train_transaction_vcol_na_group['num_columns_group'] = num_values_per



# delete data

del train_transaction, test_transaction

gc.collect()



# print out data

train_transaction_vcol_na_group
def get_plot(df_temp,KEYS,feature,flip=True):

    

    df_temp[feature] = df_temp[KEYS].mean(axis=1)

    for key_in in KEYS:

        df_temp.loc[transaction[feature] == 0,key_in] = np.nan

        

    df_temp[feature] = df_temp[KEYS].kurt(axis=1)

    dff = df_temp[[feature,'isFraud']].copy().dropna()

    

    if flip:

        dff[feature] = -dff[feature] + dff[feature].max()

    plt.figure(figsize = (8, 8))

    sns.distplot(dff.loc[dff['isFraud'] == 0, feature], label = 'target == 0',kde=False,norm_hist = True)

    sns.distplot(dff.loc[dff['isFraud'] == 1, feature] , label = 'target == 1',kde=False,norm_hist = True)

    plt.show()

    

    

ADDED_SET = []

KEEP_FEATURES = ['TransactionID']

TOTAL_KEY_LIST = []

by   = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']

for n_key in np.linspace(0,14,15):

    

    # get feature list

    v_list = list(train_transaction_vcol_na_group.column_groups[n_key])



    # setup feature names

    feature1 = 'super_combo_sum_'+ str(int(n_key))

    feature2 = 'super_combo_zscore_'+ str(int(n_key))

    feature3 = 'super_member_zscore_'+ str(int(n_key))

    

    # add features to list

    KEEP_FEATURES += [feature1,feature2,feature3]

    

    # 

    transaction[feature1] = 0

    transaction[feature2] = 0

    good_list = []

    FEATURE_2 = []

    

    for key in v_list:



        if (transaction[key].min() != 0) | (transaction[key].nunique() < 6):

            continue



        # append to total key list

        TOTAL_KEY_LIST.append(key)

        

        # append key to list

        good_list.append(key)

        

        # generate log transform

        XNORM = np.log(transaction[key] - transaction[key].min() + 1)

        

        # add log transform to feature 1

        transaction[feature1] += XNORM

        

        # z-score normalize

        transaction[key] = (XNORM - XNORM.mean())/XNORM.std()

        

        # save data under feature two

        transaction[feature2] += transaction[key]       

        

        # append feature 2

        FEATURE_2.append(feature2)

        

    # take secondary log of data

    transaction[feature1] = np.log(transaction[feature1] + 1) 

    

    # transform features

    transaction[feature3] = (transaction[feature1] - transaction.groupby(by)[feature1].transform('mean'))/(transaction.groupby(by)[feature1].transform('std')+1)

        

    dff = transaction[[feature1,feature2,feature3,'isFraud']].copy().dropna()    

    # plot data

    plot_feature_distribution(dff,[feature1,feature2,feature3])

       
#KEEP_FEATURES += ['V_std_feature']

transaction[KEEP_FEATURES].to_pickle('transaction.pkl')


# reset transactions

'''

transaction = get_transactions(transaction)



for the_key in TOTAL_KEY_LIST:

    transaction[the_key] = (transaction[the_key] - transaction[the_key].min())/(transaction[the_key].max() - transaction[the_key].min())

'''



'''

gc.collect()





# define list of new features

feature  = 'adv_group_1_mu'

KEY_LIST =  ['V303', 'V317', 'V318', 'V133', 'V281', 'V75', 'V79', 'V81', 'V87', 'V38', 'V40', 'V201', 'V242', 'V243', 'V244', 'V247', 'V248', 'V249', 'V252', 'V274', 'V146', 'V147', 'V156', 'V158', 'V163', 'V320', 'V68', 'V27', 'V312', 'V28', 'V302', 'V304', 'V17', 'V19', 'V13', 'V18', 'V33', 'V21', 'V74', 'V309', 'V34', 'V59', 'V15', 'V54', 'V67', 'V89', 'V44', 'V308', 'V16', 'V45', 'V227', 'V259', 'V184', 'V211', 'V332', 'V294', 'V36', 'V52', 'V210', 'V213']

KEY_LIST = TOTAL_KEY_LIST

score = get_score(transaction,KEY_LIST,feature)

print('original score: ',score)



CHANGED = True

EXIT    = True

OFFSET  = .0000



while EXIT:

    print('')

    print('new loop')

    OFFSET = OFFSET*.9

    

    for FEATURE in TOTAL_KEY_LIST:

        if FEATURE in KEY_LIST:

            continue



        KEY_LIST_TEMP = KEY_LIST + [FEATURE]

        new_score =  get_score(transaction,KEY_LIST_TEMP,feature)

        

        # check to update score 

        if new_score > score - OFFSET:

            score,CHANGED,KEY_LIST,BREAK = check_score(score,CHANGED,KEY_LIST,new_score,KEY_LIST_TEMP,what='add')

            if BREAK & (random.random() < .2):

                break

     

    print('')

    print('Stage II')

    gc.collect()

    

    # if changed is false do not exit

    if CHANGED == False:  EXIT = False

        

    # reset changed

    CHANGED      = False

    

    # reset key list save

    KEY_LIST_SAVE = KEY_LIST

    

    # get features in key list

    for FEATURE in KEY_LIST_SAVE:



        KEY_LIST_TEMP = [col for col in KEY_LIST if col not in  [FEATURE]]

        new_score =  get_score(transaction,KEY_LIST_TEMP,feature)



        # check to update score 

        if new_score > score:

            score,CHANGED,KEY_LIST,BREAK = check_score(score,CHANGED,KEY_LIST,new_score,KEY_LIST_TEMP)

            if BREAK & (random.random() < .05):

                break

    gc.collect()

    if CHANGED == False:  EXIT = False

'''





'''

STD_KEYS = ['V303', 'V317', 'V318', 'V133', 'V281', 'V75', 'V79', 'V81', 'V87', 'V38', 'V40', 'V201', 'V242', 'V243', 'V244', 'V247', 'V248', 'V249', 'V252', 'V274', 'V146', 'V147', 'V156', 'V158', 'V163', 'V320', 'V68', 'V27', 'V312', 'V28', 'V302', 'V304', 'V17', 'V19', 'V13', 'V18', 'V33', 'V21', 'V74', 'V309', 'V34', 'V59', 'V15', 'V54', 'V67', 'V89', 'V44', 'V308', 'V16', 'V45', 'V227', 'V259', 'V184', 'V211', 'V332', 'V294', 'V36', 'V52', 'V210', 'V213']



transaction['V_std_feature'] = np.log(1+transaction[STD_KEYS].mean(axis=1))

transaction['V_std_feature'] = (transaction['V_std_feature'] - transaction.groupby(by)['V_std_feature'].transform('mean'))/(transaction.groupby(by)['V_std_feature'].transform('std')+1)

        

for feature in ['V_std_feature']:

    

    dff = transaction[[feature,'isFraud']].copy().dropna()

    print('score: ',roc_auc_score(dff['isFraud'],dff[feature]))

    dff = transaction[[feature,'isFraud']].copy().dropna()

    

    plt.figure(figsize = (8, 8))

    sns.distplot(dff.loc[dff['isFraud'] == 0, feature], label = 'target == 0',kde=False,norm_hist = True)

    sns.distplot(dff.loc[dff['isFraud'] == 1, feature] , label = 'target == 1',kde=False,norm_hist = True)

    plt.show()



'''







'''# get feature list

v_list = list(train_transaction_vcol_na_group.column_groups[0])



for key in v_list:

    

    tr = transaction.loc[transaction.isFraud.isna() &transaction[key].notna(),key]

    te = transaction.loc[transaction.isFraud.notna() &transaction[key].notna(),key]

    tr = np.log(tr + 1)

    te = np.log(te + 1)

    

    plt.figure(figsize = (8, 8))

    sns.distplot(tr, label = 'train',kde=False,norm_hist = True)

    sns.distplot(te, label = 'test',kde=False,norm_hist = True)

    plt.show()

# get feature list

v_list = list(train_transaction_vcol_na_group.column_groups[0])



by   = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']



transaction['DBOX'] = (transaction['super_combo_sum_0'] - transaction.groupby(by)['super_combo_sum_0'].transform('mean'))#/(transaction.groupby(by)['super_combo_sum_0'].transform('std')+1)



FEAT = 'super_combo_sum_0'



# plot data

for feature in ['DBOX','super_combo_sum_0']:

    transaction['DBOX'] = transaction.groupby(by)[FEAT].transform('nunique')

    dff = transaction[[feature,'isFraud']].copy().dropna()

    plt.figure(figsize = (8, 8))

    sns.distplot(dff.loc[dff['isFraud'] == 0, feature], label = 'target == 0',kde=False,norm_hist = True)

    sns.distplot(dff.loc[dff['isFraud'] == 1, feature] , label = 'target == 1',kde=False,norm_hist = True)

    plt.show()

    

'''
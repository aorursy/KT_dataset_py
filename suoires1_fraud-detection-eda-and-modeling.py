import os

import sys

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

%matplotlib inline



# disable the warning about settingwithcopy warning:

pd.set_option('chained_assignment',None)
working_directory_path = "/kaggle/input/ieee-fraud-detection/"

os.chdir(working_directory_path)
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
train_identity = pd.read_csv("train_identity.csv")

train_transaction = pd.read_csv("train_transaction.csv")



test_identity = pd.read_csv("test_identity.csv")

test_transaction = pd.read_csv("test_transaction.csv")



train_identity = reduce_mem_usage(train_identity)

train_transaction = reduce_mem_usage(train_transaction)

test_identity = reduce_mem_usage(test_identity)

test_transaction = reduce_mem_usage(test_transaction)
# Since the number of columns are too large, we can expand it using pd.set_option()

pd.set_option('display.max_columns', None)  
train_transaction['TransactionID'].value_counts().sort_values(ascending = False)
train_full = pd.merge(train_identity, train_transaction, on = 'TransactionID')
print('Number of row in transaction:', len(train_transaction))

print('Number of row in identity:', len(train_identity))



# remove train_transaction from memory

# del train_transaction
train_full.info(verbose=True, null_counts=True)
train_full_cat = train_full.filter(regex='id|card|ProductCD|addr|email|M|DeviceType|DeviceInfo')
plt.figure(figsize=(18,9))

sns.heatmap(train_full_cat.isnull(), cbar= False)
train_full_cat[['id_01','id_12','card1','card2']].info(null_counts=True)
train_full_num = train_full.filter(regex='isFraud|TransactionDT|TransactionAmt|dist|C|D')

plt.figure(figsize=(18,9))

sns.heatmap(train_full_num.isnull(), cbar= False)
train_full_Vesta = train_full.filter(regex='V')

plt.figure(figsize=(18,9))

sns.heatmap(train_full_Vesta.isnull(), cbar= False)
msno.dendrogram(train_full_Vesta)
plt.hist(train_transaction['TransactionDT'], label='train')

plt.hist(test_transaction['TransactionDT'], label='test')

plt.legend()

plt.title('Distribution of TransactionDT')
plt.figure(figsize=(12,6))

g = sns.countplot(x = 'isFraud', data = train_full)

g.set_title("Fraud Distribution", fontsize = 17)

g.set_xlabel("Is Fraud?", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)

plt.legend(title='Fraud', labels=['No', 'Yes'])



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/len(train_full) * 100),

            ha="center", fontsize=15) 
train_full_cat.head()
plt.figure(figsize=(12,6))



total = len(train_full_cat)



plt.subplot(121)

g = sns.countplot(x = 'ProductCD', data = train_full_cat)

g.set_title('ProductCD Distribution', fontsize = 15)

g.set_xlabel("Product Code", fontsize=15)

g.set_ylabel("Count", fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 



plt.subplot(122)

g1 = sns.countplot(x='ProductCD', hue='isFraud', data=train_full)

g1.set_title('ProductCD by Fraud', fontsize = 15)

g1.set_xlabel("Product Code", fontsize=15)

g1.set_ylabel("Count", fontsize=15)

plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
train_full[train_full['isFraud'] == 1]['ProductCD'].value_counts(normalize = True)
# grouped table

train_full.groupby('ProductCD')['isFraud'].value_counts(normalize = True)
# visualization of table

plt.figure(figsize=(12,12))

a = train_full.groupby('ProductCD')['isFraud'].value_counts(normalize = True).unstack().plot.bar(stacked = True)

a.set_title('Rate of Fraud by Product Category', fontsize = 15)

plt.xticks(rotation='horizontal')
plt.figure(figsize=(12,10))

sns.boxplot(x = 'ProductCD', y = 'TransactionAmt', hue = 'isFraud', data = train_full)
plt.figure(figsize=(12,10))

sns.boxplot(x = 'ProductCD', y = 'TransactionDT', hue = 'isFraud', data = train_full)
train_full_cat.describe().loc[:,'card1':'card5']
train_full_cat.loc[:,'card1':'card5'].nunique()
plt.figure(figsize=(12,6))



total = len(train_full_cat)



plt.subplot(121)

g = sns.countplot(x = 'card4', data = train_full_cat)

g.set_title('Card Network Distribution', fontsize = 15)

g.set_xlabel("Card Issuers", fontsize=15)

g.set_ylabel("Count", fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 



plt.subplot(122)

g1 = sns.countplot(x='card4', hue='isFraud', data=train_full)

g1.set_title('Card Network by Fraud', fontsize = 15)

g1.set_xlabel("Card Issuers", fontsize=15)

g1.set_ylabel("Count", fontsize=15)

plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
train_full[train_full['isFraud'] == 1]['card4'].value_counts(normalize = True)
# grouped table

train_full.groupby('card4')['isFraud'].value_counts(normalize = True)
# visualization of table

plt.figure(figsize=(12,12))

b = train_full.groupby('card4')['isFraud'].value_counts(normalize = True).unstack().plot.bar(stacked = True)

b.set_title('Rate of Fraud by Card Network', fontsize = 15)

plt.xticks(rotation='horizontal')
plt.figure(figsize=(12,6))



total = len(train_full_cat)



plt.subplot(121)

g = sns.countplot(x = 'card6', data = train_full)

g.set_title('Card Type Distribution', fontsize = 15)

g.set_xlabel("Card Type", fontsize=15)

g.set_ylabel("Count", fontsize=15)

for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 



plt.subplot(122)

g1 = sns.countplot(x='card6', hue='isFraud', data=train_full)

g1.set_title('Card Type by Fraud', fontsize = 15)

g1.set_xlabel("Card Type", fontsize=15)

g1.set_ylabel("Count", fontsize=15)

plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
# grouped table

train_full.groupby('card6')['isFraud'].value_counts(normalize = True)
# visualization of table

plt.figure(figsize=(12,12))

b = train_full.groupby('card6')['isFraud'].value_counts(normalize = True).unstack().plot.bar(stacked = True)

b.set_title('Rate of Fraud by Card Type', fontsize = 15)

plt.xticks(rotation='horizontal')
plt.figure(figsize=(12,6))



g = sns.countplot(x = 'P_emaildomain', data = train_full)

g.set_title('Purchaser Email Domain Distribution', fontsize = 15)

g.set_xlabel("Email Domain", fontsize=15)

g.set_ylabel("Count", fontsize=15)

plt.xticks(rotation='vertical')

train_full["P_parent_emaildomain"] = train_full["P_emaildomain"].str.split('.', expand = True)[[0]]
plt.figure(figsize=(12,6))



g = sns.countplot(x = 'P_parent_emaildomain', data = train_full)

g.set_title('Purchaser Email Domain Distribution', fontsize = 15)

g.set_xlabel("Email Domain", fontsize=15)

g.set_ylabel("Count", fontsize=15)

plt.xticks(rotation= "vertical")
P_emaildomain_fraud_rate = train_full.groupby('P_parent_emaildomain')['isFraud'].value_counts(normalize = True).unstack().fillna(0)[1]



plt.figure(figsize=(12,6))



g = sns.countplot(x = 'P_parent_emaildomain', data = train_full, order = P_emaildomain_fraud_rate.index)

g.set_title('Purchaser Email Domain Distribution', fontsize = 15)

g.set_xlabel("Email Domain", fontsize=15)

g.set_ylabel("Count", fontsize=15)

plt.xticks(rotation= "vertical")



r = g.twinx()

r = sns.pointplot(x = P_emaildomain_fraud_rate.index, y = P_emaildomain_fraud_rate, color = 'blue')

r.set_ylabel("Fraud Rate", fontsize = 16, color = "blue")

protonmail_fraud = len(train_full[(train_full['P_parent_emaildomain'] == "protonmail") & (train_full['isFraud'] == 1)])

protonmail_non_fraud = len(train_full[(train_full['P_parent_emaildomain'] == "protonmail") & (train_full['isFraud'] == 0)])



protonmail_fraud_rate = protonmail_fraud/ (protonmail_fraud + protonmail_non_fraud)

print("Number of protonmail fraud transactions:", protonmail_fraud)

print("Number of protonmail non-fraud transactions:", protonmail_non_fraud)

print("Protonmail fraud rate:", protonmail_fraud_rate)
train_full["R_parent_emaildomain"] = train_full["R_emaildomain"].str.split('.', expand = True)[[0]]

train_full["R_parent_emaildomain"].fillna("NA", inplace=True)



R_emaildomain_fraud_rate = train_full.groupby('R_parent_emaildomain')['isFraud'].value_counts(normalize = True).unstack().fillna(0)[1]



plt.figure(figsize=(12,6))



g = sns.countplot(x = 'R_parent_emaildomain', data = train_full, order = R_emaildomain_fraud_rate.index)

g.set_title('Recipient Email Domain Distribution', fontsize = 15)

g.set_xlabel("Email Domain", fontsize=15)

g.set_ylabel("Count", fontsize=15)

plt.xticks(rotation= "vertical")



r = g.twinx()

r = sns.pointplot(x = R_emaildomain_fraud_rate.index, y = R_emaildomain_fraud_rate, color = "blue")

r.set_ylabel("Fraud Rate", fontsize = 16, color = "blue")
def visualize_cat_cariable(variable, df=train_full):

    train_full[variable].fillna("NA", inplace=True)

    variable_fraud_rate = df.groupby(variable)['isFraud'].value_counts(normalize = True).unstack().fillna(0)[1]

    

    plt.figure(figsize=(12,6))



    g = sns.countplot(x = variable, data = df, order = variable_fraud_rate.index)

    g.set_title('{} Count'.format(variable), fontsize = 15)

    g.set_xlabel("{}".format(variable), fontsize=15)

    g.set_ylabel("Count", fontsize=15)

    plt.xticks(rotation= "vertical")



    r = g.twinx()

    r = sns.pointplot(x = variable_fraud_rate.index, y = variable_fraud_rate, color = "blue")

    r.set_ylabel("Fraud Rate", fontsize = 16, color = "blue")

train_full_cat.loc[:,'M1':'M9'].apply(pd.value_counts)
visualize_cat_cariable('M4')
visualize_cat_cariable('DeviceType')
train_full['DeviceInfo'].value_counts()
devicelist = train_full.groupby('DeviceInfo').filter(lambda x: len(x) >500)['DeviceInfo'].unique()
visualize_cat_cariable('DeviceInfo', df = train_full[train_full['DeviceInfo'].isin(devicelist)])
# id_list = train_full.loc[:1, 'id_12':'id_38'].columns



# for i in id_list:

#     print (visualize_cat_variable(i))
visualize_cat_cariable('id_23')
visualize_cat_cariable('id_30')
train_full['major_os'] = train_full["id_30"].str.split(' ', expand = True)[[0]]



visualize_cat_cariable('major_os')
visualize_cat_cariable('id_31')
train_full['browser'] = train_full["id_31"].str.split(' ', expand = True)[[0]]



visualize_cat_cariable('browser')
browser_list = train_full.groupby('browser').filter(lambda x: len(x) > 144)['browser'].unique()

visualize_cat_cariable('browser',  df = train_full[train_full['browser'].isin(browser_list)])
def visualize_num_variable(variable, df=train_full):

    plt.figure(figsize=(12,18))

    plt.suptitle('Distribution of: {}'.format(variable), fontsize=22)



    plt.subplot(321)

    sns.distplot(df[variable], kde= False)

    plt.title('{} Distribution'.format(variable), fontsize = 15)



    plt.subplot(322)

    sns.distplot(np.log10(df[variable]), kde= False)

    plt.title('Log-transformed Distribution', fontsize = 15)





    plt.subplot(323)

    sns.distplot(df[df['isFraud'] == 0][variable], color = 'skyblue', kde= False, label = 'Not Fraud')

    sns.distplot(df[df['isFraud'] == 1][variable], color = 'red', kde= False , label = 'Fraud')

    plt.title('Fraud vs Non-Fraud Distribution', fontsize = 15)

    plt.legend()



    plt.subplot(324)

    sns.distplot(np.log10(df[df['isFraud'] == 0][variable]), color = 'skyblue', kde= False, label = 'Not Fraud')

    sns.distplot(np.log10(df[df['isFraud'] == 1][variable]), color = 'red', kde= False , label = 'Fraud')

    plt.title('Log-transformed Distribution', fontsize = 15)

    plt.legend()

    

    plt.subplot(313)

    sns.boxplot(x = 'isFraud', y = variable, data = df)

    plt.title('Transaction Amount by Fraud', fontsize = 15,  weight='bold')
visualize_num_variable('TransactionAmt')
visualize_num_variable('TransactionDT')
def visualize_num_variable(variable, df=train_full.copy()):

    # check for homogeneity:

    if len(df[variable].unique()) <= 1:

        print('{} is a homogeneous set'.format(variable))

        return

    

    # check for NAs and Zeros

    if df[variable].isnull().values.any():

        df = train_full.dropna(subset=[variable])



    if df[variable].min() < 0:

        plt.figure(figsize=(12,12))

        plt.suptitle('Distribution of: {}'.format(variable), fontsize=22)

    

        plt.subplot(221)

        sns.distplot(df[variable], kde= False)

        plt.title('{} Distribution'.format(variable), fontsize = 15)

        

        plt.subplot(222)

        sns.distplot(df[df['isFraud'] == 0][variable], color = 'skyblue', kde= False, label = 'Not Fraud')

        sns.distplot(df[df['isFraud'] == 1][variable], color = 'red', kde= False , label = 'Fraud')

        plt.title('Fraud vs Non-Fraud Distribution', fontsize = 15)

        plt.legend()

        

        plt.subplot(212)

        sns.boxplot(x = 'isFraud', y = variable, data = df)

        plt.title('{} by Fraud'.format(variable), fontsize = 15,  weight='bold')

        

    else:

        smallest_value = df[df[variable] != 0][variable].min()

        if df[variable].min() == 0:

            df[variable].replace(0, smallest_value/10, inplace=True)       



        plt.figure(figsize=(12,18))

        plt.text(x=0.5, y=0.5,

                 s="Zeros have been replaced with {} to avoid log infinity".format(smallest_value/10),

                 fontsize=12,horizontalalignment='center')



        plt.suptitle('Distribution of: {}'.format(variable), fontsize=22)



        plt.subplot(321)

        sns.distplot(df[variable], kde= False)

        plt.title('{} Distribution'.format(variable), fontsize = 15)



        plt.subplot(322)

        sns.distplot(np.log10(df[variable]), kde= False)

        plt.title('Log-transformed Distribution', fontsize = 15)





        plt.subplot(323)

        sns.distplot(df[df['isFraud'] == 0][variable], color = 'skyblue', kde= False, label = 'Not Fraud')

        sns.distplot(df[df['isFraud'] == 1][variable], color = 'red', kde= False , label = 'Fraud')

        plt.title('Fraud vs Non-Fraud Distribution', fontsize = 15)

        plt.legend()



        plt.subplot(324)

        sns.distplot(np.log10(df[df['isFraud'] == 0][variable]), color = 'skyblue', kde= False, label = 'Not Fraud')

        sns.distplot(np.log10(df[df['isFraud'] == 1][variable]), color = 'red', kde= False , label = 'Fraud')

        plt.title('Log-transformed Distribution', fontsize = 15)

        plt.legend()



        plt.subplot(313)

        sns.boxplot(x = 'isFraud', y = variable, data = df)

        plt.title('{} by Fraud'.format(variable), fontsize = 15,  weight='bold')
visualize_num_variable('dist2')
# id_list = train_full.loc[:1, 'C1':'C14'].columns



# for i in id_list:

#     print (visualize_num_variable(i))
visualize_num_variable('C3')
# id_list = train_full.loc[:1, 'D1':'D15'].columns



# for i in id_list:

#     print (visualize_num_variable(i))
visualize_num_variable('D2')

visualize_num_variable('D8')

visualize_num_variable('D9')
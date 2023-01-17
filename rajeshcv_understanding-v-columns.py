import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
len(train_transaction.isFraud[train_transaction.isFraud==1])/len(train_transaction)
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
pd.options.display.max_colwidth =300

Vcols=train_transaction.columns[train_transaction.columns.str.startswith('V')]

train_transaction_vcol_na = percent_na(train_transaction[Vcols])

train_transaction_vcol_na_group= train_transaction_vcol_na.groupby('percent_missing')['column_groups'].unique().reset_index()

num_values_per =[]

for i in range(len(train_transaction_vcol_na_group)):

    num_values_per.append(len(train_transaction_vcol_na_group['column_groups'][i]))

train_transaction_vcol_na_group['num_columns_group'] = num_values_per

train_transaction_vcol_na_group
pd.options.display.max_colwidth =300

Vcols=test_transaction.columns[test_transaction.columns.str.startswith('V')]

test_transaction_vcol_na = percent_na(test_transaction[Vcols])

test_transaction_vcol_na_group= test_transaction_vcol_na.groupby('percent_missing')['column_groups'].unique().reset_index()

num_values_per =[]

for i in range(len(test_transaction_vcol_na_group)):

    num_values_per.append(len(test_transaction_vcol_na_group['column_groups'][i]))

test_transaction_vcol_na_group['num_columns_group'] = num_values_per

test_transaction_vcol_na_group
def vcol_multiplot(col,cum_per,ax1):

    col_freq = column_value_freq(col,cum_per)      

    plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_'+str(round(cum_per,2))],kind='bar',rot=90,ax = ax1)

    for p in plot1.patches[1:]:

        h = p.get_height()

        x = p.get_x()+p.get_width()/2.

        if h != 0:

            plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")

    plot1.set(ylabel='Count')

    plot1= plot1.set(title='Data Details  in each V columns with ' + str(round(col_freq.percent_missing.mean(),4)) +'% missing values')

    

def vcol_plot(col,cum_per):

    col_freq = column_value_freq(col,cum_per)      

    plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_'+str(round(cum_per,2))],kind='bar',rot=90)

    for p in plot1.patches[1:]:

        h = p.get_height()

        x = p.get_x()+p.get_width()/2.

        if h != 0:

            plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")

    plot1.set(ylabel='Count')

    plot1= plot1.set(title='Data Details  in each V columns with ' + str(round(col_freq.percent_missing.mean(),4)) +'% missing values')
cum_per = 0.965

fig, axs = plt.subplots(2,1, figsize=(15, 16), facecolor='w', edgecolor='k',squeeze=False)

axs=axs.ravel()

vcol_multiplot(train_transaction_vcol_na_group.column_groups[0],cum_per,axs[0])

vcol_multiplot(train_transaction_vcol_na_group.column_groups[1],cum_per,axs[1])
fig, axs = plt.subplots(4,2, figsize=(15,16), facecolor='w', edgecolor='k',squeeze=False)

#fig.subplots_adjust(hspace = 0.75, wspace=.001)

axs = axs.ravel()

for i in range(2,10):

    vcol_multiplot(train_transaction_vcol_na_group.column_groups[i],cum_per,axs[i-2])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig, axs = plt.subplots(5,1, figsize=(15,16), facecolor='w', edgecolor='k',squeeze=False)

axs=axs.ravel()

vcol_multiplot(train_transaction_vcol_na_group.column_groups[10],cum_per,axs[0])

vcol_multiplot(train_transaction_vcol_na_group.column_groups[11],cum_per,axs[1])

vcol_multiplot(train_transaction_vcol_na_group.column_groups[12],cum_per,axs[2])

vcol_multiplot(train_transaction_vcol_na_group.column_groups[13],cum_per,axs[3])

vcol_multiplot(train_transaction_vcol_na_group.column_groups[14],cum_per,axs[4])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
colfreq=column_value_freq(Vcols,cum_per)

colfreqbool = colfreq[colfreq.unique_values==2]

if len(colfreqbool)%3 == 0:

    nrow = len(colfreqbool)/3

else:

    nrow = len(colfreqbool) // 3 + 1 

sns.set(rc={'figure.figsize':(14,16)})

for num, alpha in enumerate(colfreqbool.col_name):

    plt.subplot(nrow, 3, num+1)

    plot1= sns.countplot(data=train_transaction,x=alpha,hue='isFraud')

    for p in plot1.patches[1:]:

        h = p.get_height()

        x = p.get_x()+p.get_width()/2.

        if h != 0:

            plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")

    plt.legend(title='isFraud',loc='upper right')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
def cum_value_count(col):

    col_value = train_transaction[col].value_counts(normalize=True)

    colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})

    colpercount['cum_per_count'] = colpercount['per_count'].cumsum()

    return colpercount
def V_doublecat_plot(cols,cum_per,limit):

    Vcol_details=column_value_details(cols,cum_per)

    V_cat = Vcol_details[Vcol_details['num_values_per'] <= limit].reset_index()

    sns.set(rc={'figure.figsize':(14,len(V_cat)*2)})

    x=1

    for num, alpha in enumerate(V_cat.col_name):

        plt.subplot(len(V_cat),2,x)

        sns.countplot(data=train_transaction[train_transaction[alpha].isin (V_cat['values_'+str(round(cum_per,2))][num])],y=alpha,hue='isFraud')

        plt.legend(loc='lower right')

        plt.title('Count of unique values which make '+str(round(cum_per*100,3))+'% of data in column ' + str(alpha) )

        plt.subplot(len(V_cat),2,x+1)

        sns.countplot(data=train_transaction[train_transaction[alpha].isin (V_cat['values_'+str(round(1-cum_per,2))][num])],y=alpha,hue='isFraud')

        plt.legend(loc='lower right')

        plt.title('Count of unique values which make only '+str(round((1-cum_per)*100,3))+'% of data in column ' + str(alpha) )

        x= x+2

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
def V_cat_plot(cols,cum_per,limit):

    Vcol_details=column_value_details(cols,cum_per)

    V_cat = Vcol_details[Vcol_details['num_values_per'] <= limit].reset_index()

    sns.set(rc={'figure.figsize':(14,len(V_cat)*2)})

    x=1

    for num, alpha in enumerate(V_cat.col_name):

        plt.subplot(len(V_cat),2,x)

        sns.countplot(data=train_transaction[train_transaction[alpha].isin (V_cat['values_'+str(round(cum_per,2))][num])],y=alpha,hue='isFraud')

        plt.legend(loc='lower right')

        plt.title('Count of unique values which make '+str(round(cum_per*100,3))+'% of data in column ' + str(alpha) )

        plt.subplot(len(V_cat),2,x+1)

        yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (V_cat['values_'+str(round(1-cum_per,2))][num]))][alpha]

        no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (V_cat['values_'+str(round(1-cum_per,2))][num]))][alpha]

        plt.hist(yes, alpha=0.75, label='Fraud', color='r')

        plt.hist(no, alpha=0.25, label='Not Fraud', color='g')

        plt.legend(loc='upper right')

        plt.title('Histogram of values which make '+str(round((1-cum_per)*100,3))+'% of data in column ' + str(alpha) )

        x= x+2

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
def V_num_plot(cols,cum_per,limit):

    Vcol_details=column_value_details(cols,cum_per)

    V_num = Vcol_details[Vcol_details['num_values_per'] > limit].reset_index()

    sns.set(rc={'figure.figsize':(14,len(V_num)*2)})

    x=1

    for num, alpha in enumerate(V_num.col_name):

        plt.subplot(len(V_num),2,x)

        yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (V_num['values_'+str(round(cum_per,2))][num]))][alpha]

        no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (V_num['values_'+str(round(cum_per,2))][num]))][alpha]

        plt.hist(yes, alpha=0.75, label='Fraud', color='r')

        plt.hist(no, alpha=0.25, label='Not Fraud', color='g')

        plt.legend(loc='upper right')

        plt.title('Histogram of  values which make '+str(round(cum_per*100,3))+'% of data in column ' + str(alpha) )

        plt.subplot(len(V_num),2,x+1)

        yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (V_num['values_'+str(round(1-cum_per,2))][num]))][alpha]

        no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (V_num['values_'+str(round(1-cum_per,2))][num]))][alpha]

        plt.hist(yes, alpha=0.75, label='Fraud', color='r')

        plt.hist(no, alpha=0.25, label='Not Fraud', color='g')

        plt.legend(loc='upper right')

        plt.title('Histogram of values which make '+str(round((1-cum_per)*100,3))+'% of data in column ' + str(alpha) )

        x= x+2

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
colfreqpseudobool = colfreq[(colfreq.unique_values !=2) & (colfreq['num_values_'+str(round(cum_per,2))] <= 2)]
pseudoboolcat = colfreqpseudobool[colfreqpseudobool.unique_values <=15]['col_name'].values

V_doublecat_plot(pseudoboolcat,cum_per,15)
pseudoboolnum = colfreqpseudobool[colfreqpseudobool.unique_values >15]['col_name'].values

V_cat_plot(pseudoboolnum,cum_per,15)
colfreqcat = colfreq[(colfreq.unique_values <=15) & (colfreq['num_values_'+str(round(cum_per,2))] > 2)]

colfreqcat 
colfreqpseudocat = colfreq[(colfreq.unique_values >15) & (colfreq['num_values_'+str(round(cum_per,2))] <= 15) & (colfreq['num_values_'+str(round(cum_per,2))]> 2)]

V_cat_plot(colfreqpseudocat.col_name,cum_per,15)
colfreqnum = colfreq[colfreq['num_values_'+str(round(cum_per,2))]>15]

V_num_plot(colfreqnum.col_name,cum_per,15)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

#test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
# Helper functions

# 1. For calculating % na values in  columns

def percent_na(df):

    percent_missing = df.isnull().sum() * 100 / len(df)

    missing_value_df = pd.DataFrame({'column_name': percent_missing.index,

                                 'percent_missing': percent_missing.values})

    return missing_value_df

# 2. For plotting grouped histograms 

def sephist(col):

    yes = train_transaction[train_transaction['isFraud'] == 1][col]

    no = train_transaction[train_transaction['isFraud'] == 0][col]

    return yes, no
# Helper function for column value details



def column_value_freq(df,sel_col,cum_per):

    dfpercount = pd.DataFrame(columns=['col_name','num_values_99'])

    for col in sel_col:

        col_value = df[col].value_counts(normalize=True)

        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})

        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()

        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:

            num_col_99 = len(colpercount.loc[colpercount['per_count'] > (1- cum_per),])

        else:

            num_col_99 = len(colpercount.loc[colpercount['cum_per_count']< cum_per,] )

        dfpercount=dfpercount.append({'col_name': col,'num_values_99': num_col_99},ignore_index = True)

    dfpercount['unique_values'] = df[sel_col].nunique().values

    dfpercount['unique_value_to_num_values_99_ratio'] = 100 * (dfpercount.num_values_99/dfpercount.unique_values)

    dfpercount['percent_missing'] = percent_na(df[sel_col])['percent_missing'].round(3).values

    return dfpercount



def column_value_details(df,sel_col,cum_per):

    dfpercount = pd.DataFrame(columns=['col_name','values_'+str(round(cum_per,2)),'values_'+str(round(1-cum_per,2))])

    for col in sel_col:

        col_value = df[col].value_counts(normalize=True)

        colpercount = pd.DataFrame({'value' : col_value.index,'per_count' : col_value.values})

        colpercount['cum_per_count'] = colpercount['per_count'].cumsum()

        if len(colpercount.loc[colpercount['cum_per_count'] < cum_per,] ) < 2:

            values_freq = colpercount.loc[colpercount['per_count'] > (1- cum_per),'value'].tolist()

        else:

            values_freq = colpercount.loc[colpercount['cum_per_count']< cum_per,'value'].tolist() 

        values_less_freq =  [item for item in colpercount['value'] if item not in values_freq]

        dfpercount=dfpercount.append({'col_name': col,'values_'+str(round(cum_per,2)) : values_freq ,'values_'+str(round(1-cum_per,2)): values_less_freq},ignore_index = True)

    return dfpercount
def col_unique(df,cols):

    dat=df[cols].nunique()

    sns.set(rc={'figure.figsize':(8,4)})

    plot=sns.barplot(x=dat.index,y=dat.values)

    for p in plot.patches:

        plot.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

               ha='center', va='top', fontsize=12, color='black', xytext=(0, 20),

                 textcoords='offset points')

    plot=plot.set(xlabel='Column ',ylabel= 'Number of unique values')
Ccols= train_transaction.columns[train_transaction.columns.str.startswith('C')]

train_transaction[Ccols].describe()
col_freq = column_value_freq(train_transaction,Ccols,0.965)

sns.set(rc={'figure.figsize':(12,8)})

plot=col_freq.plot(x='col_name',y='percent_missing',color='r')

plot.set(ylabel='Percentage of missing values')

ax1=plot.twinx()

#Dcol_freq['percent_missing'].plot(secondary_y=True, color='k', marker='o')

#Dcol_freq['unique_value_to_num_values_99_ratio'].plot(secondary_y=True, color='r', marker='o')

plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_99'],ax=ax1,kind='bar')

for p in plot1.patches[1:]:

    h = p.get_height()

    x = p.get_x()+p.get_width()/2.

    if h != 0:

        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")

plot1.set(ylabel='Count')

plot= plot.set(title='Data Details  in each C columns of train_transaction')

col_freq = column_value_freq(test_transaction,Ccols,0.965)

sns.set(rc={'figure.figsize':(12,8)})

plot=col_freq.plot(x='col_name',y='percent_missing',color='r')

plot.set(ylabel='Percentage of missing values')

ax1=plot.twinx()

#Dcol_freq['percent_missing'].plot(secondary_y=True, color='k', marker='o')

#Dcol_freq['unique_value_to_num_values_99_ratio'].plot(secondary_y=True, color='r', marker='o')

plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_99'],ax=ax1,kind='bar')

for p in plot1.patches[1:]:

    h = p.get_height()

    x = p.get_x()+p.get_width()/2.

    if h != 0:

        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")

plot1.set(ylabel='Count')

plot= plot.set(title='Data Details  in each C columns of test_transaction')
# cards=['card1','card2','card3','card4','card5','card6']

# by = cards+['addr1']+Ccols.tolist()

# group1=train_transaction.groupby(by,as_index=False)['TransactionID'].count()

# group1.sort_values(by='TransactionID',ascending=False).head(30)
# pd.options.display.max_columns = None

# Dcols =train_transaction.columns[train_transaction.columns.str.startswith('D')]

# select=train_transaction.columns[1:55]

# group1_details=pd.merge(group1,train_transaction[select],on=by,how='right')

# #group1_details.sort_values(by=['TransactionID','TransactionDT'],ascending=False)

# #group1_details[(group1_details.card1==16075) & (group1_details.TransactionID==60)]

# group1_details[(group1_details[['D1','D2','D3']].notnull().all(1)) & (group1_details.TransactionID>30 )].head(5)

# group1_details[(group1_details.card1==1342) & (group1_details.TransactionID==39)]
pd.options.display.max_columns = None

by=['card1','card2','card3','card4','card5','card6','addr1','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11']

cards_addr1_Ccolsgroup_count=train_transaction.groupby(by,as_index=False)['TransactionID'].count()

cards_addr1_Ccolsgroup_count.rename(columns={"TransactionID": "Count"},inplace=True)

cards_addr1_Ccolsgroup_count.sort_values(by='Count',ascending=False).head(30)
print('Total number of groups: ',len(cards_addr1_Ccolsgroup_count))

print('Average number of transaction per group: ',len(train_transaction)/len(cards_addr1_Ccolsgroup_count))
cards_addr1_Ccolsgroup_count[cards_addr1_Ccolsgroup_count.card1==9885].sort_values(by='Count',ascending=False).head(30)
select=train_transaction.columns[0:55]

cards_addr1_Ccolsgroup_details=pd.merge(cards_addr1_Ccolsgroup_count,train_transaction[select],on=by,how='left')

#cards_addr1_Ccolsgroup_details.sort_values(by=['TransactionID','TransactionDT'],ascending=True)
pd.options.display.max_rows = None

cards_addr1_Ccolsgroup_details[(cards_addr1_Ccolsgroup_details.card1==9885) & (cards_addr1_Ccolsgroup_details.Count==64) ]
train_transaction[(train_transaction.card1==9885) & (train_transaction.D2==181.0) ]
Dcols= train_transaction.columns[train_transaction.columns.str.startswith('D')]
col_freq = column_value_freq(train_transaction,Dcols,0.965)

sns.set(rc={'figure.figsize':(12,8)})

plot=col_freq.plot(x='col_name',y='percent_missing',color='r')

plot.set(ylabel='Percentage of missing values')

ax1=plot.twinx()

#Dcol_freq['percent_missing'].plot(secondary_y=True, color='k', marker='o')

#Dcol_freq['unique_value_to_num_values_99_ratio'].plot(secondary_y=True, color='r', marker='o')

plot1=col_freq.plot(x='col_name',y=['unique_values','num_values_99'],ax=ax1,kind='bar')

for p in plot1.patches[1:]:

    h = p.get_height()

    x = p.get_x()+p.get_width()/2.

    if h != 0:

        plot1.annotate("%g" % p.get_height(), xy=(x,h), xytext=(0,4), rotation=90, 

                   textcoords="offset points", ha="center", va="bottom")

plot1.set(ylabel='Count')

plot= plot.set(title='Data Details  in each D columns')
np.warnings.filterwarnings('ignore')

sns.set(rc={'figure.figsize':(14,16)})

for num, alpha in enumerate(Dcols):

    plt.subplot(5,3,num+1)

    yes = train_transaction[(train_transaction['isFraud'] == 1)][alpha]

    no = train_transaction[(train_transaction['isFraud'] == 0) ][alpha]

    plt.hist(yes[yes>0], alpha=0.75, label='Fraud', color='r')

    plt.hist(no[no>0], alpha=0.25, label='Not Fraud', color='g')

    plt.legend(loc='upper right')

    plt.title('Histogram of values  in column ' + str(alpha) )

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
cards_addr1_Ccolsgroup_details[(cards_addr1_Ccolsgroup_details[['D6','D7']].notnull().all(1)) & (cards_addr1_Ccolsgroup_details.Count >10)].head(30)
pd.options.display.max_colwidth =300

Ccol_details=column_value_details(train_transaction,Ccols,0.965)

num_values_96 =[]

for i in range(len(Ccol_details)):

    num_values_96.append(len(Ccol_details['values_0.96'][i]))

Ccol_details['num_values_96'] = num_values_96

Ccol_details
C_cat = Ccol_details[Ccol_details['num_values_96'] <= 15].reset_index()

sns.set(rc={'figure.figsize':(14,16)})

x=1

for num, alpha in enumerate(C_cat.col_name):

    plt.subplot(len(C_cat),2,x)

    sns.countplot(data=train_transaction[train_transaction[alpha].isin (C_cat['values_0.96'][num])],y=alpha,hue='isFraud')

    plt.legend(loc='lower right',title='is Fraud')

    plt.title('Count of unique values which make 96.5% of data in column ' + str(alpha) )

    plt.subplot(len(C_cat),2,x+1)

    yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (C_cat['values_0.04'][num]))][alpha]

    no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (C_cat['values_0.04'][num]))][alpha]

    plt.hist(yes, alpha=0.75, label='Fraud', color='r')

    plt.hist(no, alpha=0.25, label='Not Fraud', color='g')

    plt.legend(loc='upper right')

    plt.title('Histogram of values which make 3.5% of data in column ' + str(alpha) )

    x= x+2

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
C_num = Ccol_details[Ccol_details['num_values_96'] > 15].reset_index()
sns.set(rc={'figure.figsize':(14,16)})

x=1

for num, alpha in enumerate(C_num.col_name):

    plt.subplot(len(C_num),2,x)

    yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (C_num['values_0.96'][num]))][alpha]

    no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (C_num['values_0.96'][num]))][alpha]

    plt.hist(yes, alpha=0.75, label='yes', color='r')

    plt.hist(no, alpha=0.25, label='no', color='b')

    plt.legend(loc='upper right')

    plt.title('Histogram of values which make 96.5% of data in column ' + str(alpha) )

    plt.subplot(len(C_num),2,x+1)

    yes = train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction[alpha].isin (C_num['values_0.04'][num]))][alpha]

    no = train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction[alpha].isin (C_num['values_0.04'][num]))][alpha]

    plt.hist(yes, alpha=0.75, label='Fraud', color='r')

    plt.hist(no, alpha=0.25, label='Not Fraud', color='b')

    plt.title('Histogram of values which make 3.5% of data in column ' + str(alpha) )

    plt.legend(loc='upper right')

    x= x+2

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
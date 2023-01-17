import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input/ieee-fraud-detection"))
# setting up default plotting parameters

%matplotlib inline



plt.rcParams['figure.figsize'] = [20.0, 5.0]

plt.rcParams.update({'font.size': 22,})



sns.set_palette('viridis')

sns.set_style('white')

sns.set_context('talk', font_scale=0.8)
raw_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')



print('train_transaction shape: ')

print(raw_transaction.shape)



raw_transaction.head()
# using seaborns countplot to show distribution of questions in dataset

fig, ax = plt.subplots()

g = sns.countplot(raw_transaction.isFraud, palette='viridis')

g.set_xticklabels(['Not Fraud', 'Fraud'])

g.set_yticklabels([])



# function to show values on bars

def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.0f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)

show_values_on_bars(ax)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Fraud vs Not Fraud', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
# print percentage of transactions where target == 1

(len(raw_transaction.loc[raw_transaction.isFraud==1])) / (len(raw_transaction.loc[raw_transaction.isFraud == 0])) * 100
#checking for missing values



# from https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb# from  

# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(raw_transaction).head(10)
# info method to get quick description of the data

raw_transaction.info()
# summary of numerical attributes

raw_transaction.describe()
sns.kdeplot(raw_transaction.loc[raw_transaction['isFraud'] == 0,'TransactionDT'], label='isFraud = 0 (Not Fraud)')

sns.kdeplot(raw_transaction.loc[raw_transaction['isFraud'] == 1,'TransactionDT'], label='isFraud = 1 (Fraud)', shade=True)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Transaction Time Delta by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
sns.kdeplot(raw_transaction.loc[raw_transaction['isFraud'] == 0,'TransactionAmt'], label='isFraud = 0 (Not Fraud)')

sns.kdeplot(raw_transaction.loc[raw_transaction['isFraud'] == 1,'TransactionAmt'], label='isFraud = 1 (Fraud)', shade=True)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Transaction Amount by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
# creating new dataframe for manipulated features

transaction = raw_transaction[['isFraud']].copy()

transaction['log_transaction_amt'] = np.log(raw_transaction.TransactionAmt)
sns.kdeplot(transaction.loc[transaction['isFraud'] == 0,'log_transaction_amt'], label='isFraud = 0 (Not Fraud)')

sns.kdeplot(transaction.loc[transaction['isFraud'] == 1,'log_transaction_amt'], label='isFraud = 1 (Fraud)', shade=True)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Log Transaction Amount by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
fig, ax = plt.subplots()

ax.set(yscale='log')

sns.countplot(x='ProductCD', hue='isFraud', data=raw_transaction, palette='viridis')



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Product CD by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
# create card dataset and drop null values

card = raw_transaction[['isFraud', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6']].copy()

card = card.dropna()
# checking unique values

for i in card:

    print ("Unique Values",i, " = ",card[i].nunique())
# plotting card features

sns.kdeplot(card.loc[card['isFraud'] == 0,'card1'], label='isFraud = 0 (Not Fraud)')

sns.kdeplot(card.loc[card['isFraud'] == 1,'card1'], label='isFraud = 1 (Fraud)', shade=True)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Card 1 by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()





sns.kdeplot(card.loc[card['isFraud'] == 0,'card2'], label='isFraud = 0 (Not Fraud)')

sns.kdeplot(card.loc[card['isFraud'] == 1,'card2'], label='isFraud = 1 (Fraud)', shade=True)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Card 2 by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()



sns.kdeplot(card.loc[card['isFraud'] == 0,'card3'], label='isFraud = 0 (Not Fraud)')

sns.kdeplot(card.loc[card['isFraud'] == 1,'card3'], label='isFraud = 1 (Fraud)', shade=True)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Card 3 by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()



sns.kdeplot(card.loc[card['isFraud'] == 0,'card5'], label='isFraud = 0 (Not Fraud)')

sns.kdeplot(card.loc[card['isFraud'] == 1,'card5'], label='isFraud = 1 (Fraud)', shade=True)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Card 5 by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
sns.countplot(x='card4', hue='isFraud', data=card, palette='viridis')



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Card 4 by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()



sns.countplot(x='card6', hue='isFraud', data=card, palette='viridis')



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Card 6 by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
# check unique and missing values for email features

email = raw_transaction[['isFraud','P_emaildomain', 'R_emaildomain']].copy()

for i in email:

    print('Unique Values', i, '=', email[i].nunique())



missing_values_table(email)
# fill missing values

email.P_emaildomain.fillna('none', inplace=True)

email.R_emaildomain.fillna('none', inplace=True)



# check unique and missing values for email features

for i in email:

    print('Unique Values', i, '=', email[i].nunique())



missing_values_table(email)
from collections import Counter



p_email = Counter(email.P_emaildomain)

common_email = [e[0] for e in p_email.most_common(15)]

common_counts = [e[1] for e in p_email.most_common(15)]



sns.barplot(common_email, common_counts, palette='viridis')

sns.despine(left=True, bottom=True)

plt.title("Most Common 'P' E-Mails")

plt.tick_params(axis='x', which='major', labelsize=10)

plt.show()
# keep only most common emails and set all to other

# use isin to check membership in list, ~ to negate, .loc to modify series in place

email.loc[~email['P_emaildomain'].isin(common_email), 'P_emaildomain'] = 'other'

email.P_emaildomain.nunique()
sns.countplot(x='P_emaildomain', hue='isFraud', data=email, palette='viridis')



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('P_emaildomain by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=10)

plt.show()
r_email = Counter(email.R_emaildomain)

common_email = [e[0] for e in r_email.most_common(15)]

common_counts = [e[1] for e in r_email.most_common(15)]



sns.barplot(common_email, common_counts, palette='viridis')

sns.despine(left=True, bottom=True)

plt.title("Most Common 'R' E-Mails")

plt.tick_params(axis='x', which='major', labelsize=10)

plt.show()
# keep only most common emails and set all to other

# use isin to check membership in list, ~ to negate, .loc to modify series in place

email.loc[~email['R_emaildomain'].isin(common_email), 'R_emaildomain'] = 'other'

email.R_emaildomain.nunique()
sns.countplot(x='R_emaildomain', hue='isFraud', data=email, palette='viridis')



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('R_emaildomain by Target Value', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=10)

plt.show()
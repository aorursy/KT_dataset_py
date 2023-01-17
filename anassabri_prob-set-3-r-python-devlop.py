import pandas as pd

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt



%matplotlib inline



acctdf = pd.read_csv("../input/AccountTable.csv")

clientdf = pd.read_csv('../input/ClientTable.csv')

bridgedf = pd.read_csv('../input/ClientAcctFact.csv')



display(acctdf.head())

display(clientdf.head())

display(bridgedf.head())



print("Account table dimensions: ", acctdf.shape)

print("Client table dimensions: ", clientdf.shape)

print('Client-account bridge table dimensions: ', bridgedf.shape)
print(len(np.unique(clientdf.ClientID)))
clientdf.Region.value_counts().reset_index(name='# of Clients')
clientdf['Gender'].value_counts().plot(kind='bar')
temp = acctdf['AccountStatus'].value_counts(normalize=True) * 100

print("The percentage of \'Open' is: %.1f" %  temp.iloc[[0]].values[0],"%")

print("The percentage of \'Close' is: %.1f" % temp.iloc[[1]].values[0],"%")

D_accounts = acctdf.loc[acctdf.AccountType=='D',:]

closed_D_accounts = D_accounts.loc[D_accounts.AccountStatus=='closed',:]

percentage = closed_D_accounts.shape[0]/D_accounts.shape[0]*100

print("The percentage of D accounts that have \'status = Close' is: %.1f" % percentage,"%")



acc_client = pd.merge(acctdf,bridgedf, on='AccountID')

acc_client = pd.merge(acc_client, clientdf, on='ClientID')

acc_client.head()
temp_df = acc_client.groupby('ClientID')['AccountType'].apply(lambda x: ''.join(set(x.sum())))

temp_df = pd.DataFrame({'ClientID':temp_df.index, 'ClientType':temp_df.values})

acc_client_merged = acc_client.merge(temp_df, on = "ClientID")

acc_client_merged.head()

print("There are {} number of unique client types".format(len(np.unique(acc_client_merged.ClientType))))
open_all_types = acc_client_merged

open_all_types['Type_length'] = acc_client_merged['ClientType'].apply(len)

open_all_types_filtered = open_all_types[(open_all_types['Type_length']==3) & (open_all_types['AccountStatus'] =='open')]

open_all_types_filtered = open_all_types_filtered.drop('Type_length',axis = 1)

print("The {} number of clients have all three types of accounts with \'account status = open'".format(open_all_types_filtered.shape[0]))

print("The answer is correct with 1.0 probability since we have filtered data for the clients who have opened account of all types (D,L,W)")
import warnings

warnings.filterwarnings('ignore')

temp_df = acc_client.groupby('ClientID')['AccountType'].apply(lambda x: ''.join(x.sum()))

temp_df = pd.DataFrame({'ClientID':temp_df.index, 'ClientType':temp_df.values})

acc_client_merged = acc_client.merge(temp_df, on = "ClientID")

open_all_types_filtered = acc_client_merged[(acc_client_merged['AccountType']=='W') & (acc_client_merged['AccountStatus'] =='open')]

open_all_types_filtered['L_count'] = open_all_types_filtered.ClientType.apply(lambda x: x.count('L'))

open_all_types_filtered_L2 = open_all_types_filtered[open_all_types_filtered.L_count >= 2]

average = open_all_types_filtered_L2.AccountBalance.mean()

print("The average of open W account for clients having at least two open L accounts is: %.2f" % average)

print("The answer is correct with 1.0 probability since we have filtered data for the clients who have open account of W type  as well as at least two L accounts")





temp_df = acc_client.groupby('ClientID')['AccountType'].apply(lambda x: ''.join(set(x.sum())))

temp_df = pd.DataFrame({'ClientID':temp_df.index, 'ClientType':temp_df.values})

acc_client_merged = acc_client.merge(temp_df, on = "ClientID")

open_all_types = acc_client_merged

open_all_types['Type_length'] = acc_client_merged['ClientType'].apply(len)

t_test_filtered = open_all_types[['AccountBalance','Type_length']]
ones = t_test_filtered[t_test_filtered.Type_length==1].shape[0]

twos = t_test_filtered[t_test_filtered.Type_length==2].shape[0]

threes = t_test_filtered[t_test_filtered.Type_length==3].shape[0]

count_list = [ones,twos,threes]

avg_ones = t_test_filtered[t_test_filtered.Type_length==1].AccountBalance.mean()

avg_twos = t_test_filtered[t_test_filtered.Type_length==2].AccountBalance.mean()

avg_threes = t_test_filtered[t_test_filtered.Type_length==3].AccountBalance.mean()

avg_per_account = [avg_ones,avg_twos,avg_threes]

num_product = [1,2,3]
avg_df = pd.DataFrame({'ProductsPurchased':num_product, '# of Clients': count_list,'PerAccountAverageBalance':avg_per_account})
display(avg_df)
print('The table above shows that the hypothesis depict the results as under\n\n'

     '1. The clients who have purchased all three types of products have highest average deposit amount\n'

     '2. The clients with two types of products have lower average deposit amount than those who have purchased three products\n'

     '3. The lowest average deposit amount is for those who have purchased only one type of product\n\n\n'

      'It can hence be interpreted by the results, that the hypothesis is true i.e the clients \nwith more product types have more deposit in bank')

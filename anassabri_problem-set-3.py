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
#len(np.unique(clientdf['ClientID']))

len(clientdf)
clientdf[['ClientID','Region']].groupby('Region').size()
clientdf[['ClientID','Gender']].groupby(['Gender']).count().plot.bar()
acctdf['AccountStatus'].value_counts(normalize=True) * 100
len(acctdf[(acctdf['AccountStatus']=='closed') & (acctdf['AccountType']=='D')])/len(acctdf[acctdf['AccountType']=='D'])*100
temp = pd.merge(acctdf,bridgedf,how='inner',left_on='AccountID',right_on='AccountID')

df = pd.merge(temp,clientdf,how='inner',left_on='ClientID',right_on='ClientID')

df.head()
df_clienttype = df.groupby('ClientID').AccountType.unique().reset_index()

df_clienttype.columns = ['ClientID','ClientType']

final_df = pd.merge(df,df_clienttype,how='inner',left_on='ClientID',right_on='ClientID')

final_df['ClientType'] = final_df.apply(lambda row: ''.join(str(row['ClientType']).replace('[','').replace(']','').split(',')),axis=1)

final_df.head()
len(np.unique(final_df['ClientType']))
len(np.unique(final_df[(final_df['AccountStatus']=='open') & (final_df['ClientType']=="'D' 'L' 'W'")].ClientID))
temp = final_df[['ClientID','AccountType']].groupby('ClientID').AccountType.value_counts().rename('cnt').reset_index()

temp = temp.merge(final_df[['ClientID','AccountStatus']],how='inner',left_on='ClientID',right_on='ClientID')

temp['Cl_ID'] = temp.apply(lambda row:row['ClientID'] if row['cnt']>=2 and row['AccountType']=='L' and row['AccountStatus']=='open' else 'NA',axis=1)



unique_client_ids = temp['Cl_ID'].unique()

unique_client_ids = unique_client_ids[unique_client_ids!='NA']



final_df[(final_df['ClientID'].isin(unique_client_ids)) & (final_df['AccountType']=='W') & (final_df['AccountStatus']=='open')].groupby('ClientID').agg({'AccountBalance':'mean'})



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
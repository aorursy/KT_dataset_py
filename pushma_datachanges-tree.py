# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/standard-bank-tech-impact-challenge/Train (1).csv', sep=',')

df.head()
dftest = pd.read_csv('../input/standard-bank-tech-impact-challenge/Test (1).csv', sep=',')

dftest.head()
df11 = df

unlinked = pd.read_csv('../input/standard-bank-tech-impact-challenge/unlinked_masked_final.csv', sep=',')

#df11['CustomerId'].value_counts()

df11.head()
'''for i in range (df11.shape[0]):

    if df11.at[i,'Count_trans_of_cust'] != 0:

        print('ok'+str(i))'''
dftest11 = dftest

#df11['CustomerId'].value_counts()

dftest11.head()
df11 = df11.drop(["AmountLoan","Currency","PaidOnDate","IsFinalPayBack","DueDate","PayBackId","IsThirdPartyConfirmed"], axis=1)

df11.head()
df1 = df11

df1 = df1.drop(['CurrencyCode', 'CountryCode', 'ProviderId'], axis=1)

dftest1 = dftest11

dftest1 = dftest1.drop(['CurrencyCode', 'CountryCode', 'ProviderId'], axis=1)
dftest1.head()
tr_st = df1['TransactionStatus']

test_tr_st = dftest1['TransactionStatus']
df1['Cust_id'] = df1['CustomerId'].map(lambda s: s[len("CustomerId_"):]).astype(int)

dftest1['Cust_id'] = dftest1['CustomerId'].map(lambda s: s[len("CustomerId_"):]).astype(int)
df1.head()
df2 = df1

df2['Trans_id'] = df2['TransactionId'].map(lambda s: s[len("TransactionId_"):]).astype(int)

dftest2 = dftest1

dftest2['Trans_id'] = dftest2['TransactionId'].map(lambda s: s[len("TransactionId_"):]).astype(int)

df2.head()
df2['Batch_id'] = df2['BatchId'].map(lambda s: s[len("BatchId_"):]).astype(int)

df2['Subscr_id'] = df2['SubscriptionId'].map(lambda s: s[len("SubscriptionId_"):]).astype(int)

df2['Prod_id'] = df2['ProductId'].map(lambda s: s[len("ProductId_"):]).astype(int)

df2['Chan_id'] = df2['ChannelId'].map(lambda s: s[len("ChannelId_"):]).astype(int)

dftest2['Batch_id'] = dftest2['BatchId'].map(lambda s: s[len("BatchId_"):]).astype(int)

dftest2['Subscr_id'] = dftest2['SubscriptionId'].map(lambda s: s[len("SubscriptionId_"):]).astype(int)

dftest2['Prod_id'] = dftest2['ProductId'].map(lambda s: s[len("ProductId_"):]).astype(int)

dftest2['Chan_id'] = dftest2['ChannelId'].map(lambda s: s[len("ChannelId_"):]).astype(int)

df2.head()
df2 = df2.drop(['CustomerId', 'TransactionId', 'BatchId', 'SubscriptionId', 'ChannelId'], axis=1)

dftest2 = dftest2.drop(['CustomerId', 'TransactionId', 'BatchId', 'SubscriptionId', 'ChannelId'], axis=1)

df2.head()
#df2 = df2.drop(['ProductId'], axis=1)

df2.head(30)
df22 = df2

for i in range(df22.shape[0]):

    if df22.at[i,'TransactionStatus'] == 1:

        df22.at[i,'LoanId'] = int(df22.at[i,'LoanId'][len("LoanId_"):])

        df22.at[i,'InvestorId'] = int(df22.at[i,'InvestorId'][len("InvestorId_"):])

        if df22.at[i,'LoanApplicationId'] != df22.at[i,'LoanApplicationId']:

            df22.at[i,'LoanApplicationId'] = 0

        else: df22.at[i,'LoanApplicationId'] = int(df22.at[i,'LoanApplicationId'][len("LoanApplicationId_"):])

        if df22.at[i,'ThirdPartyId'] != df22.at[i,'ThirdPartyId']:

            df22.at[i,'ThirdPartyId'] = 0

        else: df22.at[i,'ThirdPartyId'] = int(df22.at[i,'ThirdPartyId'][len("ThirdPartyId_"):])

    else:

        df22.at[i,'LoanId'] = 0

        df22.at[i,'InvestorId'] = 0

        df22.at[i,'LoanApplicationId'] = 0

        df22.at[i,'ThirdPartyId'] = 0



dftest22 = dftest2

for i in range(dftest22.shape[0]):

    if dftest22.at[i,'TransactionStatus'] == 1:

        dftest22.at[i,'LoanId'] = int(dftest22.at[i,'LoanId'][len("LoanId_"):])

        dftest22.at[i,'InvestorId'] = int(dftest22.at[i,'InvestorId'][len("InvestorId_"):])

        if dftest22.at[i,'LoanApplicationId'] != dftest22.at[i,'LoanApplicationId']:

            dftest22.at[i,'LoanApplicationId'] = 0

        else: dftest22.at[i,'LoanApplicationId'] = int(dftest22.at[i,'LoanApplicationId'][len("LoanApplicationId_"):])

        if dftest22.at[i,'ThirdPartyId'] != dftest22.at[i,'ThirdPartyId']:

            dftest22.at[i,'ThirdPartyId'] = 0

        else: dftest22.at[i,'ThirdPartyId'] = int(dftest22.at[i,'ThirdPartyId'][len("ThirdPartyId_"):])

    else:

        dftest22.at[i,'LoanId'] = 0

        dftest22.at[i,'InvestorId'] = 0

        dftest22.at[i,'LoanApplicationId'] = 0

        dftest22.at[i,'ThirdPartyId'] = 0
df22.head(40)
df22.info()
dftest2.info()
df222 = df22

df222['LoanId'] =  df222['LoanId'].astype(int) 

df222['InvestorId'] =  df222['InvestorId'].astype(int)

df222['LoanApplicationId'] =  df222['LoanApplicationId'].astype(int)

df222['ThirdPartyId'] =  df222['ThirdPartyId'].astype(int)

df222.info()



dftest222 = dftest22

dftest222['LoanId'] =  dftest222['LoanId'].astype(int) 

dftest222['InvestorId'] =  dftest222['InvestorId'].astype(int)

dftest222['LoanApplicationId'] =  dftest222['LoanApplicationId'].astype(int)

dftest222['ThirdPartyId'] =  dftest222['ThirdPartyId'].astype(int)
df3 = df222

df3 = pd.get_dummies(df3, columns=['ProductCategory'])



dftest3 = dftest222

dftest3 = pd.get_dummies(dftest3, columns=['ProductCategory'])

df3.head()
df3['amount equal value'] = (np.abs(df3['Amount']) == df3['Value']).astype(int)

df3.head()



dftest3['amount equal value'] = (np.abs(dftest3['Amount']) == dftest3['Value']).astype(int)

#df2['Inve_id'] = df2['InvestorId'].map(lambda s: s[len("InvestorId_"):]).astype(int)

#df2['Loan_id'] = df2['LoanId'].map(lambda s: s[len("LoanId_"):]).astype(int)

#df2['LoanApp_id'] = df2['LoanApplicationId'].map(lambda s: s[len("LoanApplicationId_"):]).astype(int)

#df2 = df2.head()
df3 = df3.drop(['Amount'], axis=1)



dftest3 = dftest3.drop(['Amount'], axis=1)

df3.head()
df4 = df3

df4 = df4.drop(['IssuedDateLoan'], axis=1)



dftest4 = dftest3

dftest4 = dftest4.drop(['IssuedDateLoan'], axis=1)

df4.head()
data = df4['TransactionStartTime']



df5 = df4

df5['trans_mounth'] = df5['TransactionStartTime'].map(lambda s: s[5:7]).astype(int)

df5['is_day?'] = df5['TransactionStartTime'].map(lambda s: 0 if(s[11:13]>'18' or s[11:13]<'06') else 1).astype(int)

df5 = df5.drop(['TransactionStartTime'], axis=1)

df5.head()



dftest5 = dftest4

dftest5['trans_mounth'] = dftest5['TransactionStartTime'].map(lambda s: s[5:7]).astype(int)

dftest5['is_day?'] = dftest5['TransactionStartTime'].map(lambda s: 0 if(s[11:13]>'18' or s[11:13]<'06') else 1).astype(int)

dftest5 = dftest5.drop(['TransactionStartTime'], axis=1)
df5 = df5.drop(['ProductId'], axis=1)

dftest5 = dftest5.drop(['ProductId'], axis=1)
df5.head()
df5.info()
dftest5.info()
df51 = df5.copy()

df51['Value'] = df51['Value'].astype(int)

df52 = df5.copy()

df52['Value'] = df52['Value'].astype(int)

for i in range(df51.shape[0]):

    if df51.at[i,'TransactionStatus'] == 0:

        df51.at[i,'IsDefaulted'] = 0

df51['IsDefaulted'] = df51['IsDefaulted'].astype(int)

for i in range(df52.shape[0]):

    if df52.at[i,'TransactionStatus'] == 0:

        df52.at[i,'IsDefaulted'] = 1

df52['IsDefaulted'] = df52['IsDefaulted'].astype(int)



#dftest5['Value'] = dftest5['Value'].astype(int)
dftest5.head()
df52.head()

df52 = df52.drop(['ThirdPartyId', 'LoanApplicationId', 'InvestorId', 'LoanId'], axis=1)

dftest5 = dftest5.drop(['ThirdPartyId', 'LoanApplicationId', 'InvestorId', 'LoanId'], axis=1)
df52.head()

#print(df52['Value'])

for i in range(df52.shape[0]):

    #print(df52.at[i,'Value'])

    if df52.at[i,'Value'] <= 1000:

        df52.at[i,'Size_of_value'] = 'Small'

    elif df52.at[i,'Value'] > 1000 and df52.at[i,'Value'] <= 10000:

        df52.at[i,'Size_of_value'] = 'Middle'

    else: df52.at[i,'Size_of_value'] = 'Big'

df52 = df52.drop(['Value'], axis=1)

df52 = pd.get_dummies(df52, columns=['Size_of_value'])



for i in range(dftest5.shape[0]):

    if dftest5.at[i,'Value'] <= 1000:

        dftest5.at[i,'Size_of_value'] = 'Small'

    elif dftest5.at[i,'Value'] > 1000 and dftest5.at[i,'Value'] <= 10000:

        dftest5.at[i,'Size_of_value'] = 'Middle'

    else: dftest5.at[i,'Size_of_value'] = 'Big'

dftest5 = dftest5.drop(['Value'], axis=1)

dftest5 = pd.get_dummies(dftest5, columns=['Size_of_value'])

df52['TransactionStartTime'] = data

df52['trans_day'] = df52['TransactionStartTime'].map(lambda s: s[8:10]).astype(int)

df52 = df52.drop(['TransactionStartTime'], axis=1)

df52.head()



dftest5['TransactionStartTime'] = data

dftest5['trans_day'] = dftest5['TransactionStartTime'].map(lambda s: s[8:10]).astype(int)

dftest5 = dftest5.drop(['TransactionStartTime'], axis=1)
for i in range(df52.shape[0]):

    if df52.at[i,'trans_day'] <= 10:

        df52.at[i,'period_of_mounth'] = 'Start'

    elif df52.at[i,'trans_day'] <= 20:

        df52.at[i,'period_of_mounth'] = 'Middle'

    else: df52.at[i,'period_of_mounth'] = 'End'

df52 = df52.drop(['trans_day'], axis=1)

df52 = pd.get_dummies(df52, columns=['period_of_mounth'])

df52.head(10)



for i in range(dftest5.shape[0]):

    if dftest5.at[i,'trans_day'] <= 10:

        dftest5.at[i,'period_of_mounth'] = 'Start'

    elif dftest5.at[i,'trans_day'] <= 20:

        dftest5.at[i,'period_of_mounth'] = 'Middle'

    else: dftest5.at[i,'period_of_mounth'] = 'End'

dftest5 = dftest5.drop(['trans_day'], axis=1)

dftest5 = pd.get_dummies(dftest5, columns=['period_of_mounth'])
#df52 = df52.drop(['Batch_id', 'Subscr_id', 'Prod_id', 'Chan_id'], axis=1)

#dftest5 = dftest5.drop(['Batch_id', 'Subscr_id', 'Prod_id', 'Chan_id'], axis=1)

dftest5 = dftest5.drop(['ProductCategory_ticket'], axis=1)

for i in range(df52.shape[0]):

    if df52.at[i,'TransactionStatus'] == 0:

        df52 = df52.drop([i])

for i in range(dftest5.shape[0]):

    if dftest5.at[i,'TransactionStatus'] == 0:

        dftest5 = dftest5.drop([i])

df52 = df52.drop(['TransactionStatus'], axis=1)

dftest5 = dftest5.drop(['TransactionStatus'], axis=1)
df52.head()
dftest5.head()
#'''for i in range(df52.shape[0]):

#    df52.iloc[i].at['Number_of_transactions'] = pd.Series(df52.iloc[i].at['Cust_id']).value_counts()'''

#df52['Number_of_transactions'] = df52['Cust_id'].value_counts()

df52['Number_of_transactions'] = df52.groupby(['Cust_id'])['Cust_id'].transform('count')

#print(pd.Series(df52.iloc[0].at['Cust_id']).value_counts())

df52.head(10)

dftest5['Number_of_transactions'] = dftest5.groupby(['Cust_id'])['Cust_id'].transform('count')
df52.to_csv('df_train_new_new.csv',index=False)

dftest5.to_csv('df_test_new_new.csv',index=False)
df52.info()
df52['TransactionStatus'] = tr_st

df52['%_of_succesfull_transactions'] = float(0)

a = []

for i in range(df52.shape[0]):

    k = 0

    if df52.iloc[i].at['TransactionStatus'] == 1:

        k += 1

    #print(k/df52.iloc[i].at['Number_of_transactions'])

    a.append(k/df52.iloc[i].at['Number_of_transactions'])

    #print(a[i])

df52['%_of_succesfull_transactions'] = a

df52.head()

dftest5['TransactionStatus'] = test_tr_st

dftest5['%_of_succesfull_transactions'] = float(0)

a = []

for i in range(dftest5.shape[0]):

    k = 0

    if dftest5.iloc[i].at['TransactionStatus'] == 1:

        k += 1

    #print(k/df52.iloc[i].at['Number_of_transactions'])

    a.append(k/dftest5.iloc[i].at['Number_of_transactions'])

    #print(a[i])



dftest5['%_of_succesfull_transactions'] = a

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3, random_state=19)

X = df52.drop(['IsDefaulted'], axis=1)

y = df52['IsDefaulted']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=19)
tree.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV, cross_val_score



tree_params = {'max_depth': range(2, 15)}



tree_grid = GridSearchCV(tree, tree_params,

                         cv=5, n_jobs=-1, verbose=True)



tree_grid.fit(X_train, y_train)

best_tree = tree_grid.best_estimator_

X_TESTING = dftest5

X_TESTING.info()

y_TESTING = best_tree.predict(X_TESTING)
dfans2 = dftest[['TransactionId', 'TransactionStatus']]

#dfans2['IsDefaulted'] = y_TESTING

for i in range(dfans2.shape[0]):

    if dfans2.at[i,'TransactionStatus'] == 0:

        dfans2 = dfans2.drop([i])

dfans2['IsDefaulted'] = y_TESTING

dfans2 = dfans2.drop(['TransactionStatus'], axis=1)

print(dfans2['IsDefaulted'])

dfans2.to_csv('Answer19.csv',index=False)
pd.DataFrame(tree_grid.cv_results_).T

import matplotlib.pyplot as plt



df_cv = pd.DataFrame(tree_grid.cv_results_)



plt.plot(df_cv['param_max_depth'], df_cv['mean_test_score'])

plt.xlabel("max_depth")

plt.ylabel("accuracy");
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, y_pred)
from sklearn.metrics import f1_score

f1_score(y_valid, y_pred)
import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns
df=pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')

print(df.head())
df.isnull().values.any()
transfer=df.loc[(df.isFraud==1) & (df.type=='TRANSFER')]

cash_out=df.loc[(df.isFraud==1) & (df.type=='CASH_OUT')]

print('Fraudulent Transfers',len(transfer))

print('Fraudulent Cash Outs',len(cash_out))
print(df.loc[df.isFlaggedFraud==1].type.drop_duplicates())
print("Transfers where isFraud is set",len(transfer))

print("Tranfers where isFlaggedFraud set",len(df.loc[df.isFlaggedFraud==1]))
dftype=df.loc[df.type=='TRANSFER']

flag_set=dftype.loc[dftype.isFlaggedFraud==1]

flag_not_set=dftype.loc[dftype.isFlaggedFraud==0]

amt_flag_unset=flag_not_set.amount.max()

print("Max amount transfered when isFlaggedFraud is not set",amt_flag_unset)
old_balance=dftype.loc[dftype.isFlaggedFraud==1].oldbalanceDest.drop_duplicates()

new_balance=dftype.loc[dftype.isFlaggedFraud==1].newbalanceDest.drop_duplicates()

print(old_balance)

print(new_balance)
print(len(dftype.loc[(dftype.isFlaggedFraud==0) & (dftype.oldbalanceDest==dftype.newbalanceDest)]))
print(flag_set.oldbalanceOrg.min(),flag_set.oldbalanceOrg.max())

print(flag_not_set.loc[flag_not_set.oldbalanceOrg==flag_not_set.newbalanceOrig].oldbalanceOrg.min(),flag_not_set.loc[flag_not_set.oldbalanceOrg==flag_not_set.newbalanceOrig].oldbalanceOrg.max())
print("isFlaggedFraud is set and no duplicates",len(df.loc[df.isFlaggedFraud==1].nameOrig.drop_duplicates()))

print("Number of duplicates when isFlaggedFraud is not set",len(df.loc[df.isFlaggedFraud==0].nameOrig)-len(df.loc[df.isFlaggedFraud==0].nameOrig.drop_duplicates()))
print("Cash in transactions where merchant pays")

print(df.loc[df.type=='CASH_IN'].nameOrig.str.contains('M').any())
print("Cash out transactions where the merchant is paid")

print(df.loc[df.type=='CASH_OUT'].nameDest.str.contains('M').any())
print("Are there merchants in originator accounts",df.nameOrig.str.contains('M').any())
print("Are there merchants in destination accounts except Payment",df.loc[df.type!='PAYMENT'].nameDest.str.contains('M').any())
fraud=transfer.nameDest.isin(cash_out.nameOrig).any()

print(fraud)
notfraud=df.loc[df.isFraud==0]

print(transfer.loc[transfer.nameDest.isin(notfraud.loc[notfraud.type=='CASH_OUT'].nameOrig.drop_duplicates())])
print("fraudulent tranfer to C423543548 occurs at step 486 and cashout from this account occurs at step",notfraud.loc[(notfraud.type=='CASH_OUT') & (notfraud.nameOrig=="C423543548")].step.values)
X=df.loc[(df.type=='TRANSFER') | (df.type=='CASH_OUT')]

X=X.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1)

Y=X['isFraud']
X.loc[X.type=='TRANSFER','type']=0

X.loc[X.type=='CASH_OUT','type']=1

X.type=X.type.astype(int)
X.head()
Xfraud=X.loc[Y==1]

Xnonfraud=X.loc[Y==0]
print("The fraction of fraudulent transactions with oldbalanceDest= newbalanceDest= 0 although the transacted amount is non-zero is:\n",len(Xfraud.loc[(Xfraud.oldbalanceDest==0) & (Xfraud.newbalanceDest==0)])/len(Xfraud))
print("The fraction of genuine transactions with oldbalanceDest= newbalanceDest= 0 although the transacted amount is non-zero is:\n",len(Xnonfraud.loc[(X.oldbalanceDest==0) & (X.newbalanceDest==0)])/len(Xnonfraud))
print("The fraction of fraudulent transactions with oldbalanceOrg= newbalanceOrig= 0 although the transacted amount is non-zero is:\n",len(Xfraud.loc[(Xfraud.oldbalanceOrg==0) & (Xfraud.newbalanceOrig==0)])/len(Xfraud))
print("The fraction of genuine transactions with oldbalanceOrg= newbalanceOrig= 0 although the transacted amount is non-zero is:\n",len(Xnonfraud.loc[(Xnonfraud.oldbalanceOrg==0) & (Xnonfraud.newbalanceOrig==0)])/len(Xnonfraud))
X.loc[(X.oldbalanceDest==0) & (X.newbalanceDest==0) & (X.amount!=0),['oldbalanceDest','newbalanceDest']]=-1

X.head()
X.loc[(X.oldbalanceOrg==0) & (X.newbalanceOrig==0) & (X.amount!=0),['oldbalanceOrg','newbalanceOrig']]=np.nan
X['errorBalOrig']=X.oldbalanceOrg+X.amount-X.newbalanceOrig

X['errorBalDest']=X.oldbalanceDest+X.amount-X.newbalanceDest
sns.boxplot(x='isFraud',y='step',data=X)
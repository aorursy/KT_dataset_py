import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
dataset_train=pd.read_csv('../input/application_train.csv')
dataset_bureau=pd.read_csv('../input/bureau.csv')
dataset_bureau_bal=pd.read_csv('../input/bureau_balance.csv')
dataset_credit_card_balance=pd.read_csv('../input/credit_card_balance.csv')
dataset_installments_payments=pd.read_csv('../input/installments_payments.csv')
dataset_poscash=pd.read_csv('../input/POS_CASH_balance.csv')
dataset_previous_application=pd.read_csv('../input/previous_application.csv')

import matplotlib.pyplot as plt


#importing test dataset
dataset_test=pd.read_csv('../input/application_test.csv')
x=dataset_train.loc[:,]
#x.value_counts(normalize=False, sort=True, ascending=False, bins=None)
p=x.columns[x.isna().any()].tolist()
for col in p:
        if (x[col].dtype == 'object'): # for object type
            a=x[col].value_counts().idxmax()
            x[col]=x[col].replace(np.nan,a)
        elif (x[col].dtype == 'int64'):
            x[col]=x[col].fillna(x[col].mean())
        else:
            x[col]=x[col].fillna(x[col].mean())
#x.head()

#x[col].value_counts()


y_train=x.loc[:, ['TARGET']]
x_train=x.loc[:,['SK_ID_CURR','TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','NAME_TYPE_SUITE','OCCUPATION_TYPE','NAME_INCOME_TYPE','CNT_FAM_MEMBERS','ORGANIZATION_TYPE']]
#bureau replace nan data
x_bureau=dataset_bureau.iloc[:,]

p_bureau=x_bureau.columns[x_bureau.isna().any()].tolist()
for col in p_bureau:
        if (x_bureau[col].dtype == 'object'): # for object type
            a_bureau=x_bureau[col].value_counts().idxmax()
            x_bureau[col]=x_bureau[col].replace(np.nan,a_bureau)
        elif (x_bureau[col].dtype == 'int64'):
            x_bureau[col]=x_bureau[col].fillna(x_bureau[col].mean())
        else:
            x_bureau[col]=x_bureau[col].fillna(x_bureau[col].mean())

#feature set of bureau
x_train_bureau=x_bureau.loc[:,['SK_ID_CURR','SK_ID_BUREAU','AMT_ANNUITY','CREDIT_ACTIVE','AMT_CREDIT_SUM','CREDIT_TYPE','AMT_CREDIT_SUM_DEBT']]
x_test_bureau=x_bureau.loc[:,['SK_ID_CURR','AMT_ANNUITY','CREDIT_ACTIVE','AMT_CREDIT_SUM','CREDIT_TYPE','AMT_CREDIT_SUM_DEBT']]
#replace nan value of bureau_balance
x_bureau_bal=dataset_bureau_bal.iloc[0:400000,]

p_bureau_bal=x_bureau_bal.columns[x_bureau_bal.isna().any()].tolist()
for col in p_bureau_bal:
        if (x_bureau_bal[col].dtype == 'object'): # for object type
            a_bureau_bal=x_bureau_bal[col].value_counts().idxmax()
            x_bureau_bal[col]=x_bureau_bal[col].replace(np.nan,a_bureau_bal)
        elif (x_bureau_bal[col].dtype == 'int64'):
            x_bureau_bal[col]=x_bureau_bal[col].fillna(x_bureau_bal[col].mean())
        else:
            x_bureau_bal[col]=x_bureau_bal[col].fillna(x_bureau_bal[col].mean())
#x.head()
x_bureau_bal.head()
x_bureau_merge=pd.merge(x_train_bureau,x_bureau_bal,on='SK_ID_BUREAU')
x_bureau_merge.head()
x_train2=pd.merge(x_train,x_bureau_merge,on='SK_ID_CURR')
x_train2.head()
from sklearn import preprocessing
temp=0
p=[]
for x in x_train2 :
    if (x !='SK_ID_CURR' or x !='TARGET' or x !='SK_ID_BUREAU' ): 
        if(x_train2[x].dtype =='int64' or x_train2[x].dtype == 'uint8' or x_train2[x].dtype =='float64' ) :
            std_scale = preprocessing.StandardScaler().fit(x_train2[[x]])
            df_std = std_scale.transform(x_train2[[x]])
            minmax_scale = preprocessing.MinMaxScaler().fit(x_train2[[x]])
            df_minmax = minmax_scale.transform(x_train2[[x]])
            i=pd.DataFrame(df_minmax, columns=[[x]])
            p.append(i)
            x_train2[x]=p[temp]
            temp=temp+1
        else :
            x_train2= pd.get_dummies(x_train2 , columns=[x])
            

#drop id column
x_train2.drop(['SK_ID_CURR'], axis = 1, inplace = True)
x_train2.drop(['SK_ID_BUREAU'], axis = 1, inplace = True)
x_train2.head()
p_test=dataset_test.columns[dataset_test.isna().any()].tolist()
for col in p_test:
        if (dataset_test[col].dtype == 'object'): # for object type
            a_test=dataset_test[col].value_counts().idxmax()
            dataset_test[col]=dataset_test[col].replace(np.nan,a_test)
        elif (dataset_test[col].dtype == 'int64'):
            dataset_test[col]=dataset_test[col].fillna(dataset_test[col].mean())
        else:
            dataset_test[col]=dataset_test[col].fillna(dataset_test[col].mean())
#x.head()
#x_test2=pd.merge(x_test,x_train_bureau,on='SK_ID_CURR')
#x_test2.head() 
x_test=dataset_test.loc[:,['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','NAME_TYPE_SUITE','OCCUPATION_TYPE','NAME_INCOME_TYPE','CNT_FAM_MEMBERS','ORGANIZATION_TYPE']]
x_test2=pd.merge(x_test,x_bureau_merge,on='SK_ID_CURR')
x_test2.head() 

from sklearn import preprocessing
temp=0
p=[]
for x in x_test2 :
    if (x !='SK_ID_CURR'): 
        if(x_test2[x].dtype =='int64' or x_test2[x].dtype == 'uint8' or x_test2[x].dtype =='float64' ) :
            std_scale = preprocessing.StandardScaler().fit(x_test2[[x]])
            df_std = std_scale.transform(x_test2[[x]])
            minmax_scale = preprocessing.MinMaxScaler().fit(x_test2[[x]])
            df_minmax = minmax_scale.transform(x_test2[[x]])
            i=pd.DataFrame(df_minmax, columns=[[x]])
            p.append(i)
            x_test2[x]=p[temp]
            temp=temp+1
        else :
            x_test2= pd.get_dummies(x_test2 , columns=[x])

x_test2=x_test2.drop(['SK_ID_CURR'],axis=1)
x_test2.head()   

T=x_train2['TARGET']
x_train2=x_train2.drop(['TARGET'],axis=1)
x=x_train2.columns.difference(x_test2.columns)
for y in x :
    for z in x_train2 :
        if(z == y and z != 'TARGET') :
            idx = x_train2.columns.get_loc(z)
            new_col = 0
            x_test2.insert(loc=idx,column=z,value=new_col)
            #print(z, merge_train.columns.get_loc(z))






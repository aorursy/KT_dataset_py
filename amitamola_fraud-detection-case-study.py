import numpy as np

import pandas as pd
train_df=pd.read_csv('/kaggle/input/fraud-prediction/train.csv')

test_df=pd.read_csv('/kaggle/input/fraud-prediction/test.csv')

df_desc=pd.read_csv('/kaggle/input/fraud-prediction/column_Desc.csv')
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



import datetime
def uniquelo(df):

    col=[]

    vals=[]

    for a in df.columns:

        col.append(a)

        vals.append(len(df[a].unique()))

    percent_missing = df.isnull().sum() * 100 / len(df)

    d = {'Unique_val': vals,'percent_missing': percent_missing, 'Data_Types' :df.dtypes.values}

    return pd.DataFrame(data=d)
out=uniquelo(train_df)

# out.sort_values('percent_missing', inplace=True,ascending=False)

out['Column Info']=df_desc.Description.values

out
train_df['BIRTHDT'] = pd.to_datetime(train_df['BIRTHDT'], format='%Y-%m-%d')

l = [(datetime.datetime.now()-i).days/365 for i in train_df['BIRTHDT']]

l = [int(i) if np.isnan(i)==False else np.nan for i in l]

train_df['AGE'] = l
test_df['BIRTHDT'] = pd.to_datetime(test_df['BIRTHDT'], format='%Y-%m-%d')

l = [(datetime.datetime.now()-i).days/365 for i in test_df['BIRTHDT']]

l = [int(i) if np.isnan(i)==False else np.nan for i in l]

test_df['AGE'] = l
train_df.drop(['BIRTHDT','REPORTEDDT','LOSSDT','N_PAYTO_NAME_cleaned_root','Prov_Name_All_final_root','INSUREDNA_cleaned_root',

                   'N_REFRING_PHYS_final_root','CLMNT_NA_cleaned_root','N_PRVDR_NAME_NONPHYS_cleaned_root'],axis=1,inplace=True)



test_df.drop(['BIRTHDT','REPORTEDDT','LOSSDT','N_PAYTO_NAME_cleaned_root','Prov_Name_All_final_root','INSUREDNA_cleaned_root',

                   'N_REFRING_PHYS_final_root','CLMNT_NA_cleaned_root','N_PRVDR_NAME_NONPHYS_cleaned_root'],axis=1,inplace=True)
uniquelo(train_df)
for a in train_df.columns:

    if (train_df[a].dtype!='object') and (a!='TARGET'):

        train_df[a].fillna(train_df[a].mean(),inplace=True)

        test_df[a].fillna(train_df[a].mean(),inplace=True)

        

    elif (a=='TARGET') or (a=='CLAIMNO'):

        continue

    

    else:

        train_df[a].fillna(train_df[a].mode()[0],inplace=True)

        test_df[a].fillna(train_df[a].mode()[0],inplace=True)
y = train_df['TARGET']

train_df.drop(['TARGET'],axis=1,inplace=True)
x = pd.DataFrame()

test = pd.DataFrame()



for i in train_df.columns:

    if (train_df[i].dtype!='object') or (i=='CLAIMNO'):

        continue

    

    else:

        train_dummy = pd.get_dummies(train_df[i])

        test_dummy = pd.get_dummies(test_df[i])

#         if(a.columns[0] in (b.columns)):

#             b.drop(a.columns[0],axis=1,inplace=True)

#         a.drop(a.columns[0],axis=1,inplace=True)

        missing_test = set(train_dummy.columns)-set(test_dummy.columns)

#         train_miss = set(b.columns)-set(a.columns)



        for k in missing_test:

            test_dummy[k] = 0

            

        for p in train_dummy.columns:

            #print(j)

            x[i+'_'+str(p)] = train_dummy[p]

            test[i+'_'+str(p)] = test_dummy[p]
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn import model_selection
clf = RandomForestClassifier(random_state=42,verbose=3, n_estimators=50)
model = BaggingClassifier(base_estimator = clf, n_estimators = 10, n_jobs=-1, verbose=3, random_state = 240)
model.fit(x,y)
l = model.predict_proba(test)[:,1]



claim_no = test_df['CLAIMNO']



output = pd.DataFrame(columns=['CLAIMNO','TARGET'])



output['CLAIMNO'] = claim_no



output['TARGET'] = l



output.to_csv('For_Kernel3.csv',index=False)



output
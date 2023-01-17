import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier
path='../input/av-healthcare-analytics-ii/healthcare'

train_orig=pd.read_csv(os.path.join(path,'train_data.csv'))

test_orig=pd.read_csv(os.path.join(path,'test_data.csv'))

subm=pd.read_csv(os.path.join(path,'sample_sub.csv'))
for col in test_orig.columns:

    print("{}:\ntrain dataset:{}\ntest dataset:{}".format(col,train_orig[col].nunique(),test_orig[col].nunique()))

    print("=======================================")
test_orig.isna().sum()
train_orig.isna().sum()
train_orig['Admission_Deposit'].describe()
test_orig['Admission_Deposit'].describe()
train_orig.head()
test_orig.head()
train_orig.info()
print(train_orig['Stay'].unique())

print(f"\nTotal number of target values:{train_orig['Stay'].nunique()}")
data= pd.concat([train_orig,test_orig],sort=False)
data.isna().sum()
data.info()
for col in data.columns:

    print("{}:{}".format(col,data[col].nunique()))

    print("=======================================")

    

# Hence case_id is unique for every row
categorical_col=[]

for col in data.columns:

    if data[col].dtype== object and data[col].nunique()<=50:

        categorical_col.append(col)

print(categorical_col)
for col in categorical_col:

    print(f"{col}:\n{data[col].value_counts()}")

    print("=======================================")
data.groupby(['Hospital_region_code','Ward_Facility_Code']).size()
data['Hospital_region_code_FEAT_Ward_Facility_Code']= data['Hospital_region_code']+'_'+data['Ward_Facility_Code']
data.groupby(['Hospital_type_code','Hospital_code']).size()
data['Hospital_code']= data['Hospital_code'].apply(lambda x: str(x))

data['Hospital_type_code_FEAT_Hospital_code']= data['Hospital_type_code']+'_'+data['Hospital_code']

data['Hospital_code']= data['Hospital_code'].apply(lambda x: int(x))
data.groupby(['Hospital_type_code','Hospital_region_code']).size()
data['Hospital_type_code_FEAT_Hospital_region_code']= data['Hospital_type_code']+'_'+data['Hospital_region_code']
data.groupby(['Hospital_region_code','City_Code_Hospital']).size()
data['City_Code_Hospital']= data['City_Code_Hospital'].apply(lambda x: str(x))

data['Hospital_region_code_FEAT_City_Code_Hospital']= data['Hospital_region_code']+'_'+data['City_Code_Hospital']

data['City_Code_Hospital']= data['City_Code_Hospital'].apply(lambda x: int(x))
data.groupby(['Bed Grade','Ward_Facility_Code']).size()
data['Visitors with Patient'].unique()
data['City_Code_Patient'].unique()
data['Stay'].value_counts()
data['prev_hosp_code']= data['Hospital_code'].shift(1,axis=0)

data['prev_patientid']= data['patientid'].shift(1,axis=0)

data['prev_hosp_code'].fillna(0,inplace=True)

data['prev_patientid'].fillna(31397,inplace=True)
def fxy(prev_hosp_code,hosp_code,prev_patientid,patientid):

    if ((prev_patientid-patientid==0)&(prev_hosp_code-hosp_code==0))==True:

        return 1

    else:

        return 0

data['patient_visiting_consecutive']= data.apply(lambda x: fxy(x['prev_hosp_code'],x['Hospital_code'],

                                                               x['prev_patientid'],x['patientid']),axis=1)
data['patient_visiting_consecutive'].value_counts()
data.head().T
data.drop(['case_id','patientid','Stay','prev_hosp_code','prev_patientid'],axis=1,inplace=True)
categorical_col=[]

for col in data.columns:

    if data[col].dtype== object and data[col].nunique()<=50:

        categorical_col.append(col)

print(categorical_col)
le= LabelEncoder()
for col in categorical_col:

    data[col]= le.fit_transform(data[col])
#Filling null values

data['City_Code_Patient'].fillna(data['City_Code_Patient'].median(),inplace=True)

data['Bed Grade'].fillna(-1,inplace=True)
train_new= data[:len(train_orig)]

test_new= data[len(train_orig):]
y_le= LabelEncoder()



y= y_le.fit_transform(train_orig['Stay'])
check=pd.concat([train_new,pd.DataFrame(data=y,columns=['Stay'])],axis=1)
check.corr()['Stay'].sort_values()
y_le.classes_


X_train, X_test, y_train, y_test = train_test_split(train_new, y, test_size=0.2, random_state=101)
model = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.6, gamma=0.1, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.0300000012, max_delta_step=0, max_depth=8,

              min_child_weight=3, monotone_constraints=None,

              n_estimators=500, n_jobs=0, num_class=11, num_parallel_tree=1,

              objective='multi:softprob', random_state=0, reg_alpha=0.1,

              reg_lambda=1, scale_pos_weight=None, subsample=0.6,

              tree_method=None, validate_parameters=False, verbosity=None)
model.fit(X_train,y_train)
pred= model.predict(X_test)
print(classification_report(pred,y_test))
from xgboost import plot_importance



plot_importance(model);
testset_pred= model.predict(test_new)
testset_pred= list(y_le.inverse_transform(testset_pred))
subm.head()
final_subm= pd.DataFrame(data= testset_pred,index=subm['case_id'],columns=['Stay'])
final_subm.to_csv('final_subm_new.csv')
df= pd.read_csv('final_subm_new.csv')

df.head()
# import Libraries

import pandas as pd

import numpy as np

import matplotlib as pyplot

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# to see all the comands result in a single kernal 

%load_ext autoreload

%autoreload 2

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# to increase no. of rows and column visibility in outputs

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
#Upload data

train= pd.read_csv(r'../input/janatahack-healthcare-analytics-ii/Train/train.csv')

test= pd.read_csv(r'../input/janatahack-healthcare-analytics-ii/test.csv')

sample= pd.read_csv(r'../input/janatahack-healthcare-analytics-ii/sample_submission.csv')

train.shape

test.shape

sample.shape
#to find no of common and distinct values in test and train 

#findout no common Paient ID in test and train set

print('Checking Data distribution for Train! \n')

for col in test.columns:

     print(f'Distinct entries in {col}: {train[col].nunique()}')

     print(f'Distinct entries in {col}: {test[col].nunique()}')    

     print(f'Common # of {col} entries in test and train: {len(np.intersect1d(train[col].unique(), test[col].unique()))}')
sample.head()

sample.info()
train.head()
#to find out mo. of nan values

train.isna().sum()

test.isna().sum()
# for analysis add a new variable 

train['tar']=train['Stay'].replace({'0-10':5,'11-20':15,'21-30':25,'31-40':35,'41-50':45,'51-60':55,'61-70':65,'71-80':75,'81-90':85,'91-100':95,'More than 100 Days':110})
train['Admision_type_severity']=train['Type of Admission']+train['Severity of Illness']

test['Admision_type_severity']=test['Type of Admission']+test['Severity of Illness']
train.groupby(['Hospital_region_code','City_Code_Hospital','Hospital_type_code','Ward_Type'])['tar'].count()
# gives analysis that there are 3 region which have 11 cities, which have 32 hospitals 
train.groupby('Ward_Type')['tar'].mean().plot()
train.groupby(['Hospital_code'])['tar'].mean().plot()
train['Stay'].value_counts()/len(train)
train.corr()
# sns.pairplot(train)

train['is_train']=1

test['is_train']=0
df=pd.concat([train,test],axis=0)
df.shape
#total visits of a patient to hospital, total no. of visitors to patient

Encoding = df.groupby('patientid')['case_id'].count()

df['total_visits']= df['patientid'].map(Encoding)

Encoding = df.groupby('patientid')['Visitors with Patient'].sum()

df['total_visitors']= df['patientid'].map(Encoding)

Encoding = df.groupby('patientid')['Visitors with Patient'].mean()

df['avg_visitors']= df['patientid'].map(Encoding)
Encoding = df.groupby('patientid')['Admission_Deposit'].sum()

df['sum_ad']= df['patientid'].map(Encoding)

Encoding = df.groupby('patientid')['Admission_Deposit'].mean()

df['mean_ad']= df['patientid'].map(Encoding)

Encoding = df.groupby('patientid')['Admission_Deposit'].max()

df['max_ad']= df['patientid'].map(Encoding)

Encoding = df.groupby('patientid')['Admission_Deposit'].min()

df['min_ad']= df['patientid'].map(Encoding)
Encoding = df.groupby('Available Extra Rooms in Hospital')['Admission_Deposit'].mean()

df['aer_ad']= df['Available Extra Rooms in Hospital'].map(Encoding)

Encoding = df.groupby('Department')['Admission_Deposit'].mean()

df['dept_ad']= df['Department'].map(Encoding)

Encoding = df.groupby('Ward_Type')['Admission_Deposit'].mean()

df['wt_ad']= df['Ward_Type'].map(Encoding)

Encoding = df.groupby('Admision_type_severity')['Admission_Deposit'].mean()

df['ads_ad']= df['Admision_type_severity'].map(Encoding)
df['mean_Admission_Deposit_per_patient_hosp']=df.groupby(['patientid','Hospital_code'])['Admission_Deposit'].transform('mean')

df['sum_Admission_Deposit_per_patient_hosp']=df.groupby(['patientid','Hospital_code'])['Admission_Deposit'].transform('sum')

df['max_Admission_Deposit_per_patient_hosp']=df.groupby(['patientid','Hospital_code'])['Admission_Deposit'].transform('max')

df['min_Admission_Deposit_per_patient_hosp']=df.groupby(['patientid','Hospital_code'])['Admission_Deposit'].transform('min')
df=df.fillna(0)

df['City_Code_Patient']=df['City_Code_Patient'].astype(int)

df['Bed Grade']=df['Bed Grade'].astype(int)
train1=df[df['is_train']==1]

test1=df[df['is_train']==0]
col_1=['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Available Extra Rooms in Hospital', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness', 'Visitors with Patient', 'Age', 'Admission_Deposit', 'total_visits','Admision_type_severity', 'total_visitors', 'avg_visitors','sum_ad', 'mean_ad', 'max_ad', 'min_ad', 'aer_ad', 'dept_ad', 'wt_ad', 'ads_ad', 'mean_Admission_Deposit_per_patient_hosp', 'sum_Admission_Deposit_per_patient_hosp', 'max_Admission_Deposit_per_patient_hosp', 'min_Admission_Deposit_per_patient_hosp']
from sklearn.model_selection import train_test_split

X_t, X_tt, y_t, y_tt = train_test_split(train1[col_1], train1['Stay'], test_size=.3, random_state=2,shuffle=True,stratify= train1['Stay'])
train1.head()
train1.columns
cat_col=['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Available Extra Rooms in Hospital', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient', 'Type of Admission','Admision_type_severity', 'Severity of Illness','Age']
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

catb = CatBoostClassifier(iterations=5000,eval_metric='Accuracy',depth=7,l2_leaf_reg=2 ,task_type = 'GPU')

catb= catb.fit(X_t , y_t, cat_features=cat_col,eval_set=(X_tt, y_tt),plot=True,verbose=500)

y_cat = catb.predict(X_tt)

print(catb.score(X_t , y_t))

print(catb.score(X_tt , y_tt))
print(catb.score(X_t , y_t))

print(catb.score(X_tt , y_tt))
# 43.19 by adding pateint hospital admission fees sum mean max min

# 43.23 by changind nestimator from 3000 to 5000 # 43.00 in public leaderboard and 42.91 on private leader board 
feat_importances = pd.Series(catb.feature_importances_, index=X_t.columns)

#feat_importances.nlargest(30).plot(kind='barh')

feat_importances.nsmallest(20).plot(kind='barh')

plt.show()
catb= catb.fit(train1[col_1],train['Stay'],cat_features=cat_col,verbose=1000)

y_cat = catb.predict(test1[col_1])

sample['Stay']=y_cat

sample.to_csv('cat.csv',index=False)
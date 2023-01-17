import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
pd.set_option('display.max_row',10000)
pd.set_option('display.max_columns',150)
app_train = pd.read_csv('../input/application_train.csv')
app_test=pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_bal = pd.read_csv('../input/bureau_balance.csv')
pos_cash = pd.read_csv('../input/POS_CASH_balance.csv')
cc_bal = pd.read_csv('../input/credit_card_balance.csv')
app_prev = pd.read_csv('../input/previous_application.csv')
instal_pay = pd.read_csv('../input/installments_payments.csv')
app_train.shape
app_train.head()
len(app_train['SK_ID_CURR'].unique().tolist())
app_test.shape
len(app_test['SK_ID_CURR'].unique().tolist())
bureau.shape
bureau.head()
len(bureau['SK_ID_CURR'].unique().tolist())
len(bureau['SK_ID_BUREAU'].unique().tolist())
pos_cash.shape
len(pos_cash['SK_ID_CURR'].unique().tolist())
cc_bal.shape
len(cc_bal['SK_ID_CURR'].unique().tolist())
bad_dist = app_train['TARGET'].value_counts().reset_index()
bad_dist.columns = ['val','count']
bad_dist['percent'] = (bad_dist['count']/app_train.shape[0])*100
bad_dist
import matplotlib.pyplot as plt
plt.figure()
plt.bar(app_train['TARGET'].value_counts().index,app_train['TARGET'].value_counts());
train_id = app_train['SK_ID_CURR']
test_id = app_test['SK_ID_CURR']
train_y = app_train['TARGET']
app_train['key'] = 'train'
app_train.drop(['SK_ID_CURR','TARGET'],axis=1,inplace=True)
app_test.drop(['SK_ID_CURR'],axis=1,inplace=True)
app_test['key']='test'
data=pd.concat((app_train,app_test)).reset_index(drop=True)

data.shape
bureau.info()
#Randomly looking at the distribution of the variables that i'm curious about
bureau.groupby(['CREDIT_CURRENCY'])['CREDIT_CURRENCY'].count()
bureau_bal.info()
bureau_bal.groupby(['STATUS'])['STATUS'].count()
instal_pay.info()
app_prev.info()
cc_bal.info()
pos_cash.info()
cat_data=data.select_dtypes(include=["O"])
num_data = data.select_dtypes(include=["number"])
num_data.shape
cat_data.shape
cat_data.describe()
pd.set_option('display.float_format',lambda x: '%.1f' % x)
num_data.describe()
missing = num_data.isnull().sum(axis=0).reset_index()
missing.columns=['column_name','missing_count']
percent = ((num_data.isnull().sum(axis=0)/num_data.shape[0])*100).reset_index()
percent.columns=['variable','percent']
missing_data = pd.concat([missing,percent['percent']],axis=1)
missing_data=missing_data.ix[missing_data['missing_count']>0].sort_values(by='missing_count')
missing_data
missing_col=missing_data['column_name'][missing_data['percent']> 45].tolist()
for i in missing_col:
    num_data.drop([i],axis=1,inplace=True)
num_data.shape
num_data['DAYS_LAST_PHONE_CHANGE']= num_data['DAYS_LAST_PHONE_CHANGE'].fillna(num_data['DAYS_LAST_PHONE_CHANGE'].mode()[0])
num_data['CNT_FAM_MEMBERS']= num_data['CNT_FAM_MEMBERS'].fillna(num_data['CNT_FAM_MEMBERS'].mode()[0])
num_data['AMT_ANNUITY']= num_data['AMT_ANNUITY'].fillna(num_data['AMT_ANNUITY'].mean())
num_data['AMT_GOODS_PRICE']= num_data['AMT_GOODS_PRICE'].fillna(0)
num_data['EXT_SOURCE_2']= num_data['EXT_SOURCE_2'].fillna(num_data['EXT_SOURCE_2'].mean())
num_data['DEF_60_CNT_SOCIAL_CIRCLE']= num_data['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(num_data['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0])
num_data['OBS_60_CNT_SOCIAL_CIRCLE']= num_data['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(num_data['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0])
num_data['DEF_30_CNT_SOCIAL_CIRCLE']= num_data['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(num_data['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0])
num_data['OBS_30_CNT_SOCIAL_CIRCLE']= num_data['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(num_data['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0])
num_data['AMT_REQ_CREDIT_BUREAU_MON']= num_data['AMT_REQ_CREDIT_BUREAU_MON'].fillna(num_data['AMT_REQ_CREDIT_BUREAU_MON'].mode()[0])
num_data['AMT_REQ_CREDIT_BUREAU_WEEK']= num_data['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(num_data['AMT_REQ_CREDIT_BUREAU_WEEK'].mode()[0])
num_data['AMT_REQ_CREDIT_BUREAU_DAY']= num_data['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(num_data['AMT_REQ_CREDIT_BUREAU_DAY'].mode()[0])
num_data['AMT_REQ_CREDIT_BUREAU_HOUR']= num_data['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(num_data['AMT_REQ_CREDIT_BUREAU_HOUR'].mode()[0])
num_data['AMT_REQ_CREDIT_BUREAU_QRT']= num_data['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(num_data['AMT_REQ_CREDIT_BUREAU_QRT'].mode()[0])
num_data['AMT_REQ_CREDIT_BUREAU_YEAR']= num_data['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(num_data['AMT_REQ_CREDIT_BUREAU_YEAR'].mode()[0])
num_data['EXT_SOURCE_3']= num_data['EXT_SOURCE_3'].fillna(num_data['EXT_SOURCE_3'].mean())
missing_cat = cat_data.isnull().sum(axis=0).reset_index()
missing_cat.columns=['column_name','missing_count']
percent_cat = ((cat_data.isnull().sum(axis=0)/cat_data.shape[0])*100).reset_index()
percent_cat.columns=['variable','percent']
missing_data_cat = pd.concat([missing_cat,percent_cat['percent']],axis=1)
missing_data_cat=missing_data_cat.ix[missing_data_cat['missing_count']>0].sort_values(by='missing_count')
missing_data_cat
missing_col_cat=missing_data_cat['column_name'][missing_data_cat['percent']> 45].tolist()
for i in missing_col_cat:
    cat_data.drop([i],axis=1,inplace=True)
cat_data.shape
cat_data['NAME_TYPE_SUITE']= cat_data['NAME_TYPE_SUITE'].fillna(cat_data['NAME_TYPE_SUITE'].mode()[0])
cat_data['NAME_EDUCATION_TYPE'][cat_data.OCCUPATION_TYPE.isnull()].value_counts()
cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Secondary / secondary special'].value_counts()
cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Secondary / secondary special'] =  cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Secondary / secondary special'].fillna(cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Secondary / secondary special'].mode()[0])
cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Higher education'] =  cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Higher education'].fillna(cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Higher education'].mode()[0])
cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Incomplete higher'] = cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Incomplete higher'].fillna(cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Incomplete higher'].mode()[0])
cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Lower secondary'] = cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Lower secondary'].fillna(cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Lower secondary'].mode()[0])
cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Academic degree'] = cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Academic degree'].fillna(cat_data['OCCUPATION_TYPE'][cat_data['NAME_EDUCATION_TYPE']=='Academic degree'].mode()[0])
for i in num_data.columns:
    ll,ul = np.percentile(num_data[i],[1,99])
    num_data[i].ix[num_data[i]>ul] = ul
    num_data[i].ix[num_data[i] < ll]=ll
    
num_data.describe()

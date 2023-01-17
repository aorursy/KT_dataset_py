import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc



from sklearn.preprocessing import LabelBinarizer,PolynomialFeatures

from sklearn.metrics import roc_auc_score, roc_curve,auc

from sklearn.model_selection import KFold

import lightgbm as lgb

%matplotlib inline

plt.style.use('ggplot')

seed =420

pd.options.display.max_rows = 100
nrows = 100000

#nrows =None

path = '../input/'

#path = 'dataset/'

app_train = pd.read_csv(path+'application_train.csv', nrows= nrows)

app_test = pd.read_csv(path+'application_test.csv', nrows= None)

bureau_balance = pd.read_csv(path+'bureau_balance.csv', nrows=nrows)

bureau = pd.read_csv(path+'bureau.csv', nrows=nrows)

installments_payments = pd.read_csv(path+ 'installments_payments.csv', nrows= nrows)

POS_CASH_balance = pd.read_csv(path+ 'POS_CASH_balance.csv', nrows= nrows)

credit_card_balance = pd.read_csv(path+'credit_card_balance.csv',nrows=nrows)

previous_application = pd.read_csv(path+ 'previous_application.csv', nrows= nrows)

gc.collect()



test_index = app_test['SK_ID_CURR'] # Store test index
# Reduce memory of dataset

def reduce_memory_usage(df):

    """ The function will reduce memory of dataframe """

    intial_memory = df.memory_usage().sum()/1024**2

    print('Intial memory usage:',intial_memory,'MB')

    for col in df.columns:

        mn = df[col].min()

        mx = df[col].max()

        if df[col].dtype != object:            

            if df[col].dtype == int:

                if mn >=0:

                    if mx < np.iinfo(np.uint8).max:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < np.iinfo(np.uint16).max:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < np.iinfo(np.uint32).max:

                        df[col] = df[col].astype(np.uint32)

                    elif mx < np.iinfo(np.uint64).max:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)

            if df[col].dtype == float:

                df[col] =df[col].astype(np.float32)

    

    red_memory = df.memory_usage().sum()/1024**2

    print('Memory usage after complition: ',red_memory,'MB')
def basic_details(df):

    """ Returns basic detials of features in dataset"""

    k = pd.DataFrame()

    k['missing_value'] = df.isnull().sum()

    k['%missing_value'] = round(df.isnull().sum()*100/df.shape[0],2)

    k['dtypes'] = df.dtypes

    k['N unique'] = df.nunique()

    #k['mean'] = df.mean()

    return k
# Fill missing value with mode

def missing_value_fill(df ,columns, mean_or_mode='mode'):

    """ Fill missing value with

        mode: for categorical variable

        mean: for numerical variable"""

    for i in columns:

        if (df[i].isnull().sum()>0) and (mean_or_mode =='mode'):

            df[i].fillna(df[i].mode()[0], inplace=True)

        elif (df[i].isnull().sum()>0) and (mean_or_mode =='mean'):

            df[i].fillna(df[i].mean(), inplace=True)
# Replace missing value np.nan

def replace_XNA_XAP(df):

    "Replace XNA,XAP"

    df.replace(to_replace = {'XNA':np.nan,'XAP':np.nan},inplace=True,value= None)

    return df
def one_hot_encoding(df,columns,nan_as_category = True):

    """ One hot encoding of categorical variable"""

    print('*'*5,'One hot encoding of categorical variable','*'*5)

    print('Original shape',df.shape)

    original_columns = df.columns

    # One hot encoding using get dummies function

    df = pd.get_dummies(df, columns= columns,drop_first=True,dummy_na=nan_as_category)

    new_columns = [i for i in df.columns if i not in original_columns]

    print('After OHE',df.shape)

    return df,new_columns
def descriptive_stat_feat(df,columns):

    """ Descriptive statistics feature

    genarating function: Mean,Median,Q1,Q3"""

    print('*'*5,'Descriptive statistics feature','*'*5)

    print('Before',df.shape)

    mean = df[columns].mean()

    median = df[columns].median()

    Q1 = np.percentile(df[columns], 25, axis=0)

    Q3 = np.percentile(df[columns], 75, axis=0)

    for i,j in enumerate(columns):

        df['mean_'+j] = (df[j] < mean[i]).astype('int8')

        df['median_'+j] = (df[j] > median[i]).astype('int8')

        df['Q1'+j] = (df[j] < Q1[i]).astype('int8')

        df['Q3'+j] = (df[j] > Q3[i]).astype('int8')

    print('After ',df.shape)
def binary_encoding(df,columns):

    """Binary encoding"""

    print('*'*5,'Binary encoding','*'*5)

    lb = LabelBinarizer()

    print('Original shape:',df.shape)

    original_col = df.columns

    #columns = [i for i in columns if df[columns].nunique()>2]

    for i in columns:

        if df[i].nunique() >2:

            result = lb.fit_transform(df[i].fillna(df[i].mode()[0],axis=0))

            col = ['BIN_'+ str(i)+'_'+str(c) for c in lb.classes_]

            result1 = pd.DataFrame(result, columns=col)

            df = df.join(result1)

    print('After:',df.shape)

    new_col = [c for c in df.columns if c not in original_col]

    return df, new_col
def dist_box__plot_with_log(df,column,ncols=2,Trans_func= None):

    """Plot distribution plot with log on diffirent target value

    Can be used for train/ test dataset

    Trans_fucn: log, log1p, exp, sqrt, expm1...

    """

    # Apply natural log on total income

    fig,a = plt.subplots(nrows=1,ncols=ncols,figsize=(14,4))

    # Box plot

    ax = plt.subplot(1,ncols,1)    

    sns.boxplot(x ='TARGET', y=column, data=df,ax=ax,palette='RdYlGn_r')

    plt.title('Boxplot')

    

    tmp_0 = df[df['TARGET']==1][column].dropna()

    tmp_1 = df[df['TARGET']==0][column].dropna()

    # Distribution plot    

    ax = plt.subplot(1,ncols,2)    

    sns.distplot(tmp_0,ax=ax,color='orange',label='Target=1',hist=False)

    sns.distplot(tmp_1,ax=ax,color='green',label='Target=0',hist=False)

    ax.set_title('Distribution plot')

    

    # Distribution plot with log(x+1) on column

    if Trans_func !=None:

        ax = plt.subplot(1,ncols,3)

        tmp_0 = df[df['TARGET']==1][column].dropna().apply(Trans_func)

        tmp_1 = df[df['TARGET']==0][column].dropna().apply(Trans_func)

        sns.distplot(np.log1p(tmp_0), ax=ax,color='orange',label='Target=1',hist=False)

        sns.distplot(np.log1p(tmp_1), ax=ax,color='green',label='Target=0',hist=False)

        ax.set_title('${}$'.format(Trans_func))

    

    plt.tight_layout()

    plt.legend()
def box_dist_plot_with_trans(df,column,nrows=1,ncols=2,Trans_func=None):

    """Dirtibution plot and Box plot 

    log,log1p,exp,sqrt,expm1.. numpy function

    """

    fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(14,4*nrows))

    

    # Box plot

    ax = plt.subplot(nrows,ncols,1)

    sns.boxplot(x =column, data=df,ax=ax)

    ax.set_title('Box plot')

    

    # Distribution plot

    ax = plt.subplot(nrows,ncols,2)

    sns.distplot(df[column].dropna(),ax=ax,color='blue',bins=30)

    ax.set_title('Distribution plot')

    

    # Transformation plot

    if Trans_func !=None:

        tmp = df[column].dropna().apply(Trans_func)

        ax = plt.subplot(nrows,ncols,3)

        sns.distplot(tmp,ax=ax,color='red',bins=30)

        ax.set_title('${}$'.format(Trans_func))

      

    plt.tight_layout()
print('Number of rows and columns in train dataset: ',app_train.shape)

app_train.head()
print('Number of rows and columns in test dataset: ',app_test.shape)

app_test.head()
pd.read_csv(path+'sample_submission.csv').head()
app_train.columns.values
#sns.scatterplot(range(app_train.shape[0]),app_train['SK_ID_CURR'].sort_values())

#sns.scatterplot(range(app_test.shape[0]),app_test['SK_ID_CURR'].sort_values())
test_index = app_test['SK_ID_CURR']

app_train_col_drop = [] # Drop the columns, which have least importance
print('Count\n',app_train['TARGET'].value_counts())

print('%\n',app_train['TARGET'].value_counts()*100/app_train.shape[0])
f =plt.figure(figsize=(14,6))

ax= f.add_subplot(221)

sns.countplot(app_train['NAME_CONTRACT_TYPE'])

ax=f.add_subplot(222)

sns.countplot(app_train['CODE_GENDER'])

ax=f.add_subplot(223)

sns.countplot(app_train['FLAG_OWN_CAR'])

ax=f.add_subplot(224)

sns.countplot(app_train['FLAG_OWN_REALTY'])

plt.tight_layout()
# convert to categorical type

app_train[['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']] =   app_train[

    ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']].astype('object')

app_test[['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']] =   app_test[

    ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']].astype('object')

#plt.figure(figsize=(14,4))

#sns.countplot(app_train['CNT_CHILDREN'])

tmp = app_train['CNT_CHILDREN'].value_counts().to_frame()

tmp['%'] = (app_train['CNT_CHILDREN'].value_counts() *100 / app_train.shape[0])

tmp
# convert to categorical type

app_train['CNT_CHILDREN'] = app_train['CNT_CHILDREN'].astype('object')

app_test['CNT_CHILDREN'] = app_test['CNT_CHILDREN'].astype('object')
app_train[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',]].describe()
dist_box__plot_with_log(app_train,column='AMT_INCOME_TOTAL',ncols=3,Trans_func='log')
print('Default',app_train[app_train['AMT_INCOME_TOTAL'] >0.2e8]['AMT_INCOME_TOTAL'])



# apply log on feature 

app_train['AMT_INCOME_TOTAL'] = np.log(app_train['AMT_INCOME_TOTAL'])

app_test['AMT_INCOME_TOTAL'] = np.log(app_test['AMT_INCOME_TOTAL'])
dist_box__plot_with_log(app_train,column='AMT_CREDIT',ncols=3,Trans_func='sqrt')
# apply square root on feature

app_train['AMT_CREDIT'] = np.sqrt(app_train['AMT_CREDIT'])

app_test['AMT_CREDIT'] = np.sqrt(app_test['AMT_CREDIT'])
dist_box__plot_with_log(app_train,column='AMT_ANNUITY',ncols=3,Trans_func='log')
# apply log on feature

app_train['AMT_ANNUITY'] = np.log(app_train['AMT_ANNUITY'])

app_test['AMT_ANNUITY'] = np.log(app_test['AMT_ANNUITY'])
dist_box__plot_with_log(app_train,column='AMT_GOODS_PRICE',ncols=3,Trans_func='log1p')
f = plt.figure(figsize= (14,12))

#plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))



ax= f.add_subplot(321)

tmp = app_train['NAME_TYPE_SUITE'].value_counts()

sns.barplot(tmp.values,tmp.index,palette='cool')

for i, v in enumerate(tmp.values):

    ax.text(0.8,i,v,color='k',fontsize=10)

ax.set_title('Relationship type')



ax = f.add_subplot(322)

tmp = app_train['NAME_INCOME_TYPE'].value_counts()

sns.barplot(tmp.values,tmp.index,palette='Wistia')

for i, v in enumerate(tmp.values):

    ax.text(0.8,i,v,color='k',fontsize=10)

ax.set_title('Employment type')



ax = f.add_subplot(323)

tmp = app_train['NAME_EDUCATION_TYPE'].value_counts()

sns.barplot(tmp.values,tmp.index,palette='Wistia')

for i, v in enumerate(tmp.values):

    ax.text(0.8,i,v,color='k',fontsize=10)

ax.set_title('Education type')



ax = f.add_subplot(324)

tmp = app_train['NAME_FAMILY_STATUS'].value_counts()

sns.barplot(tmp.values,tmp.index,palette='cool')

for i, v in enumerate(tmp.values):

    ax.text(0.8,i,v,color='k',fontsize=10)

ax.set_title('Family status')



ax = f.add_subplot(325)

tmp = app_train['NAME_HOUSING_TYPE'].value_counts()

sns.barplot(tmp.values,tmp.index,palette='cool')

for i, v in enumerate(tmp.values):

    ax.text(0.8,i,v,color='k',fontsize=10)

ax.set_title('House type')



plt.subplots_adjust(wspace=0.4)
# convert to categorical type

app_train[['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 

           'NAME_HOUSING_TYPE']] = app_train[['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',

       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']].astype('object')



app_test[['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 

           'NAME_HOUSING_TYPE']] = app_test[['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',

       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']].astype('object')
dist_box__plot_with_log(app_train,column='REGION_POPULATION_RELATIVE', ncols=2)
(app_train['DAYS_BIRTH']/-365).describe()
tmp = app_train[['TARGET','DAYS_BIRTH']]

tmp['DAYS_BIRTH'] = tmp['DAYS_BIRTH']/-365

dist_box__plot_with_log(tmp,'DAYS_BIRTH')
print('missing value:',app_train['DAYS_EMPLOYED'].isnull().sum())

(app_train['DAYS_EMPLOYED']/-365).describe()
((app_train['DAYS_EMPLOYED']/-365)[(app_train['DAYS_EMPLOYED']/-365)<0][:5],

app_train['DAYS_EMPLOYED'][app_train['DAYS_EMPLOYED']>0][:5],

app_test['DAYS_EMPLOYED'][app_test['DAYS_EMPLOYED']>0][:5])
# fill missing value

app_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)

app_test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
tmp = app_train[['TARGET','DAYS_EMPLOYED']]

tmp['DAYS_EMPLOYED'] = tmp['DAYS_EMPLOYED']/-365

dist_box__plot_with_log(tmp,'DAYS_EMPLOYED')
(app_train['DAYS_REGISTRATION']/-365).describe()
tmp = app_train[['TARGET','DAYS_REGISTRATION']]

tmp['DAYS_REGISTRATION'] = tmp['DAYS_REGISTRATION']/-365

dist_box__plot_with_log(tmp,'DAYS_REGISTRATION')
(app_train['DAYS_ID_PUBLISH']/-365).describe()
tmp = app_train[['TARGET','DAYS_ID_PUBLISH']]

tmp['DAYS_ID_PUBLISH'] = tmp['DAYS_ID_PUBLISH']/-365

dist_box__plot_with_log(tmp,'DAYS_ID_PUBLISH')
plt.figure(figsize= (14,3))

sns.countplot(app_train['OWN_CAR_AGE'])

#plt.title('Count plot of Own car age')

plt.xticks(rotation=90);
app_train['OWN_CAR_AGE'] = app_train['OWN_CAR_AGE'].astype('object')

app_test['OWN_CAR_AGE'] = app_test['OWN_CAR_AGE'].astype('object')
f = plt.figure(figsize= (14,6))



ax= f.add_subplot(231)

tmp = app_train['FLAG_MOBIL'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))

ax.set_title('Mobile available')



ax = f.add_subplot(232)

tmp = app_train['FLAG_EMP_PHONE'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('Wistia'))

ax.set_title('Mobile emp availablity')



ax = f.add_subplot(233)

tmp = app_train['FLAG_WORK_PHONE'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))

ax.set_title('Work Phone availablity')



ax = f.add_subplot(234)

tmp = app_train['FLAG_CONT_MOBILE'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('Wistia'))

ax.set_title('Mobile count')



ax = f.add_subplot(235)

tmp = app_train['FLAG_PHONE'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))

ax.set_title('Phone availablity')



ax = f.add_subplot(236)

tmp = app_train['FLAG_EMAIL'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('coolwarm'))

ax.set_title('Email availablity')



plt.subplots_adjust(wspace=0.4)
# drop few columns

app_train_col_drop.append('FLAG_MOBIL')

app_train_col_drop.append('FLAG_CONT_MOBILE')

app_train_col_drop.append('FLAG_EMAIL')



# convert to category

app_train[['FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_PHONE']] = app_train[[

    'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_PHONE']].astype('object')

app_test[['FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_PHONE']] = app_test[[

    'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_PHONE']].astype('object')
plt.figure(figsize= (14,3))

sns.countplot(app_train['CNT_FAM_MEMBERS'])

plt.title('Count plot of Family members')

plt.xticks(rotation=90);
# convert to category

app_train['CNT_FAM_MEMBERS'] = app_train['CNT_FAM_MEMBERS'].astype('object')

app_test['CNT_FAM_MEMBERS'] = app_test['CNT_FAM_MEMBERS'].astype('object')
f = plt.figure(figsize= (14,3))



ax = f.add_subplot(121)

tmp = app_train['REGION_RATING_CLIENT'].value_counts()

ax.pie(tmp.values, labels= tmp.index, autopct= '%1.2f%%',colors=sns.color_palette('coolwarm'))

ax.set_title('Region rating client')



ax = f.add_subplot(122)

tmp = app_train['REGION_RATING_CLIENT_W_CITY'].value_counts()

ax.pie(tmp.values, labels= tmp.index, autopct= '%1.2f%%',colors=sns.color_palette('Spectral'))

ax.set_title('Region rating client with city');
# convert to category

app_train[['REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']] = app_train[[

    'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']].astype('object')



app_test[['REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']] = app_test[[

    'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']].astype('object')
plt.figure(figsize= (14,3))

sns.countplot(app_train['WEEKDAY_APPR_PROCESS_START'],

              order=['SUNDAY','MONDAY','TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY','SATURDAY', ])

#plt.title('Count plot of Own car age')

plt.xticks(rotation=90);
# convert to category

app_train['WEEKDAY_APPR_PROCESS_START'] = app_train['WEEKDAY_APPR_PROCESS_START'].astype('object')

app_test['WEEKDAY_APPR_PROCESS_START'] = app_test['WEEKDAY_APPR_PROCESS_START'].astype('object')
plt.figure(figsize= (14,3))

sns.countplot(app_train['HOUR_APPR_PROCESS_START'])

#plt.title('Count plot of Own car age')

plt.xticks(rotation=90);
# convert to category

app_train['HOUR_APPR_PROCESS_START'] = app_train['HOUR_APPR_PROCESS_START'].astype('object')

app_test['HOUR_APPR_PROCESS_START'] = app_test['HOUR_APPR_PROCESS_START'].astype('object')
f = plt.figure(figsize= (14,6))



ax= f.add_subplot(231)

tmp = app_train['REG_REGION_NOT_LIVE_REGION'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))

ax.set_title('REG_REGION_NOT_LIVE_REGION')



ax = f.add_subplot(232)

tmp = app_train['REG_REGION_NOT_WORK_REGION'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('Wistia'))

ax.set_title('REG_REGION_NOT_WORK_REGION')



ax = f.add_subplot(233)

tmp = app_train['LIVE_REGION_NOT_WORK_REGION'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))

ax.set_title('LIVE_REGION_NOT_WORK_REGION')



ax = f.add_subplot(234)

tmp = app_train['REG_CITY_NOT_LIVE_CITY'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('Wistia'))

ax.set_title('REG_CITY_NOT_LIVE_CITY')



ax = f.add_subplot(235)

tmp = app_train['REG_CITY_NOT_WORK_CITY'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))

ax.set_title('REG_CITY_NOT_WORK_CITY')



ax = f.add_subplot(236)

tmp = app_train['LIVE_CITY_NOT_WORK_CITY'].value_counts()

plt.pie(tmp.values,labels=tmp.index,autopct='%1.1f%%',colors=sns.color_palette('coolwarm'))

ax.set_title('LIVE_CITY_NOT_WORK_CITY')



plt.subplots_adjust(wspace=0.4)
# drop a feature

app_train_col_drop.append('REG_REGION_NOT_LIVE_REGION')



# convert to category

app_train[['REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',

       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']] = app_train[['REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',

       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']].astype('object')



app_test[['REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',

       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']] = app_test[['REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',

       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']].astype('object')
plt.figure(figsize=(14,8))

tmp = app_train['ORGANIZATION_TYPE'].value_counts()[:20]

sns.barplot(tmp.values, tmp.index, palette='coolwarm')

plt.title('Organization type')

for i,v in enumerate(tmp.values):

    plt.text(1,i,v,fontsize=8,color='k')
# XNA is missing value in dataset

app_train['ORGANIZATION_TYPE'].replace('XNA',np.nan,inplace=True)

app_test['ORGANIZATION_TYPE'].replace('XNA',np.nan,inplace=True)



# convert to category

app_train['ORGANIZATION_TYPE'] = app_train['ORGANIZATION_TYPE'].astype('object')

app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].astype('object')
basic_details(app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']])
#g = sns.pairplot(app_train,vars =['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'],

#               palette='magma',hue='TARGET',kind='reg',aspect=1.5)

F = plt.figure(figsize=(14,4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].plot(kind='box',ax=ax)
basic_details(app_test[['APARTMENTS_AVG','BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG', 

           'COMMONAREA_AVG', 'ELEVATORS_AVG','ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',

       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']]

)
F = plt.figure(figsize=(14,4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['APARTMENTS_AVG','BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG']].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['APARTMENTS_AVG','BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14,5))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_AVG', 'ELEVATORS_AVG','ENTRANCES_AVG',]].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_AVG', 'ELEVATORS_AVG','ENTRANCES_AVG',]].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14,4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG']].plot(kind='kde',ax=ax)

ax = F.add_subplot(122)



plt.title('Information adout buliding where client lives')

app_train[['FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14,5))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
plt.figure(figsize= (14,8))

plt.title('Correlation matrix')

sns.heatmap(app_train[['APARTMENTS_AVG','BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG', 

           'COMMONAREA_AVG', 'ELEVATORS_AVG','ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',

       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']].corr(),

           annot=True, cmap = 'coolwarm');
(app_train[['APARTMENTS_AVG','BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG', 

           'COMMONAREA_AVG', 'ELEVATORS_AVG','ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',

       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']]

.plot(kind='box',figsize=(14,4)))

plt.xticks(rotation=90);
F = plt.figure(figsize=(14,4))

ax = F.add_subplot(121)

app_train[['APARTMENTS_MODE', 'BASEMENTAREA_MODE',

       'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE']].plot(kind='kde',ax=ax)

ax = F.add_subplot(122)

app_train[['APARTMENTS_MODE', 'BASEMENTAREA_MODE',

       'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14, 4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',]].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',]].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14, 4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',]].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',]].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14, 4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',

       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE']].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',

       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
plt.figure(figsize= (14,8))

plt.title('Correlation matrix')

sns.heatmap(app_train[['APARTMENTS_MODE', 'BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',

       'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE','FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',

       'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE']].corr(),

           annot=True, cmap = 'coolwarm');
F = plt.figure(figsize=(14, 4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['APARTMENTS_MEDI','BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',

       'YEARS_BUILD_MEDI',]].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['APARTMENTS_MEDI','BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',

       'YEARS_BUILD_MEDI']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14, 4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',]].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14, 4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['FLOORSMAX_MEDI', 'FLOORSMIN_MEDI','LANDAREA_MEDI',]].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['FLOORSMAX_MEDI', 'FLOORSMIN_MEDI','LANDAREA_MEDI',]].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
F = plt.figure(figsize=(14, 4))

ax = F.add_subplot(121)

plt.title('Information adout buliding where client lives')

app_train[['LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',

       'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI']].plot(kind='kde',ax=ax)



ax = F.add_subplot(122)

plt.title('Information adout buliding where client lives')

app_train[['LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',

       'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI']].plot(kind='box',ax=ax)

plt.xticks(rotation=90);
plt.figure(figsize= (14,8))

plt.title('Correlation matrix')

sns.heatmap(app_train[['APARTMENTS_MEDI','BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',

       'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI','ENTRANCES_MEDI', 

        'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI','LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 

        'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',]].corr(),

           annot=True, cmap = 'coolwarm');
dist_box__plot_with_log(app_train,column='TOTALAREA_MODE',ncols=2)
basic_details(app_train[['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',

       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']])
f = plt.figure(figsize= (14,8))

ax = f.add_subplot(221)

sns.countplot(app_train['FONDKAPREMONT_MODE'],ax=ax)

plt.xticks(rotation=90)

#plt.title('FONDKAPREMONT_MODE')



ax = f.add_subplot(222)

sns.countplot(app_train['HOUSETYPE_MODE'],ax=ax)

plt.xticks(rotation=90)

#plt.title('HOUSETYPE_MODE')



ax = f.add_subplot(223)

sns.countplot(app_train['WALLSMATERIAL_MODE'],ax=ax)

plt.xticks(rotation=90)

#plt.title('WALLSMATERIAL_MODE')



ax = f.add_subplot(224)

sns.countplot(app_train['EMERGENCYSTATE_MODE'],ax=ax)

plt.xticks(rotation=90)

#plt.title('EMERGENCYSTATE_MODE')



plt.tight_layout()
app_train[['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']] = app_train[['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',

       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']].astype('object')



app_test[['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']] = app_test[['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',

       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']].astype('object')
basic_details(app_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',

       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE',]])
f = plt.figure(figsize= (14,8))

ax = f.add_subplot(221)

sns.countplot(app_train['OBS_30_CNT_SOCIAL_CIRCLE'], ax=ax)



ax = f.add_subplot(222)

sns.countplot(app_train['DEF_30_CNT_SOCIAL_CIRCLE'], ax=ax)



ax = f.add_subplot(223)

sns.countplot(app_train['OBS_60_CNT_SOCIAL_CIRCLE'], ax=ax)



ax = f.add_subplot(224)

sns.countplot(app_train['DEF_60_CNT_SOCIAL_CIRCLE'], ax=ax)
# convert to category

app_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',

       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',]] = app_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',

       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',]].astype('object')



app_test[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',

       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',]] = app_test[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',

       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',]].astype('object')
dist_box__plot_with_log(app_train,column='DAYS_LAST_PHONE_CHANGE',ncols=2)
flag_col = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',

        'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 

        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',]



fig, ax = plt.subplots(4,5, figsize=(14,8),sharey=False)

axs = ax.ravel()

for i,c in enumerate(flag_col):

    sns.countplot(app_train[c],ax = axs[i],palette='cool')

    #axs[i].set_title(c)

    axs[i].set_ylabel('')

plt.tight_layout()
# we will keep FLAG_DOCUMENT_3,FLAG_DOCUMENT_6, FLAG_DOCUMENT_8

app_train[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',

        'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 

        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',]]  = app_train[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',

        'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 

        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',]].astype('object')



app_test[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',

        'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 

        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',]]  = app_test[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',

        'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 

        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',]].astype('object')





app_train_col_drop.extend(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7',  'FLAG_DOCUMENT_9',

        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 

        'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',

       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'])
basic_details(app_train[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',

       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',

       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']])
flag_col = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',

       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',

       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',]



fig, ax = plt.subplots(2,3, figsize=(14,6),sharey=False)

axs = ax.ravel()

for i,c in enumerate(flag_col):

    sns.countplot(app_train[c],ax = axs[i],palette='magma')

    #axs[i].set_title(c)

    axs[i].set_ylabel('')

plt.tight_layout()
app_train[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK', 

    'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',]] = app_train[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK', 

    'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',]].astype('object')



app_test[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',

    'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',]] = app_test[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK', 

    'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',]].astype('object')

gc.collect()
#app_train['is_train'] = 'Yes'

#app_test['is_train'] = 'No'

train_test = pd.concat([app_train,app_test],axis=0)

print('Number of rows and columns in train dataset: ',app_train.shape)

print('Number of rows and columns in test dataset: ',app_test.shape)

print('Number of rows and columns in train + test dataset: ',train_test.shape)

gc.collect()
# drop columns least significant

train_test = train_test.drop(list(set(app_train_col_drop)), axis=1)



categorical_col = train_test.select_dtypes('object').columns

numeric_col = train_test.select_dtypes('number').columns



# Int type

numeric_col = numeric_col.drop('TARGET') # Our dependent variable

gc.collect()
# Check basic details

basic_details(train_test[categorical_col])
# Check missing value

basic_details(train_test[numeric_col])
# Replace XNA, XPA as np.nan

train_test = replace_XNA_XAP(train_test)

# Replace 365243 with missing value

#train_test['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace=True)



# Fill missing value 

#missing_value_fill(train_test, categorical_col, mean_or_mode='mode')

#missing_value_fill(train_test, numeric_col, mean_or_mode='mode') 



# Binary encoding

train_test,_ = binary_encoding(train_test, categorical_col)



# One hot encoding

train_test,_ = one_hot_encoding(train_test,categorical_col,nan_as_category=True)

#train_test,cat_cols = one_hot_encoder(train_test,nan_as_category=True)



#Descriptive statistics feature

descriptive_stat_feat(train_test,numeric_col)

del app_train,app_test



# Reduce memory usage train_test dataset

reduce_memory_usage(train_test)

gc.collect()
print('Number of rows and columns in bureau dataset: ',bureau_balance.shape)

bureau_balance.head()
basic_details(bureau_balance)
box_dist_plot_with_trans(bureau_balance,column='MONTHS_BALANCE',ncols=2)
sns.countplot(bureau_balance['STATUS'])
# Replace XNA, XAP as np.nan

bureau_balance = replace_XNA_XAP(bureau_balance)



# binary encoding

bb_bin =[]

#bureau_balance,bb_bin = binary_encoding(bureau_balance,['STATUS'])



# one hot encoding

bureau_balance, bb_cat = one_hot_encoding(bureau_balance,['STATUS'],nan_as_category=False)



# aggregate

bb_aggregations = {'MONTHS_BALANCE':['min','max','mean']}



for col in bb_cat+bb_bin:

    bb_aggregations[col] =['sum','mean']



bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bb_aggregations)

bb_agg.columns = [e[0]+ "_" +e[1].upper() for e in bb_agg.columns.tolist()]

bb_agg.head(2)
print('Number of rows and columns in bureau dataset: ',bureau.shape)

bureau.head()
basic_details(bureau)
bureau_drop_col = [] # drop columns list

bureau.columns
f = plt.figure(figsize= (14,4))

ax = f.add_subplot(121)

sns.countplot(bureau['CREDIT_ACTIVE'], ax =ax)



ax = f.add_subplot(122)

sns.countplot(bureau['CREDIT_CURRENCY'], ax =ax)
bureau.columns
bureau[['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE','DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT']].describe()
box_dist_plot_with_trans(bureau,column='DAYS_CREDIT',)
box_dist_plot_with_trans(bureau,column='CREDIT_DAY_OVERDUE',Trans_func='log1p',ncols=3)
# Drop credit day overdue

#bureau_drop_col.append('CREDIT_DAY_OVERDUE')
box_dist_plot_with_trans(bureau,column='DAYS_CREDIT_ENDDATE',ncols=2)
box_dist_plot_with_trans(bureau,column='DAYS_ENDDATE_FACT',ncols=2)
box_dist_plot_with_trans(bureau,column='DAYS_CREDIT_UPDATE',ncols=2)
f = plt.figure(figsize= (14,4))

ax = f.add_subplot(121)

sns.countplot(bureau['CREDIT_TYPE'],palette='rainbow', ax = ax)

plt.xticks(rotation=90)



ax = f.add_subplot(122)

sns.countplot(bureau['CNT_CREDIT_PROLONG'])
bureau[['CREDIT_TYPE','CNT_CREDIT_PROLONG']] = bureau[['CREDIT_TYPE','CNT_CREDIT_PROLONG']].astype('object')



# Drop credit day overdue

#bureau_drop_col.append('CNT_CREDIT_PROLONG')
bureau[['AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 

        'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']].describe()
box_dist_plot_with_trans(bureau,column='AMT_CREDIT_MAX_OVERDUE',ncols=3,Trans_func='log1p')
box_dist_plot_with_trans(bureau,column='AMT_CREDIT_SUM',ncols=3,Trans_func='log1p')
# apply lo(x+1) Amount credit sum

bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].apply(np.log1p)
box_dist_plot_with_trans(bureau,column='AMT_CREDIT_SUM_DEBT',ncols=2,Trans_func=None)
# apply

bureau['AMT_CREDIT_SUM_DEBT'] = np.log1p(bureau['AMT_CREDIT_SUM_DEBT'])
box_dist_plot_with_trans(bureau,column='AMT_CREDIT_SUM_LIMIT',ncols=2)
box_dist_plot_with_trans(bureau,column='AMT_CREDIT_SUM_OVERDUE',ncols=3,Trans_func='log1p')
box_dist_plot_with_trans(bureau,column='AMT_ANNUITY',ncols=3,Trans_func='log1p')
# apply log

bureau['AMT_ANNUITY'] = np.log1p(bureau['AMT_ANNUITY'])
plt.figure(figsize= (14,6))

sns.heatmap(bureau[['AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG','AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 

        'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']].corr(), 

            cmap='viridis', annot=True)
# drop features

bureau = bureau.drop(bureau_drop_col, axis=1)



# Replace XNA XAP

bureau = replace_XNA_XAP(bureau)



# columns segrigation

bureau_cat_col = bureau.select_dtypes('object').columns

bureau_numeric_col = bureau.select_dtypes('number').columns

bureau_numeric_col = bureau_numeric_col.drop(['SK_ID_CURR', 'SK_ID_BUREAU'])



## binary encoding

bureau_bin =[]

bureau, bureau_bin = binary_encoding(bureau, bureau_cat_col)



### one hot encoding

bureau,bureau_cat = one_hot_encoding(bureau,bureau_cat_col,nan_as_category=False)



# join bb_agg

bureau = bureau.join(bb_agg,how='left',on ='SK_ID_BUREAU')

bureau.drop('SK_ID_BUREAU',axis = 1,inplace=True)



# bureau and bureau_balance numeric feature

num_aggregators = {

    'DAYS_CREDIT': ['min','max','var'],

    'DAYS_CREDIT_ENDDATE': ['min','max','mean'],

    'DAYS_ENDDATE_FACT': ['mean'],

    'CREDIT_DAY_OVERDUE':['max','mean'],

    'AMT_CREDIT_SUM':['mean','sum'],

    'AMT_CREDIT_SUM_DEBT':['sum','max','mean'],

    'AMT_CREDIT_SUM_LIMIT':['sum','mean'],

    'AMT_CREDIT_SUM_OVERDUE':['mean'],

    'AMT_ANNUITY':['mean','max'],

    'MONTHS_BALANCE_MIN':['min'],

    'MONTHS_BALANCE_MAX':['max'],

    'MONTHS_BALANCE_MEAN':['mean']

}



#bureau and bueau_balance categorical feature

cat_aggregates = {}

for col in bureau_cat+bureau_bin:

    cat_aggregates[col] = ['sum','mean']

for col in bb_cat:

    cat_aggregates[col+'_MEAN'] = ['mean']

    cat_aggregates[col+'_SUM'] = ['sum']



bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregators, **cat_aggregates})

bureau_agg.columns = pd.Index(['BUREAU_'+e[0]+'_'+e[1].upper() for e in bureau_agg.columns.tolist()])



# Bureau active credict using numerical columns

# Credit active status active

# Has one hot encoding removed active column so closed ==0 means active

bureau_agg = bureau_agg.reset_index()

active = bureau[bureau['CREDIT_ACTIVE_Closed'] ==0] 

active_agg = active.groupby('SK_ID_CURR').agg(num_aggregators)

active_agg.columns = pd.Index(['ACTIVE_'+ e[0]+'_'+e[1].upper() for e in active_agg.columns.tolist()])

bureau_agg = bureau_agg.join(active_agg, on ='SK_ID_CURR', how='left')

del active, active_agg



# Credit active status closed

closed = bureau[bureau['CREDIT_ACTIVE_Closed'] ==1]

closed_agg = bureau.groupby('SK_ID_CURR').agg(num_aggregators)

closed_agg.columns = pd.Index(['CLOSED_'+ e[0]+'_'+e[1].upper()for e in closed_agg.columns.tolist()])

bureau_agg = bureau_agg.join(closed_agg, on = 'SK_ID_CURR', how='left')

del closed_agg, closed

del bureau,bureau_balance,bb_agg

gc.collect()

bureau_agg = bureau_agg.set_index('SK_ID_CURR')

bureau_agg.head()
reduce_memory_usage(bureau_agg)
POS_CASH_balance.head()
basic_details(POS_CASH_balance)
box_dist_plot_with_trans(POS_CASH_balance,column='MONTHS_BALANCE',)
box_dist_plot_with_trans(POS_CASH_balance,column='CNT_INSTALMENT',)
box_dist_plot_with_trans(POS_CASH_balance, column='CNT_INSTALMENT_FUTURE', )
sns.countplot(POS_CASH_balance['NAME_CONTRACT_STATUS'])
box_dist_plot_with_trans(POS_CASH_balance, 'SK_DPD')
box_dist_plot_with_trans(POS_CASH_balance,'SK_DPD_DEF')
# Replace XNA, XAP with np.nan

POS_CASH_balance = replace_XNA_XAP(POS_CASH_balance)

# Binary encoding

pos_bin =[]

POS_CASH_balance,pos_bin = binary_encoding(POS_CASH_balance,columns=['NAME_CONTRACT_STATUS'])



# One hot encoding

POS_CASH_balance,pos_cat = one_hot_encoding(POS_CASH_balance,columns=['NAME_CONTRACT_STATUS'], nan_as_category=True)



# Aggregate

pos_aggregate ={

    'MONTHS_BALANCE':['mean','min','max'],

    'CNT_INSTALMENT':['mean','min','max'],

    'CNT_INSTALMENT_FUTURE':['mean','min','max'],

    'SK_DPD':['min','max','mean'],

    'SK_DPD_DEF':['min','max','mean']

}

for col in pos_cat+pos_bin:

    pos_aggregate[col] =['sum','mean']

pos_agg = POS_CASH_balance.groupby('SK_ID_CURR').agg(pos_aggregate)

pos_agg.columns = pd.Index(['POS_'+ e[0]+ '_'+ e[1].upper() for e in pos_agg.columns.tolist()])

# Count pos transcations

pos_agg['POS_COUNT'] = POS_CASH_balance.groupby('SK_ID_CURR').size()

del POS_CASH_balance

pos_agg.head(2)
gc.collect()

reduce_memory_usage(pos_agg)
print('Number of rows and columns in dataset:',credit_card_balance.shape)

credit_card_balance.head()
basic_details(credit_card_balance)
credit_card_balance.columns
box_dist_plot_with_trans(credit_card_balance, 'MONTHS_BALANCE')
box_dist_plot_with_trans(credit_card_balance,'AMT_BALANCE')
box_dist_plot_with_trans(credit_card_balance, 'AMT_DRAWINGS_CURRENT')
box_dist_plot_with_trans(credit_card_balance, 'AMT_CREDIT_LIMIT_ACTUAL',ncols=3,Trans_func='sqrt')
# apply sqrt

credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL'] = np.sqrt(credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL'])
box_dist_plot_with_trans(credit_card_balance, 'AMT_DRAWINGS_OTHER_CURRENT', ncols=3, Trans_func='log1p')
box_dist_plot_with_trans(credit_card_balance, 'AMT_DRAWINGS_ATM_CURRENT', ncols=2)
box_dist_plot_with_trans(credit_card_balance, 'AMT_DRAWINGS_POS_CURRENT', ncols=2,)
box_dist_plot_with_trans(credit_card_balance, 'AMT_INST_MIN_REGULARITY')
box_dist_plot_with_trans(credit_card_balance, 'AMT_PAYMENT_CURRENT',)
box_dist_plot_with_trans(credit_card_balance, 'AMT_RECEIVABLE_PRINCIPAL')
box_dist_plot_with_trans(credit_card_balance, 'AMT_RECIVABLE')
box_dist_plot_with_trans(credit_card_balance, 'AMT_TOTAL_RECEIVABLE', )
box_dist_plot_with_trans(credit_card_balance, 'CNT_DRAWINGS_CURRENT', ncols=3, Trans_func='sqrt')
credit_card_balance['CNT_DRAWINGS_CURRENT'] = np.sqrt(credit_card_balance['CNT_DRAWINGS_CURRENT'])
box_dist_plot_with_trans(credit_card_balance, 'CNT_DRAWINGS_ATM_CURRENT',)
plt.figure(figsize=(14,4))

sns.countplot(credit_card_balance['CNT_DRAWINGS_OTHER_CURRENT'],palette='magma')
# convert ot category

credit_card_balance['CNT_DRAWINGS_OTHER_CURRENT'] = credit_card_balance['CNT_DRAWINGS_OTHER_CURRENT'].astype('object')
box_dist_plot_with_trans(credit_card_balance, 'CNT_DRAWINGS_POS_CURRENT', Trans_func='sqrt', ncols=3)
# Apply square root

credit_card_balance['CNT_DRAWINGS_POS_CURRENT'] = np.sqrt(credit_card_balance['CNT_DRAWINGS_POS_CURRENT'])
box_dist_plot_with_trans(credit_card_balance, 'CNT_INSTALMENT_MATURE_CUM')
# Apply square root

credit_card_balance['CNT_INS'] = np.sqrt(credit_card_balance['CNT_INSTALMENT_MATURE_CUM'])
box_dist_plot_with_trans(credit_card_balance, 'SK_DPD')
box_dist_plot_with_trans(credit_card_balance, 'SK_DPD_DEF')
plt.figure(figsize=(14,4))

sns.countplot(credit_card_balance['NAME_CONTRACT_STATUS'],palette='magma')
credit_card_balance['NAME_CONTRACT_STATUS'] = credit_card_balance['NAME_CONTRACT_STATUS'].astype('object')
# Replace XNA, XAP

credit_card_balance = replace_XNA_XAP(credit_card_balance)



# Column segrigation

credit_object_col = credit_card_balance.select_dtypes('object').columns



# Binary encoding 

credit_bin =[]

credit_card_balance,credit_bin = binary_encoding(credit_card_balance,credit_object_col)



## One hot encoding

credit_card_balance,credit_cat = one_hot_encoding(credit_card_balance,credit_object_col, nan_as_category=False)



# General aggregation

credit_aggregation = {

    'MONTHS_BALANCE':['mean','min','max'],

    'AMT_BALANCE':['mean','min'],

    'AMT_CREDIT_LIMIT_ACTUAL':['mean','min'], 

    'AMT_DRAWINGS_ATM_CURRENT':['mean','min','max'],

    'AMT_DRAWINGS_CURRENT':['mean','min','max'], 

    'AMT_DRAWINGS_OTHER_CURRENT':['mean','min','max'],

    'AMT_DRAWINGS_POS_CURRENT':['mean','min','max'], 

    'AMT_INST_MIN_REGULARITY':['mean','min','max'],

    'AMT_PAYMENT_CURRENT':['mean','min','max'], 

    'AMT_PAYMENT_TOTAL_CURRENT':['mean','min','max'],

    'AMT_RECEIVABLE_PRINCIPAL':['mean','min','max'], 

    'AMT_RECIVABLE':['mean','min','max'], 

    'AMT_TOTAL_RECEIVABLE':['mean','min','max'],

    'CNT_DRAWINGS_ATM_CURRENT':['mean','min','max'], 

    'CNT_DRAWINGS_CURRENT':['mean','min','max'],

    'CNT_DRAWINGS_POS_CURRENT':['mean','min','max'],

    'CNT_INSTALMENT_MATURE_CUM':['mean','min','max'], 

    'SK_DPD':['mean','min','max'],

    'SK_DPD_DEF':['mean','min','max'],

    }



for col in credit_cat+credit_bin:

    credit_aggregation[col] = ['mean','sum']



#Credit_card_balance.drop('SK_ID_PREV',axis=1,inplace=True)

credit_agg = credit_card_balance.groupby('SK_ID_CURR').agg(credit_aggregation)

credit_agg.columns = pd.Index(['CREDIT_'+e[0]+'_'+ e[1].upper() for e in credit_agg.columns.tolist()])

# Count credit card transactions

credit_agg['CREDIT_COUNT'] = credit_card_balance.groupby('SK_ID_CURR').size()

del credit_card_balance



credit_agg.head(2)
previous_application.head()
prev_cat_col = previous_application.select_dtypes(include='object').columns

int_col = [i for i in previous_application.columns.values if i not in prev_cat_col]
basic_details(previous_application[prev_cat_col])
basic_details(previous_application[int_col])
previous_application.columns
f = plt.figure(figsize=(14,4))

ax = f.add_subplot(121)

sns.countplot(previous_application['NAME_CONTRACT_TYPE'], ax= ax)
box_dist_plot_with_trans(previous_application,'AMT_ANNUITY', ncols=3,Trans_func='log1p')
# apply log on both side

previous_application['AMT_ANNUITY'] = np.log1p(previous_application['AMT_ANNUITY'])
box_dist_plot_with_trans(previous_application,'AMT_APPLICATION',Trans_func='sqrt', ncols=3)
# apply sqrt

previous_application['AMT_APPLICATION'] = np.sqrt(previous_application['AMT_APPLICATION'])
box_dist_plot_with_trans(previous_application, 'AMT_CREDIT', ncols=3,Trans_func='sqrt')
# apply sqrt

previous_application['AMT_CREDIT'] = np.sqrt(previous_application['AMT_CREDIT'])
box_dist_plot_with_trans(previous_application, 'AMT_DOWN_PAYMENT', ncols=3,Trans_func='log1p')
# apply log1p

previous_application['AMT_DOWN_PAYMENT'] = np.log1p(previous_application['AMT_DOWN_PAYMENT'])
box_dist_plot_with_trans(previous_application, 'AMT_GOODS_PRICE',ncols=3,Trans_func='log1p')
# apply sqrt

previous_application['AMT_GOODS_PRICE'] = np.log1p(previous_application['AMT_GOODS_PRICE'])
f = plt.figure(figsize=(14,8))

ax = f.add_subplot(221)

sns.countplot(previous_application['WEEKDAY_APPR_PROCESS_START'], ax =ax)

plt.xticks(rotation=90)



ax = f.add_subplot(222)

sns.countplot(previous_application['HOUR_APPR_PROCESS_START'], ax =ax)



ax = f.add_subplot(223)

sns.countplot(previous_application['FLAG_LAST_APPL_PER_CONTRACT'], ax =ax)



ax = f.add_subplot(224)

sns.countplot(previous_application['NFLAG_INSURED_ON_APPROVAL'], ax =ax)



plt.tight_layout()
# convert to object

previous_application[['HOUR_APPR_PROCESS_START','NFLAG_INSURED_ON_APPROVAL']] = previous_application[['HOUR_APPR_PROCESS_START','NFLAG_INSURED_ON_APPROVAL']].astype('object')
box_dist_plot_with_trans(previous_application, 'RATE_DOWN_PAYMENT')
box_dist_plot_with_trans(previous_application, 'RATE_INTEREST_PRIMARY',)
box_dist_plot_with_trans(previous_application, 'RATE_INTEREST_PRIVILEGED')
box_dist_plot_with_trans(previous_application,'DAYS_DECISION')
f = plt.figure(figsize=(14,8))

ax = f.add_subplot(221)

sns.countplot(previous_application['CODE_REJECT_REASON'], ax =ax)



ax = f.add_subplot(222)

sns.countplot(previous_application['NAME_CONTRACT_STATUS'], ax =ax)



ax = f.add_subplot(223)

sns.countplot(previous_application['NAME_PAYMENT_TYPE'], ax =ax)

plt.xticks(rotation=90)



ax = f.add_subplot(224)

sns.countplot(previous_application['NAME_CASH_LOAN_PURPOSE'], ax =ax)

plt.xticks(rotation=90)



plt.tight_layout()
f = plt.figure(figsize=(14,8))

ax = f.add_subplot(221)

sns.countplot(previous_application['NAME_PORTFOLIO'], ax =ax)



ax = f.add_subplot(222)

sns.countplot(previous_application['NAME_CLIENT_TYPE'], ax =ax)



ax = f.add_subplot(223)

sns.countplot(previous_application['NAME_GOODS_CATEGORY'], ax =ax)

plt.xticks(rotation=90)



ax = f.add_subplot(224)

sns.countplot(previous_application['NAME_TYPE_SUITE'], ax =ax)

plt.xticks(rotation=90)



plt.tight_layout()
f = plt.figure(figsize=(14,8))

ax = f.add_subplot(221)

sns.countplot(previous_application['NAME_PRODUCT_TYPE'], ax =ax)



ax = f.add_subplot(222)

sns.countplot(previous_application['NAME_YIELD_GROUP'], ax =ax)



ax = f.add_subplot(223)

sns.countplot(previous_application['CHANNEL_TYPE'], ax =ax)

plt.xticks(rotation=90)



ax = f.add_subplot(224)

sns.countplot(previous_application['NAME_SELLER_INDUSTRY'], ax =ax)

plt.xticks(rotation=90)



plt.tight_layout()
box_dist_plot_with_trans(previous_application, 'SELLERPLACE_AREA',)
previous_application[previous_application['SELLERPLACE_AREA']>500000]
plt.figure(figsize=(14,4))

sns.countplot(previous_application['CNT_PAYMENT'])

plt.xticks(rotation=90);
plt.figure(figsize=(14,4))

sns.countplot(previous_application['PRODUCT_COMBINATION'])

plt.xticks(rotation=90);
previous_application[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',

       'DAYS_LAST_DUE', 'DAYS_TERMINATION']].describe()
## Fill missing value

previous_application['DAYS_FIRST_DRAWING'].replace({365243:np.nan},inplace=True)

previous_application['DAYS_FIRST_DUE'].replace({365243:np.nan},inplace=True)

previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace({365243:np.nan},inplace=True)

previous_application['DAYS_LAST_DUE'].replace({365243:np.nan},inplace=True)

previous_application['DAYS_TERMINATION'].replace({365243:np.nan},inplace=True)

box_dist_plot_with_trans(previous_application,'DAYS_FIRST_DRAWING',)
box_dist_plot_with_trans(previous_application,'DAYS_FIRST_DUE')
box_dist_plot_with_trans(previous_application,'DAYS_LAST_DUE_1ST_VERSION')
box_dist_plot_with_trans(previous_application,'DAYS_LAST_DUE',)
box_dist_plot_with_trans(previous_application, 'DAYS_TERMINATION')
sns.countplot(previous_application['NFLAG_INSURED_ON_APPROVAL'])
# Replace XNA XAP

previous_application = replace_XNA_XAP(previous_application)



# Binary encoding

prev_bin =[]

previous_application,prev_bin = binary_encoding(previous_application,prev_cat_col)



# One hot encoding

previous_application,prev_cat = one_hot_encoding(previous_application,columns= prev_cat_col,nan_as_category=True)



prev_aggregate = {

    'AMT_ANNUITY':['mean','sum','min'],

    'AMT_APPLICATION':['mean'],

    'AMT_CREDIT':['mean','min','max'],

    'AMT_DOWN_PAYMENT':['mean','min'],

    'AMT_GOODS_PRICE':['mean','sum','max'],

    'HOUR_APPR_PROCESS_START':['mean','min'],

     'NFLAG_LAST_APPL_IN_DAY': ['mean'],

     'RATE_DOWN_PAYMENT': ['mean'],

     'RATE_INTEREST_PRIMARY':['mean','min','max'],

     'RATE_INTEREST_PRIVILEGED':['mean','min'],

     'DAYS_DECISION':['mean'],

     'SELLERPLACE_AREA':['mean'],

     'CNT_PAYMENT':['mean','sum'],

     'DAYS_FIRST_DRAWING':['mean','min'],

     'DAYS_FIRST_DUE':['mean','min'],

     'DAYS_LAST_DUE_1ST_VERSION':['mean','min'],

     'DAYS_LAST_DUE':['mean','max'],

     'DAYS_TERMINATION':['mean','max'],

     'NFLAG_INSURED_ON_APPROVAL' : ['mean'],

}



cat_prev_aggregate = {}

for col in prev_cat+prev_bin:

    cat_prev_aggregate[col] =['mean','sum']



prev_agg = previous_application.groupby('SK_ID_CURR').agg({**prev_aggregate,**cat_prev_aggregate})

prev_agg.columns = pd.Index(['PREV_'+e[0]+ '_'+ e[1].upper() for e in prev_agg.columns.tolist()])



# Prevoius application 

# Previous application refused

prev_agg = prev_agg.reset_index()

refused = previous_application[previous_application['NAME_CONTRACT_STATUS_Refused'] ==1]

refused_agg = refused.groupby('SK_ID_CURR').agg(prev_aggregate)

refused_agg.columns = pd.Index(['REFUSE_'+e[0]+'_'+ e[1].upper() for e in refused_agg.columns.tolist()])

prev_agg = prev_agg.join(refused_agg, on='SK_ID_CURR', how='left')

del refused, refused_agg



# Previous application canceled

canceled = previous_application[previous_application['NAME_CONTRACT_STATUS_Canceled']==1]

canceled_agg = canceled.groupby('SK_ID_CURR').agg(prev_aggregate)

canceled_agg.columns = pd.Index(['CANC_'+ e[0]+ '_'+ e[1].upper() for e in canceled_agg.columns.tolist()])

prev_agg = prev_agg.join(canceled_agg, on='SK_ID_CURR', how='left')

del canceled, canceled_agg



# Previous application unused

#unused = previous_application[previous_application['NAME_CONTRACT_STATUS_Unused']==1]

#unused_agg = canceled.groupby('SK_ID_CURR').agg(prev_aggregate)

#unused_agg.columns = pd.Index(['UNUSE_'+ e[0]+ '_'+ e[1].upper() for e in unused_agg.columns.tolist()])

#prev_agg = prev_agg.join(unused_agg, on='SK_ID_CURR', how='left')

# Count prevoius application

prev_agg['PREV_COUNT'] = previous_application.groupby('SK_ID_CURR').size()

#del unused, unused_agg

prev_agg = prev_agg.set_index('SK_ID_CURR')

del previous_application

prev_agg.head()
installments_payments.head()
basic_details(installments_payments)
box_dist_plot_with_trans(installments_payments, 'NUM_INSTALMENT_VERSION', ncols=3, Trans_func='sqrt')
box_dist_plot_with_trans(installments_payments,'NUM_INSTALMENT_NUMBER', Trans_func='sqrt', ncols=3)
box_dist_plot_with_trans(installments_payments,'AMT_INSTALMENT', Trans_func='sqrt',ncols=3)
box_dist_plot_with_trans(installments_payments, 'AMT_PAYMENT', Trans_func='sqrt', ncols=3)
box_dist_plot_with_trans(installments_payments,'DAYS_INSTALMENT',)
box_dist_plot_with_trans(installments_payments,'DAYS_ENTRY_PAYMENT')
plt.figure(figsize= (14,4))

sns.heatmap(installments_payments.corr(), cmap = 'magma',annot=True)
# aggregate numeric variable

inst_aggregator = {

    'NUM_INSTALMENT_VERSION':['mean'],

    'NUM_INSTALMENT_NUMBER':['mean','max'],

    'DAYS_INSTALMENT':['min','mean'],

    'AMT_INSTALMENT':['mean'],

    'AMT_PAYMENT':['mean']

}



inst_agg = installments_payments.groupby('SK_ID_CURR').agg(inst_aggregator)

inst_agg.columns = pd.Index(['INST_'+e[0]+ '_'+ e[1].upper() for e in inst_agg.columns.tolist()])

# Count instalment

inst_agg['INST_COUNT'] = installments_payments.groupby('SK_ID_CURR').size()

del installments_payments

inst_agg.head()
# Join all aggregated df with train_test

train_test = train_test.join(bureau_agg,how='left',on='SK_ID_CURR') # bureau

del bureau_agg

train_test = train_test.join(pos_agg, how='left',on='SK_ID_CURR') # POS_CASH

del pos_agg

train_test = train_test.join(inst_agg,how='left',on='SK_ID_CURR') # install

del inst_agg

train_test = train_test.join(credit_agg, how='left',on='SK_ID_CURR') # credit card

del credit_agg

train_test = train_test.join(prev_agg, how='left',on='SK_ID_CURR') # previous

del prev_agg

reduce_memory_usage(train_test)
train_test.head(3)
# Select columns whose variance > 0

col = train_test.columns

feat = train_test.columns[train_test.var() >0]

train_test = train_test[feat]

print('Number of columns generated:',len(col))

print('Number of columns removed')

len(feat) - len(col)
#from sklearn.model_selection import train_test_split

col_drop = ['TARGET','SK_ID_CURR']

X = train_test[train_test['TARGET'].notnull()].drop(col_drop, axis=1)

y = train_test[train_test['TARGET'].notnull()]['TARGET']

test_new = train_test[train_test['TARGET'].isnull()].drop(col_drop, axis=1)



#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=seed)
# Lightgbm



def model(X_train, X_valid, y_train, y_valid,test_new,random_seed):

    

    lg_param = {}

    lg_param['learning_rate'] = 0.02

    lg_param['n_estimators'] = 10000

    lg_param['max_depth'] = 8

    lg_param['num_leaves'] = 34

    lg_param['boosting_type'] = 'gbdt'

    lg_param['feature_fraction'] = 0.9

    lg_param['bagging_fraction'] = 0.9

    lg_param['min_child_samples'] = 30

    lg_param['lambda_l1'] = 0.04

    lg_param['lambda_l2'] = 0.08

    lg_param['silent'] = -1

    lg_param['verbose'] = -1

    lg_param['nthread'] = 4

    lg_param['seed'] = random_seed

    

    lgb_model = lgb.LGBMClassifier(**lg_param)

    print('-'*10,'*'*20,'-'*10)

    lgb_model.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_valid,y_valid)], 

                 eval_metric ='auc', verbose =100, early_stopping_rounds=200)

    y_pred = lgb_model.predict_proba(X_valid)[:,1]

    print('roc_auc_score',roc_auc_score(y_valid,y_pred),'-'*30,i+1)

    y_pred_new = lgb_model.predict_proba(test_new)[:,1]

    return y_pred,y_pred_new,lgb_model
# KFold cross validation

kf = KFold(n_splits=3, shuffle=True, random_state=seed)



#y_pred = 0

y_pred_new = 0



for i,(train_index, valid_index) in enumerate(kf.split(X,y)):    

    X_train, X_valid = X.loc[train_index], X.loc[valid_index]

    y_train, y_valid = y[train_index], y[valid_index]

    print('\n{} fold of {} KFold'.format(i+1,kf.n_splits))

    y_pred,y_pred2,lgb_model = model(X_train, X_valid, y_train, y_valid,test_new,random_seed = i)

    #y_pred += y_pred1

    y_pred_new += y_pred2

lgb.plot_importance(lgb_model,max_num_features=20)
feat_impo = pd.DataFrame({'Columns':X.columns,'Importance':lgb_model.feature_importances_})

feat_impo.sort_values('Importance',ascending=False).head()

feat_impo.to_csv('feat_impo.csv',index=False)
feat_impo.sort_values('Importance',ascending=True).head()         
fpr,tpr,threshold =roc_curve(y_valid, y_pred)

plt.figure(figsize= (10,6))

auc_value = round(auc(fpr,tpr),4)

plt.text(0.9,0,'AUC:'+str(auc_value),color='r')

plt.plot(fpr,tpr, 'r-.',label='roc')

plt.plot([0,1],[0,1],'b-')

plt.xlabel('True positive rate')

plt.ylabel('False positive rate')

plt.title('Reciver Operating Characteristics')
submit  = pd.DataFrame({'SK_ID_CURR':test_index,'TARGET':y_pred_new/kf.n_splits})

submit.to_csv('home_credit.csv',index=False)

submit.head()
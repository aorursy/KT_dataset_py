import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV

from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer



import datetime as dt



% matplotlib inline

seed = 129
path = '../input/'

#path = ''

train = pd.read_csv(path+'train_v2.csv',dtype={'is_churn':np.int8})

test = pd.read_csv(path+'sample_submission_v2.csv',dtype={'is_churn':np.int8})

members = pd.read_csv(path+'members_v3.csv',parse_dates=['registration_init_time'],dtype={'city':np.int8,'bd':np.int8,

                                                                                         'registered_via':np.int8})

transactions = pd.read_csv(path+'transactions_v2.csv',parse_dates=['transaction_date','membership_expire_date'],

                          dtype={'payment_method_id':np.int8,'payment_plan_days':np.int8,'plan_list_price':np.int8,

                                'actual_amount_paid':np.int8,'is_auto_renew':np.int8,'is_cancel':np.int8})



user_log = pd.read_csv(path+'user_logs_v2.csv',parse_dates=['date'],dtype={'num_25':np.int16,'num_50':np.int16,

                                    'num_75':np.int16,'num_985':np.int16,'num_100':np.int16,'num_unq':np.int16})
print('Number of rows  & columns',train.shape)

train.head()
print('Number of rows  & columns',test.shape)

test.head()
print('Number of rows  & columns',members.shape)

members.head()
print('Number of rows & columns',transactions.shape)

transactions.head()
print('Number of rows & columns',user_log.shape)

user_log.head()
print('\nTrain:',train.describe().T)

print('\nTest:',test.describe().T)

print('\nMembers:',members.describe().T)

print('\nTransactions:',transactions.describe().T)

print('\nUser log:',user_log.describe().T)

train = pd.merge(train,members,on='msno',how='left')

test = pd.merge(test,members,on='msno',how='left')

train = pd.merge(train,transactions,how='left',on='msno',left_index=True, right_index=True)

test = pd.merge(test,transactions,how='left',on='msno',left_index=True, right_index=True,)

train = pd.merge(train,user_log,how='left',on='msno',left_index=True, right_index=True)

test = pd.merge(test,user_log,how='left',on='msno',left_index=True, right_index=True)



del members,transactions,user_log

print('Number of rows & columns',train.shape)

print('Number of rows & columns',test.shape)

train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].describe()
train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].isnull().sum()
train['registration_init_time'] = train['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))

test['registration_init_time'] = test['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))
def date_feature(df):

    

    col = ['registration_init_time' ,'transaction_date','membership_expire_date','date']

    var = ['reg','trans','mem_exp','user_']

    #df['duration'] = (df[col[1]] - df[col[0]]).dt.days 

    

    for i ,j in zip(col,var):

        df[j+'_day'] = df[i].dt.day.astype('uint8')

        df[j+'_weekday'] = df[i].dt.weekday.astype('uint8')        

        df[j+'_month'] = df[i].dt.month.astype('uint8') 

        df[j+'_year'] =df[i].dt.year.astype('uint16') 



date_feature(train)

date_feature(test)
train.columns
train.isnull().sum()
train.info()
col = [ 'city', 'bd', 'gender', 'registered_via']

def missing(df,columns):

    col = columns

    for i in col:

        df[i].fillna(df[i].mode()[0],inplace=True)



missing(train,col)

missing(test,col)
def unique_value(df):

    col = df.columns

    for i in col:

        print('Number of unique value in {} is {}'.format(i,df[i].nunique()))



unique_value(train)
plt.figure(figsize=(8,6))

sns.set_style('ticks')

sns.countplot(train['is_churn'],palette='summer')

plt.xlabel('The subscription within 30 days of expiration is True/False')
print(train['city'].unique())

fig,ax = plt.subplots(2,2,figsize=(16,8))

ax1,ax2,ax3,ax4 = ax.flatten()



sns.set(style="ticks")

sns.countplot(train['city'],palette='summer',ax=ax1)

#ax1.set_yscale('log')



ax1.set_xlabel('City')

#ax1.set_xticks(rotation=45)



sns.countplot(x='gender',data = train,palette='winter',ax=ax2)

#ax2.set_yscale('log')

ax2.set_xlabel('Gender')



sns.countplot(x='registered_via',data=train,palette='winter',ax=ax3)

#ax3.set_yscale('')

ax3.set_xlabel('Register via')



sns.countplot(x='payment_method_id',data= train,palette='winter',ax=ax4)

ax4.set_xlabel('Payment_method_id')



print(train['bd'].describe())
fig,ax = plt.subplots(1,2,figsize=(16,8))

ax1,ax2 = ax.flatten()

sns.set_style('ticks')

sns.distplot(train['bd'].fillna(train['bd'].mode()[0]),bins=100,color='r',ax=ax1)

plt.title('Distribution of birth day')
plt.figure(figsize=(14,6))

sns.distplot(train.loc[train['bd'].value_counts()]['bd'].fillna(0),bins=50,color='b')
print(pd.crosstab(train['is_churn'],train['gender']))
regi = train.groupby('registration_init_time').count()['is_churn']

plt.subplot(211)

plt.plot(regi,color='b',label='count')

plt.legend(loc='center')

regi = train.groupby('registration_init_time').mean()['is_churn']

plt.subplot(212)

plt.plot(regi,color='r',label='mean')

plt.legend(loc='center')

plt.tight_layout()
regi = train.groupby('registration_init_time').mean()['is_churn']

plt.figure(figsize=(14,6))

sns.distplot(regi,bins=100,color='r')
fig,ax = plt.subplots(2,2,figsize=(16,8))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.countplot(train['reg_day'],palette='Set2',ax=ax1)

sns.countplot(data=train,x='reg_month',palette='Set1',ax=ax2)

sns.countplot(data=train,x='reg_year',palette='magma',ax=ax3)

cor = train.corr()

plt.figure(figsize=(16,12))

sns.heatmap(cor,cmap='Set1',annot=False)

plt.xticks(rotation=45);
le = LabelEncoder()

train['gender'] = le.fit_transform(train['gender'])

test['gender'] = le.fit_transform(test['gender'])
def OHE(df):

    #col = df.select_dtypes(include=['category']).columns

    col = ['city','gender','registered_via']

    print('Categorical columns in dataset',col)

    

    c2,c3 = [],{}

    for c in col:

        if df[c].nunique()>2 :

            c2.append(c)

            c3[c] = 'ohe_'+c

    

    df = pd.get_dummies(df,columns=c2,drop_first=True,prefix=c3)

    print(df.shape)

    return df

train1 = OHE(train)

test1 = OHE(test)
train1.columns
unwanted = ['msno','is_churn','registration_init_time','transaction_date','membership_expire_date','date']



X = train1.drop(unwanted,axis=1)

y = train1['is_churn'].astype('category')

x_test = test1.drop(unwanted,axis=1)

lr = LogisticRegression(class_weight='balanced',C=1)

lr.fit(X,y)

y_pred = lr.predict_proba(x_test)[:,1]

lr.score(X,y)
y_proba = lr.predict_proba(X)[:,1]

fpr,tpr,th = roc_curve(y,y_proba)



plt.figure(figsize=(14,6))

plt.plot(fpr,tpr,color='r')

plt.plot([0,1],[0,1],color='b')

plt.title('Reciever operating Charactaristics')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')
#y_pred = pred_test_full/5

submit = pd.DataFrame({'msno':test['msno'],'is_churn':y_pred})

submit.to_csv('kk_pred.csv',index=False)

#submit.to_csv('kk_pred.csv.gz',index=False,compression='gzip')
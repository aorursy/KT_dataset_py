# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_user = pd.read_csv('../input/train_users_2.csv')
countries = pd.read_csv('../input/countries.csv')

age_gender = pd.read_csv('../input/age_gender_bkts.csv')

sessions = pd.read_csv('../input/sessions.csv')
sns.countplot(x='country_destination', data=train_user)



#there  are large number of undefined countries with small other counties

#gender of most is unknown with very small unknown
sns.countplot(x='gender' , data=train_user)
sns.countplot(x='signup_flow' , data=train_user)
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

device_percentage = train_user.first_device_type.value_counts() / train_user.shape[0] * 100

device_percentage.plot(kind='bar',color='#196F3D')

plt.xlabel('Device used by user')

plt.ylabel('Percentage')

sns.despine()
sns.distplot(train_user.age.dropna(), color='#16A085')
train_user['corrected_age']=train_user['age'].apply(lambda x : 36 if x>100 else x)

sns.distplot(train_user.corrected_age.dropna(), color='#16A085')

display((train_user.isnull().sum()/train_user.shape[0])*100)
train_user['date_account_created_new'] = pd.to_datetime(train_user['date_account_created'])

train_user['date_first_active_new'] = pd.to_datetime((train_user.timestamp_first_active // 1000000), format='%Y%m%d')
train_user['date_account_created_day'] = train_user.date_account_created_new.dt.weekday_name

train_user['date_account_created_month'] = train_user.date_account_created_new.dt.month

train_user['date_account_created_year'] = train_user.date_account_created_new.dt.year

train_user['date_first_active_day'] = train_user.date_first_active_new.dt.weekday_name

train_user['date_first_active_month'] = train_user.date_first_active_new.dt.month

train_user['date_first_active_year'] = train_user.date_first_active_new.dt.year
train_user['isequal']=~(train_user['date_account_created_new'] == train_user['date_first_active_new'])

#there are 178 rows whose values are not same for date_account_created adn date_first_active
#display(train_user.loc[train_user['isequal']==True])
#sns.countplot(x='country_destination', data=train_user.loc[train_user['isequal']==False])

#if both value are equal then there are greater changes of travel for value which are equal but 

# 213273 values are same and only 178 rows are different so they are reducible

display(train_user.head())

train_user['isequal'].value_counts()/train_user.shape[0]*100
train_user['gender'].value_counts()/train_user.shape[0]*100
train_user['signup_method'].value_counts()/train_user.shape[0]*100
train_user['language'].value_counts()/train_user.shape[0]*100
train_user['affiliate_channel'].value_counts()/train_user.shape[0]*100

train_user['affiliate_provider'].value_counts()/train_user.shape[0]*100
train_user['first_affiliate_tracked'].value_counts()/train_user.shape[0]*100
#Dealing with first_afflicated_data and flling with untracked as it has most unfilled data

def definefirstaffiliate(cols):

    if pd.isnull(cols[0]):

        return 'untracked'

    else:

        return cols[0]

    
train_user['first_affiliate_tracked_updated'] = train_user[['first_affiliate_tracked']].apply( definefirstaffiliate , axis = 1) 
train_user['first_affiliate_tracked_updated'].value_counts()/train_user.shape[0]*100

#cleared affiliated data
#current plan is to drop the date_first_booking

#something to do with age

#thinking maybe predicting the age with logistic regresson can work

#thinking filing them with 36 presently can work

display(train_user.sample(10))
def defineage(cols):

    if pd.isnull(cols[0]):

        return 36

    else:

        return cols[0]

    
train_user['corrected_age'] = train_user[['corrected_age']].apply( defineage , axis = 1) 
display(train_user.sample(15))

#creating dummy variable

gender = pd.get_dummies(train_user['gender'] ,prefix='gender')
gender.sample(5)
gender['gender_combine']=(gender['gender_-unknown-']) | (gender['gender_OTHER'])
gender = gender.drop(['gender_-unknown-','gender_OTHER'],axis=1)

gender.sample(10)
signup_method = pd.get_dummies(train_user['signup_method'] ,prefix='signup_method')

affiliate_channel = pd.get_dummies(train_user['affiliate_channel'] ,prefix='affiliate_channel')

affiliate_provider = pd.get_dummies(train_user['affiliate_provider'] ,prefix='affiliate_provider')
affiliate_provider.sample(10)
affiliate_provider['affiliate_provider_combine']= affiliate_provider['affiliate_provider_vast'] | affiliate_provider['affiliate_provider_padmapper'] | affiliate_provider['affiliate_provider_facebook-open-graph'] | affiliate_provider['affiliate_provider_yahoo'] | affiliate_provider['affiliate_provider_gsp'] | affiliate_provider['affiliate_provider_meetup'] | affiliate_provider['affiliate_provider_email-marketing']  | affiliate_provider['affiliate_provider_naver'] | affiliate_provider['affiliate_provider_baidu']|affiliate_provider['affiliate_provider_yandex']|affiliate_provider['affiliate_provider_wayn'] | affiliate_provider['affiliate_provider_daum'] 
affiliate_provider = affiliate_provider.drop(['affiliate_provider_vast' , 'affiliate_provider_padmapper' ,'affiliate_provider_facebook-open-graph','affiliate_provider_gsp' , 'affiliate_provider_meetup' , 'affiliate_provider_email-marketing' , 'affiliate_provider_naver' , 'affiliate_provider_baidu' , 'affiliate_provider_yandex' ,'affiliate_provider_wayn' ,'affiliate_provider_daum' ], axis = 1)  

affiliate_provider.sample(10)
first_affiliate_tracked_updated = pd.get_dummies(train_user['first_affiliate_tracked_updated'] ,prefix='first_affiliate_tracked_updated')
first_affiliate_tracked_updated['first_affiliate_tracked_updated_combine'] = first_affiliate_tracked_updated['first_affiliate_tracked_updated_marketing']|first_affiliate_tracked_updated['first_affiliate_tracked_updated_product']|first_affiliate_tracked_updated['first_affiliate_tracked_updated_local ops']
first_affiliate_tracked_updated = first_affiliate_tracked_updated.drop(['first_affiliate_tracked_updated_marketing' , 'first_affiliate_tracked_updated_product' , 'first_affiliate_tracked_updated_local ops'],axis =1)    

first_affiliate_tracked_updated.sample(10)
signup_app = pd.get_dummies(train_user['signup_app'] ,prefix='signup_app')

first_device_type = pd.get_dummies(train_user['first_device_type'] ,prefix='first_device_type')

# first_browser = pd.get_dummies(train_user['first_browser'] ,prefix='first_browser')
def definefirstbrowser(cols):

    if(cols[0]=='Chrome' or cols[0]=='Safari' or cols[0]=='Firefox' or cols[0]=='-unknown-' or cols[0]=='IE' or cols[0]=='Mobile Safari'):

        return cols[0]

    else:

        return 'Other'
train_user['first_browser_updated'] = train_user[['first_browser']].apply( definefirstbrowser , axis = 1) 
# train_user[train_user['first_browser_updated'] =='Other']

#this is working

#so creating dummy for first_browser_updated

first_browser = pd.get_dummies(train_user['first_browser_updated'] ,prefix='first_browser_updated')
first_browser.sample(5)
def languageisenglish(cols):

    if(cols[0] == 'en'):

        return 1

    else:

        return 0
train_user['languageisenglish'] = train_user[['language']].apply( languageisenglish , axis = 1) 
train_user['signup_flow'].value_counts()/train_user.shape[0]*100
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
train_user['signup_flow_updated'] = ss.fit_transform(train_user[['signup_flow']])
#so dummy tables are gender ,signup_method , affiliate_channel ,affiliate_provider ,first_affiliate_tracked_updated 

#signup_app first_device_type first_browser first_browser

train_user_combine = pd.concat([train_user,gender,signup_method,affiliate_channel ,affiliate_provider ,first_affiliate_tracked_updated ,signup_app,first_device_type,first_browser],axis=1)
train_user_combine.sample(10)
train_user_combine['date_account_created_year_updated'] =  train_user_combine['date_account_created_year'].apply(lambda x: x%2009 )   
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

lb.fit(train_user_combine['date_account_created_day'])

train_user_combine['date_account_created_day_updated'] = lb.transform(train_user_combine['date_account_created_day'])
ss1 = StandardScaler()

train_user_combine['corrected_age_updated']=ss1.fit_transform(train_user_combine[['corrected_age']])
lb1 = LabelEncoder()

lb.fit(train_user_combine['country_destination'])

train_user_combine['country_destination_updated'] = lb.transform(train_user_combine['country_destination'])
train_user_combine.sample(10)
train_user_combine_final = train_user_combine[['date_account_created_month','isequal',

       'languageisenglish', 'signup_flow_updated', 'gender_FEMALE',

       'gender_MALE', 'gender_combine', 'signup_method_basic',

       'signup_method_facebook', 'signup_method_google',

       'affiliate_channel_api', 'affiliate_channel_content',

       'affiliate_channel_direct', 'affiliate_channel_other',

       'affiliate_channel_remarketing', 'affiliate_channel_sem-brand',

       'affiliate_channel_sem-non-brand', 'affiliate_channel_seo',

       'affiliate_provider_bing', 'affiliate_provider_craigslist',

       'affiliate_provider_direct', 'affiliate_provider_facebook',

       'affiliate_provider_google', 'affiliate_provider_other',

       'affiliate_provider_yahoo', 'affiliate_provider_combine',

       'first_affiliate_tracked_updated_linked',

       'first_affiliate_tracked_updated_omg',

       'first_affiliate_tracked_updated_tracked-other',

       'first_affiliate_tracked_updated_untracked',

       'first_affiliate_tracked_updated_combine', 'signup_app_Android',

       'signup_app_Moweb', 'signup_app_Web', 'signup_app_iOS',

       'first_device_type_Android Phone', 'first_device_type_Android Tablet',

       'first_device_type_Desktop (Other)', 'first_device_type_Mac Desktop',

       'first_device_type_Other/Unknown',

       'first_device_type_SmartPhone (Other)',

       'first_device_type_Windows Desktop', 'first_device_type_iPad',

       'first_device_type_iPhone', 'first_browser_updated_-unknown-',

       'first_browser_updated_Chrome', 'first_browser_updated_Firefox',

       'first_browser_updated_IE', 'first_browser_updated_Mobile Safari',

       'first_browser_updated_Other', 'first_browser_updated_Safari',

       'date_account_created_year_updated', 'date_account_created_day_updated',

       'corrected_age_updated', 'country_destination_updated']]
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier
t_train , t_test , s_train , s_test = train_test_split(train_user_combine_final.drop('country_destination_updated',axis=1),train_user_combine_final['country_destination_updated'] , test_size = 0.20 , random_state=101 )  
rfc = RandomForestClassifier()

rfc.fit(t_train , s_train)
pred = rfc.predict(t_test)
print(accuracy_score(s_test, pred))
dtc = DecisionTreeClassifier()

dtc.fit(t_train , s_train)

pred = dtc.predict(t_test)

print(accuracy_score(s_test, pred))
pred
xgb = XGBClassifier(max_depth=10, learning_rate=0.03, n_estimators=22,

                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=40)

xgb.fit(t_train , s_train)

pred = xgb.predict(t_test) 

print(accuracy_score(s_test, pred))




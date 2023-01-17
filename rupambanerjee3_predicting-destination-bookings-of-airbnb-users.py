import pandas as pd
import numpy as np
from pandas import DataFrame, Series

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
df_train=pd.read_csv("Datasets/train_users_2.csv")
df_test=pd.read_csv("Datasets/test_users.csv")
print(str('There are total '+ str(df_train.shape[0])+' rows in training dataset.'))
print(str('There are total '+ str(df_test.shape[0])+' rows in test dataset.'))
df_train.isnull().sum()
df_train.isnull().sum()
df_test.isnull().sum()
df_test.isnull().sum()
df_train.affiliate_channel.unique()
df_train.affiliate_provider.unique()
df_train.age.unique()
df_train.country_destination.unique()
df_train.date_account_created.unique()
df_train.date_first_booking.unique()
df_train.first_affiliate_tracked.unique()
df_train.first_browser.unique()
from IPython.display import Image
Image(filename="img/Browser.jpg", width=550, height=350)
df_train.first_device_type.unique()
df_train.gender.unique()
Image(filename="img/Gender.jpg", width=450, height=300)
df_train.id.unique()
df_train.language.unique()
df_train.signup_app.unique()
df_train.signup_flow.unique()
df_train.signup_method.unique()
df_train.timestamp_first_active.unique()
df_train.first_browser.replace('-unknown-', np.nan, inplace=True)
df_train.gender.replace('-unknown-', np.nan, inplace=True)
df_train.head()
df_train.age.describe()
print('Summary statistics for age<15:')
print(' ')
print(df_train[df_train.age<15].describe())
print('-------------------------------------------------------')
print('Summary statistics for age>150:')
print(' ')
print(df_train[df_train.age>150].describe())
df_test.loc[df_test.age>18, 'age']
df_abnormal_age=df_train['age']>150
df_train.loc[df_abnormal_age,'age']=2015 - df_train.loc[df_abnormal_age,'age']
df_train.age.describe()
df_train.loc[df_train.age<18,'age']=np.nan
df_train.loc[df_train.age>100,'age']=np.nan
df_train.age.describe()
df_without_ndf=df_train[df_train.country_destination !='NDF']
df_without_ndf.head()
print('Datatype of date_account_type: '+ str(df_without_ndf.date_account_created.dtype))
print('')
print('Datatype of date_first_booking: '+ str(df_without_ndf.date_first_booking.dtype))
print('')
print('Datatype of timestamp_first_active: '+ str(df_without_ndf.timestamp_first_active.dtype))
#To convert the dates into datetime format
df_without_ndf.date_account_created=pd.to_datetime(df_without_ndf.date_account_created)
df_without_ndf.date_first_booking=pd.to_datetime(df_without_ndf.date_first_booking)
df_without_ndf['timestamp_first_active'] = pd.to_datetime((df_without_ndf.timestamp_first_active)//1000000, format='%Y%m%d')
print('Datatype of date_account_type: '+ str(df_without_ndf.date_account_created.dtype))
print('')
print('Datatype of date_first_booking: '+ str(df_without_ndf.date_first_booking.dtype))
print('')
print('Datatype of timestamp_first_active: '+ str(df_without_ndf.timestamp_first_active.dtype))
#visualizing the distribution of user's selection of country
plt.figure(figsize=(12,6))
destination_percentage=df_without_ndf.country_destination.value_counts()/df_without_ndf.shape[0]*100
destination_percentage.plot(kind='bar')
# sns.countplot(x='country_destination', data=df_without_ndf,order=df_without_ndf.country_destination.value_counts().index)
plt.ylabel("% of users")
plt.title("Distribution of destination countries among users")
plt.xticks(rotation='horizontal')
plt.show()
#Let's visualize the ages of users
plt.figure(figsize=(12,6))
sns.distplot(df_without_ndf.age.dropna())
plt.title("Age Distribution of users")
plt.ylabel('% of users')
plt.show()
#Let's check how age is distributed across the destination countries
plt.figure(figsize=(12,6))
sns.boxplot(y='age' , x='country_destination',data=df_without_ndf)
plt.title("Age Distribution across the destinations")
plt.xlabel("")
plt.show()
#gender distribution
plt.figure(figsize=(10,6))
gender_percentage=df_without_ndf.gender.value_counts()/df_without_ndf.shape[0]*100
gender_percentage.plot(kind='bar')
# sns.countplot(x='gender',data=df_without_ndf)
plt.ylabel("% of users")
plt.title("Gender Distribution of users")
plt.xticks(rotation='horizontal')
plt.show()
# fig,axes=plt.subplots(nrows=1,ncols=2, figsize=(12,4))

plt.figure(figsize=(12,6))
sns.countplot(x='country_destination',data=df_without_ndf, hue='gender', 
              order=df_without_ndf.country_destination.value_counts().index)
plt.xlabel('')
plt.ylabel('No. of users')

# plt.figure(figsize=(12,6))
ctab=pd.crosstab(df_without_ndf.country_destination,df_without_ndf['gender']).apply(lambda x: x/x.sum()*100, axis=1)
ctab.plot(kind='bar',stacked=True,legend=True)
plt.ylabel('% of users')
plt.xlabel('')
plt.xticks(rotation='horizontal')

plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

plt.show()
#User signup app distribution
plt.figure(figsize=(12,6))
signup_app_percentage=df_without_ndf.signup_app.value_counts()/df_without_ndf.shape[0]*100
signup_app_percentage.plot(kind='bar')
# sns.countplot(x='signup_app', data=df_without_ndf, order=df.signup_app.value_counts().index)
plt.title("Signup app distribution of users")
plt.ylabel('% of users')
plt.xticks(rotation='horizontal')
plt.show()
df_without_ndf_and_US=df_without_ndf[df_without_ndf.country_destination!='US']
# fig, axes=plt.subplots(nrows=1,ncols=2,figsize=(15,5))
plt.figure(figsize=(12,6))
sns.countplot(x='country_destination',data=df_without_ndf,hue='signup_app',
              order=df_without_ndf.country_destination.value_counts().index,
              hue_order=['Web', 'iOS', 'Moweb', 'Android'])
plt.title("Distribution of Signup app across destination countries")
plt.ylabel('No. of users')
plt.xlabel('')

plt.figure(figsize=(12,6))
sns.countplot(x='country_destination',data=df_without_ndf_and_US,hue='signup_app',
              order=df_without_ndf_and_US.country_destination.value_counts().index,
              hue_order=['Web', 'iOS', 'Moweb', 'Android'])
plt.title("Distribution of Signup app across destination countries excluding US")
plt.ylabel('No. of users')
plt.xlabel('')

plt.show()
plt.figure(figsize=(12,6))
affiliate_channel_percentage=df_without_ndf.affiliate_channel.value_counts()/df_without_ndf.shape[0]*100
affiliate_channel_percentage.plot(kind='bar')
plt.title('Distribution of Affiliate channels used to attract the users')
plt.ylabel('% of users')
plt.xticks(rotation='horizontal')
# sns.countplot(x='affiliate_channel',data=df_without_ndf,order=df_without_ndf.affiliate_channel.value_counts().index)
plt.show()
df_without_ndf.affiliate_channel.unique()
#Channel Distribution based on Destination countries
# fig, axes=plt.subplots(nrows=1, ncols=2,figsize=(15,5))
plt.figure(figsize=(12,6))
sns.countplot(x='country_destination',data=df_without_ndf,hue='affiliate_channel',
              order=df_without_ndf.country_destination.value_counts().index,
              hue_order=['direct', 'sem-brand', 'sem-non-brand', 'seo', 'other', 'api', 'content', 'remarketing'])
plt.title('Distribution of Affiliate channels among the destination countries')
plt.ylabel('No. of users')

plt.figure(figsize=(12,6))
sns.countplot(x='country_destination',data=df_without_ndf_and_US,hue='affiliate_channel',
              order=df_without_ndf_and_US.country_destination.value_counts().index,
              hue_order=['direct', 'sem-brand', 'sem-non-brand', 'seo', 'other', 'api', 'content', 'remarketing'])
plt.title('Distribution of Affiliate channels among the destination countries excluding US')
plt.xlabel('country_destination without US')
plt.ylabel('No. of users')

plt.show()
#Now lets check which apps are being used to signup
plt.figure(figsize=(12,6))
signup_method_percentage=df_without_ndf.signup_method.value_counts()/df_without_ndf.shape[0]*100
signup_method_percentage.plot(kind='bar')
plt.title("Signup Method distribution among users")
plt.ylabel('% of users')
plt.xticks(rotation='horizontal')
# sns.countplot(x='signup_method',data=df_without_ndf, order=df_without_ndf.signup_method.value_counts().index)
plt.show()
#Destination country based on signup app
plt.figure(figsize=(12,6))
sns.countplot(x='country_destination',data=df_without_ndf, order=df_without_ndf.country_destination.value_counts().index,
             hue='signup_method', hue_order=['basic','facebook','google'])
plt.ylabel("No. of users")
plt.title("Distribution of Signup Methods among the destination countries")
plt.legend(loc='upper right')

plt.show()
#First Device type distribution
plt.figure(figsize=(18,6))
first_device_type_percentage=df_without_ndf.first_device_type.value_counts()/df_without_ndf.shape[0]*100
first_device_type_percentage.plot(kind='bar')
# sns.countplot(x='first_device_type',data=df_without_ndf,order=df_without_ndf.first_device_type.value_counts().index)
plt.ylabel("% of users")
plt.title("First Device type distribution among users")
plt.xticks(rotation='horizontal')
plt.show()
#First Device type distribition across destinations
plt.figure(figsize=(18,6))
sns.countplot(x='country_destination',data=df_without_ndf, order=df_without_ndf.country_destination.value_counts().index,
             hue='first_device_type',hue_order=['Mac Desktop', 'Windows Desktop', 'iPhone', 'iPad', 'Other/Unknown', 
                                                'Android Phone','Desktop (Other)', 'Android Tablet', 'SmartPhone (Other)'])
plt.ylabel("No. of users")
plt.title('Distribution of First Device type across destination countries')
plt.legend(loc='upper right')

plt.figure(figsize=(18,6))
sns.countplot(x='country_destination',data=df_without_ndf_and_US, 
              order=df_without_ndf_and_US.country_destination.value_counts().index,
              hue='first_device_type', hue_order=['Mac Desktop', 'Windows Desktop', 'iPhone', 'iPad', 'Other/Unknown', 
                                                'Android Phone','Desktop (Other)', 'Android Tablet', 'SmartPhone (Other)'])
plt.ylabel("No. of users")
plt.title('Distribution of First Device type across destination countries excluding US')
plt.legend(loc='upper right')
plt.show()
#First Browser distribution
plt.figure(figsize=(18,6))
first_browser_percentage=df_without_ndf.first_browser.value_counts()/df_without_ndf.shape[0]*100
first_browser_percentage.plot(kind='bar')
# sns.countplot(x='first_browser', data=df_without_ndf, order=df_without_ndf.first_browser.value_counts().index)
plt.ylabel("% of users")
plt.show()
plt.figure(figsize=(12,6))
signup_flow_percentage=df_without_ndf.signup_flow.value_counts()/df_without_ndf.shape[0]*100
signup_flow_percentage.plot(kind='bar')
plt.title('Pages accessed before landing on Airbnb page')
plt.ylabel('% of users')
plt.xlabel('Page no.')
# sns.countplot(x='signup_flow',data=df_without_ndf)
plt.show()
#New account created over time
plt.figure(figsize=(12,6))
(df_without_ndf.date_account_created.value_counts().plot(kind='line',linewidth=1))
plt.ylabel("No. of Customers")
plt.xlabel("Date")
plt.show()
#Assign the df_without_ndf as df_train_without_NDF
df_train1=df_train[df_train['country_destination']!='NDF']
df_train1.head()
#Replacing the unknowns
df_test.first_browser.replace('-unknown-', np.nan, inplace=True)
df_test.gender.replace('-unknown-', np.nan, inplace=True)
df_test.language.replace('-unknown-', np.nan, inplace=True)
#Fixing the age issues
df_abnormal_age_test=df_test['age']>150
df_test.loc[df_abnormal_age_test,'age']=2015 - df_test.loc[df_abnormal_age_test,'age']
df_test.loc[df_test.age<18,'age']=np.nan
df_test.loc[df_test.age>100,'age']=np.nan
print(df_test.age.describe())
#For date_account_created
df_train1['date_account_created']=pd.to_datetime(df_train1['date_account_created'])
df_train1['date_account_created_year']=df_train1.date_account_created.dt.year
df_train1['date_account_created_month']=df_train1.date_account_created.dt.month
df_train1['date_account_created_day']=df_train1.date_account_created.dt.day

df_test['date_account_created']=pd.to_datetime(df_test['date_account_created'])
df_test['date_account_created_year']=df_test.date_account_created.dt.year
df_test['date_account_created_month']=df_test.date_account_created.dt.month
df_test['date_account_created_day']=df_test.date_account_created.dt.day
#For timestamp_first_active
df_train1['timestamp_first_active']=pd.to_datetime((df_train1.timestamp_first_active//1000000),format='%Y%m%d')
df_train1['timestamp_first_active_year']=df_train1.timestamp_first_active.dt.year
df_train1['timestamp_first_active_month']=df_train1.timestamp_first_active.dt.month
df_train1['timestamp_first_active_day']=df_train1.timestamp_first_active.dt.day

df_test['timestamp_first_active']=pd.to_datetime((df_test.timestamp_first_active//1000000),format='%Y%m%d')
df_test['timestamp_first_active_year']=df_test.timestamp_first_active.dt.year
df_test['timestamp_first_active_month']=df_test.timestamp_first_active.dt.month
df_test['timestamp_first_active_day']=df_test.timestamp_first_active.dt.day
#Drop the main columns
df_train1=df_train1.drop(['date_account_created','timestamp_first_active'], axis=1)
df_test=df_test.drop(['date_account_created','timestamp_first_active'], axis=1)
df_train1.head()
#Replace the NULL values in age 
df_train1['age'].fillna(-1, inplace=True)
df_test['age'].fillna(-1,inplace=True)
#Create the target variable and drop it from train dataset
y_train=df_train1['country_destination']
x_train=df_train1.drop(['country_destination'], axis=1)
x_test=df_test
#Drop the unwanted columns from both the datasets
id_test=x_test['id']
x_train=x_train.drop(['date_first_booking'], axis=1)
x_test=df_test.drop(['date_first_booking'], axis=1)
#Check the total rows and columns in both rain and test datasets
print("Train Dataset: "+str(x_train.shape))
print("Test Dataset: "+str(x_test.shape))
#Overview of train dataset
x_train.head()
#Overview of test dataset
x_test.head()
#Merge x_train and y_train dataset
merge_train_test=pd.concat([x_train,x_test],axis=0)
#Use get_dummies function to convert the categorical variables into one hot encoding
categorical_columns=['gender', 'signup_method', 'signup_flow', 'language',
       'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
       'signup_app', 'first_device_type', 'first_browser',
       'date_account_created_year', 'date_account_created_month',
       'date_account_created_day', 'timestamp_first_active_year',
       'timestamp_first_active_month', 'timestamp_first_active_day']
merge_train_test1=pd.get_dummies(merge_train_test,columns=categorical_columns)
merge_train_test2=merge_train_test1.set_index('id')
x_train2=merge_train_test2.loc[x_train['id']]
x_train2.shape
x_test2=merge_train_test2.loc[x_test['id']]
x_test2.shape
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
encoded_y_train=label_encoder.fit_transform(y_train)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
x_train2.dtypes
import xgboost as xgb
xg_train = xgb.DMatrix(x_train2, label=encoded_y_train)
#Specifying the hyperparameters
params = {'max_depth': 10,
    'learning_rate': 1,
    'n_estimators': 5,
    'objective': 'multi:softprob',
    'num_class': 12,
    'gamma': 0,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'base_score': 0.5,
    'missing': None,
    'nthread': 4,
    'seed': 42
          }
num_boost_round = 5
print("Train a XGBoost model")
gbm = xgb.train(params, xg_train, num_boost_round)
y_pred=gbm.predict(xgb.DMatrix(x_test2))
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += label_encoder.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
y_pred[0]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
forest_class = RandomForestClassifier(random_state = 42)

n_estimators = [100, 500]
min_samples_split = [10, 20]

param_grid_forest = {'n_estimators' : n_estimators, 'min_samples_split' : min_samples_split}


rand_search_forest = GridSearchCV(forest_class, param_grid_forest, cv = 4, refit = True,
                                 n_jobs = -1, verbose=2)

rand_search_forest.fit(x_train2, encoded_y_train)
random_estimator = rand_search_forest.best_estimator_

y_pred_random_estimator = random_estimator.predict_proba(final_train_X)
y_pred = random_estimator.predict_proba(final_test_X) 

# We take the 5 highest probabilities for each person
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generating a csv file with the predictions 
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('output_randomForest.csv',index=False)
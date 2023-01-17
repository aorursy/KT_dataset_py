#Importing all dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


#algorithms
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#Loading datasets
data_dir = "../input/"
train_users = pd.read_csv(data_dir + 'train_users_2.csv',parse_dates=['date_account_created','timestamp_first_active','date_first_booking'], index_col=False)
sessions=pd.read_csv(data_dir + 'sessions.csv')
test_users=pd.read_csv(data_dir + 'test_users.csv', parse_dates=['date_account_created','timestamp_first_active','date_first_booking'], index_col=False)
#Printing the columns
print(train_users.columns)
print(test_users.columns)
#Mark test and train users and merge them for data preparation
train_users['type']='Train'
test_users['country_destination']='NULL'
test_users['type']='Test'
users = pd.concat([train_users, test_users], ignore_index=True)
#verify the counts
print('# of train users: ',train_users.id.count())
print('# of test users: ',test_users.id.count())
print('# of total users: ',train_users.id.count()+test_users.id.count())
print('# of users: ',users.id.count())
def feature_engineering(data):

    #Date account created
    data['Day_Acct_Created'] = data['date_account_created'].dt.day
    data['Month_Acct_Created'] = data['date_account_created'].dt.month
    data['Year_Acct_Created'] = data['date_account_created'].dt.year
    data['Hour_Acct_Created'] = data['date_account_created'].dt.hour    
    data['DayOfWeek_Acct_Created'] = data['date_account_created'].dt.dayofweek
    data['WeekOfYear_Acct_Created'] = data['date_account_created'].dt.weekofyear
    
    #Timestamp of first active
    data['Day_First_Active'] = data['timestamp_first_active'].dt.day
    data['Month_First_Active'] = data['timestamp_first_active'].dt.month
    data['Year_First_Active'] = data['timestamp_first_active'].dt.year
    data['Hour_First_Active'] = data['timestamp_first_active'].dt.hour    
    data['DayOfWeek_First_Active'] = data['timestamp_first_active'].dt.dayofweek
    data['WeekOfYear_First_Active'] = data['timestamp_first_active'].dt.weekofyear
    
    #Date of first booking
    data['Day_First_Booking'] = data['date_first_booking'].dt.day
    data['Month_First_Booking'] = data['date_first_booking'].dt.month
    data['Year_First_Booking'] = data['date_first_booking'].dt.year
    data['Hour_First_Booking'] = data['date_first_booking'].dt.hour    
    data['DayOfWeek_First_Booking'] = data['date_first_booking'].dt.dayofweek
    data['WeekOfYear_First_Booking'] = data['date_first_booking'].dt.weekofyear
    
    #Replace unknowns by NA
    data.gender.replace('-unknown-', np.nan, inplace=True)
           
    #Replace Ages
    data.loc[data.age > 95, 'age'] = np.nan
    data.loc[data.age < 13, 'age'] = np.nan
        
    #Converting categorical to numeric    
    enc = LabelEncoder()
    #data['gender_cd'] = enc.fit_transform(data['gender'])
    data['signup_method_cd'] = enc.fit_transform(data['signup_method'])
    data['language_cd'] = enc.fit_transform(data['language'])
    data['affiliate_channel_cd'] = enc.fit_transform(data['affiliate_channel'])
    data['affiliate_provider_cd'] = enc.fit_transform(data['affiliate_provider'])
    #data['first_affiliate_tracked_cd'] = enc.fit_transform(data['first_affiliate_tracked'])
    data['signup_app_cd'] = enc.fit_transform(data['signup_app'])
    data['first_device_type_cd'] = enc.fit_transform(data['first_device_type'])
    data['first_browser_cd'] = enc.fit_transform(data['first_browser'])

    #Converting the target variable as it is in category
    category_encoder = LabelEncoder()
    category_encoder.fit(data['country_destination'])
    data['country_destination_cd'] = category_encoder.transform(data['country_destination'])
    #print(category_encoder.classes_)
    
    
    return data
temp=feature_engineering(users)
#Manual feature engineering

#gender
#Converting categorial to numeric
temp.gender[temp.gender=='nan']='-1'
temp.gender[temp.gender=='MALE']='0'
temp.gender[temp.gender=='FEMALE']='1'
temp.gender[temp.gender=='OTHER']='2'

#first_affiliate_tracked
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='nan']='-1'
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='untracked']='0'
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='omg']='1'
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='linked']='2'
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='tracked-other']='3'
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='product']='4'
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='marketing']='5'
temp.first_affiliate_tracked[temp.first_affiliate_tracked=='local ops']='6'

temp = temp.fillna(-1)
#Split train and test sets
train=temp[temp['type']=='Train']
test=temp[temp['type']=='Test']
print(train.id.count(),test.id.count())
#Creating train_xx and target_xx
train_xx=train[[ 'Day_Acct_Created',
       'Month_Acct_Created', 'Year_Acct_Created', 'Hour_Acct_Created',
       'DayOfWeek_Acct_Created', 'WeekOfYear_Acct_Created', 'Day_First_Active',
       'Month_First_Active', 'Year_First_Active', 'Hour_First_Active',
       'DayOfWeek_First_Active', 'WeekOfYear_First_Active',
       'Day_First_Booking', 'Month_First_Booking', 'Year_First_Booking',
       'Hour_First_Booking', 'DayOfWeek_First_Booking',
       'WeekOfYear_First_Booking', 'signup_method_cd', 'language_cd',
       'affiliate_channel_cd', 'affiliate_provider_cd', 'signup_app_cd',
       'first_device_type_cd', 'first_browser_cd','gender','age']]
target_xx=train['country_destination_cd']
predict_xx=test[[ 'Day_Acct_Created',
       'Month_Acct_Created', 'Year_Acct_Created', 'Hour_Acct_Created',
       'DayOfWeek_Acct_Created', 'WeekOfYear_Acct_Created', 'Day_First_Active',
       'Month_First_Active', 'Year_First_Active', 'Hour_First_Active',
       'DayOfWeek_First_Active', 'WeekOfYear_First_Active',
       'Day_First_Booking', 'Month_First_Booking', 'Year_First_Booking',
       'Hour_First_Booking', 'DayOfWeek_First_Booking',
       'WeekOfYear_First_Booking', 'signup_method_cd', 'language_cd',
       'affiliate_channel_cd', 'affiliate_provider_cd', 'signup_app_cd',
       'first_device_type_cd', 'first_browser_cd','gender','age']]
target_xx.head()
#Splitting train and test
X = train_xx
y = target_xx 
X_test = predict_xx

#Classifier
xgb = RandomForestClassifier()                
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)  

test['x']=y_pred
output=test[['id','x']]

output.head()
test['prediction']=xx

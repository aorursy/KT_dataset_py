# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

%matplotlib inline

sns.set_style("white")

sns.set_context('talk')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read The Files

train = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv')

age = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv')

countries = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/countries.csv')

sessions = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv')

test = pd.read_csv('/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv')
print(train.describe())

print(train.info())
print(test.describe())

print(test.info())
#Join the test & train data to fix them both at the same time



df = train.append(test, ignore_index = True, sort = True)

print(df.info())

# Checking Catrgorical columns for unsusal entries

cat = ['gender','signup_method','language','affiliate_channel','affiliate_provider'

       ,'first_affiliate_tracked','signup_app','first_device_type','first_browser','country_destination']



for x in cat :

    print ("Col Name is :",x,"\nand it's values are :\n",df[x].value_counts())

#Plotting distribution of the data



for x in cat:

    sns.countplot(x=x, data=df,palette='RdBu')

    plt.ylabel('Number of users')

    plt.title('Users '+ x + ' Distribution')

    plt.xticks(rotation='vertical')

    plt.show()

    plt.savefig('plot'+str(x)+'.png')

    
# Too Much Unknown Data In Columns : Gender & First Browser , will need to fix that later

# Now Let's Focus on the Dates Data
# Investigate The Time Users Spend Between Being First Active and Actually Making A reservation



df['timestamp_first_active'] = pd.to_datetime((df.timestamp_first_active // 1000000), format='%Y%m%d')

df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])

df['time_to_booking']= df['date_first_booking'] - df['timestamp_first_active']

print(df.time_to_booking.describe())
# Investigate Month and Year Of Users Bookings And Signing up to see most active years/months



df['month_booking']= df.date_first_booking.dt.month

df['year_booking']= df.date_first_booking.dt.year

df['date_account_created'] = pd.to_datetime(df['date_account_created'])

df['month_create']=df.date_account_created.dt.month

df['year_create']=df.date_account_created.dt.year
for x in ['month_booking','year_booking','month_create','year_create'] :

    sns.countplot(x=x,data=df)

    plt.xticks(rotation='vertical')

    plt.show()

    plt.savefig('plot'+str(x)+'.png')

df.date_account_created.value_counts().plot(kind='line')

plt.xlabel('Date')

plt.title('New Accounts Created Over Time')

plt.xticks(rotation='vertical')

plt.show()

plt.savefig('plot New Accounts Created Over Time.png')

new2 = sessions.groupby('user_id').count()

print(new2.describe())
sessions2 = sessions.groupby('user_id').sum()

df2 = df.merge(sessions2,left_on='id',right_on='user_id',how='inner')



secs =[]

counts =[]

for x in df2.country_destination.unique():

    dfndf = df2[df2.country_destination == x]

    dfndf['hour']=dfndf.secs_elapsed // 3600

    counts.append(dfndf.id.count())

    secs.append(dfndf.hour.mean())

    

sns.set_context('notebook')    

sns.scatterplot(x=counts, y=secs , hue =df2.country_destination.unique())

plt.xlabel('No.Of.Users')

plt.ylabel('Mean Hours Users Spends on the Website')

plt.title('Web Sessions Data of Users')

plt.show()

plt.savefig('plot Web Sessions Data of Users.png')

#Drop Year Column because it's the same for all entries (2015)

age = age.drop('year',axis = 1)
# Group and Plot Age Data 

g = age.groupby(['age_bucket','gender']).sum().reset_index().sort_values('population_in_thousands')

sns.set_context('talk')

sns.barplot(x='age_bucket',y = 'population_in_thousands',data=g)

plt.xticks(rotation='vertical')

plt.title('Different Age Groups')

plt.show()

plt.savefig('plot Different Age Groups.png')



# The Age Data



#set any value bigger than 130 or lower than 18 to be nan

df.age[df.age > 110] = np.nan

df.age[df.age < 18] = np.nan



#Replace Missing age data with the mean 

df.loc[df['age'].isnull(),'age'] = df.age.median()
#look at age distribution

sns.distplot(df.age)

plt.title('Age Distribution Of Users')

plt.show()
#Extract the remaining date information

df['month_active']= df.timestamp_first_active.dt.month

df['year_active']= df.timestamp_first_active.dt.year
df.info()
#Drop unnecessary columns after the extraction of useful data

df1 = df.drop(['date_first_booking','time_to_booking','month_booking','year_booking','date_account_created',

              'timestamp_first_active','timestamp_first_active','country_destination','id'],axis=1)
# Handle categorical Columns

ndf = pd.get_dummies(df1,columns=['affiliate_channel','affiliate_provider','first_affiliate_tracked',

                                'first_browser','first_device_type','language','signup_app','year_active'

                                ,'signup_flow','signup_method','month_create','year_create','month_active'],

                     drop_first =True,dtype='float16')
ndf.head()
# Splitting the data sets to impute gender data using KNN

xtrn1 = ndf.loc[ndf['gender'] != '-unknown-'].drop('gender',axis=1)

ytrn1 = ndf.gender[ndf['gender'] != '-unknown-']
xtst1 = ndf.loc[ndf['gender'] == '-unknown-'].drop('gender',axis=1)

ytst1 = ndf.gender[ndf['gender'] == '-unknown-']
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(xtrn1,ytrn1)
yprd = neigh.predict(xtst1)
yprd1 = pd.DataFrame(yprd)

yprd1.index = ytst1.index

xtfinal = pd.concat([yprd1,xtst1],axis=1)
xtfinal.rename(columns={0:'gender'},inplace = True )
xtrain_final = pd.concat([ytrn1,xtrn1],axis=1)
xfinal = xtrain_final.append(xtfinal)

xfinal = pd.concat([xfinal,df.country_destination],axis=1)
xfinal.head()
xy = pd.get_dummies(xfinal,columns=['gender'],drop_first =True,dtype='float16')

x_dum = xy[xy.columns.difference(['country_destination'])]

y_dum = xy['country_destination'].reset_index()

#Building the Classfication Model
x_train = x_dum.iloc[0:213451,:].values

y_train=y_dum.iloc[0:213451,:].drop('index',axis=1).values

x_test = x_dum.iloc[213451:,:].values
from sklearn.preprocessing import StandardScaler

from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import BernoulliNB
class_weight_list = compute_class_weight('balanced',

                                                 np.unique(np.ravel(y_train,order='C')),

                                                 np.ravel(y_train,order='C'))

class_weight = dict(zip(np.unique(y_train), class_weight_list))



print(class_weight)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from xgboost.sklearn import XGBClassifier



clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,

                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
# Using Submession System To Evaluate The Model
submission = pd.DataFrame({'id':test['id'],'country':y_pred})

submission.head()



filename = 'Airbnb Predictions 1_1.csv'

submission.to_csv(filename,index=False)
df = train.append(test, ignore_index = True, sort = True)

df.age[df.age > 110] = np.nan

df.age[df.age < 18] = np.nan



#Replace Missing age data with the mean 

df.loc[df['age'].isnull(),'age'] = -1
# Extracting Age Data As before



df['timestamp_first_active'] = pd.to_datetime((df.timestamp_first_active // 1000000), format='%Y%m%d')

df['day_active'] = df.timestamp_first_active.dt.day

df['month_active']= df.timestamp_first_active.dt.month

df['year_active']= df.timestamp_first_active.dt.year



df['date_account_created'] = pd.to_datetime(df['date_account_created'])

df['day_create'] = df.date_account_created.dt.day

df['month_create']=df.date_account_created.dt.month

df['year_create']=df.date_account_created.dt.year
df.head()
ndf2 = df.drop(['date_first_booking','date_account_created',

              'timestamp_first_active','timestamp_first_active','id'],axis = 1 )
xy2 = pd.get_dummies(ndf2,columns=['affiliate_channel','affiliate_provider','first_affiliate_tracked',

                                'first_browser','first_device_type','gender','language','signup_app','signup_flow'

                                ,'signup_method'])
x_dum2 = xy2[xy2.columns.difference(['country_destination'])]

y_dum2 = xy2['country_destination'].reset_index()
x_train = x_dum2.iloc[0:213451,:].values

y_train=y_dum2.iloc[0:213451,:].drop('index',axis=1).values

x_test = x_dum2.iloc[213451:,:].values
from xgboost.sklearn import XGBClassifier



clf = XGBClassifier(max_depth=6, learning_rate=0.2, n_estimators=50,class_weight=class_weight ,

                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
submission = pd.DataFrame({'id':test['id'],'country':y_pred})

submission.head()



filename = 'Airbnb Predictions 1_2.csv'

submission.to_csv(filename,index=False)
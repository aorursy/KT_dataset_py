# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Unzipping and reading from data files



import zipfile



zf = zipfile.ZipFile('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip') 

train_users = pd.read_csv(zf.open('train_users_2.csv'))

zf = zipfile.ZipFile('/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip') 

test_users = pd.read_csv(zf.open('test_users.csv'))

test_users.head()

print("We have", train_users.shape[0], "users in the training set and", 

      test_users.shape[0], "in the test set.")

print("In total we have", train_users.shape[0] + test_users.shape[0], "users.")

# Merge train and test users

users = pd.concat((train_users, test_users), axis=0, ignore_index=True, sort=True)





users.head()
users.describe()


train_users.loc[train_users['age'] == 2014]

web_2014 = users.loc[users['age'] == 2014, 'signup_app'].value_counts() 

print (web_2014)
print(users.signup_app.value_counts())
np.unique(users[users.age > 122]['age'])
print(sum(users.age > 122))

print(sum(users.age < 18))
users.loc[users.age > 95, 'age'].count()
users.loc[users.age < 13, 'age'].count()
users.loc[users.age > 95, 'age'] = np.nan

users.loc[users.age < 13, 'age'] = np.nan
users.age.value_counts()
users.corr()
import seaborn as sns

sns.pairplot(train_users)
users["gender"]=users["gender"].fillna("-unknown-" )

users.head()
#How much data is missing from the dataset (apart from destination country)

users_nan = (users.isnull().sum() / users.shape[0]) * 100

users_nan[users_nan > 0].drop('country_destination')
train_users.date_first_booking.isnull().sum() / train_users.shape[0] * 100
train_users.country_destination.isnull().sum() / train_users.shape[0] * 100
users.gender.value_counts(dropna=False)
users.gender.describe()
users.age.describe()
#For now, let's fill the missing values of age with the median since the mean is highly affectd by extreme values

users["age"]=users["age"].fillna( users["age"].median())



users.head()

categorical_features = [

    'affiliate_channel',

    'affiliate_provider',

    'country_destination',

    'first_affiliate_tracked',

    'first_browser',

    'first_device_type',

    'gender',

    'language',

    'signup_app',

    'signup_method'

]



for categorical_feature in categorical_features:

    print (categorical_feature,users[categorical_feature].unique())



users['date_account_created'] = pd.to_datetime(users['date_account_created'])

users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])

users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')
users.head()
users.gender.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)

plt.xlabel('Gender')

sns.despine()
women = sum(users['gender'] == 'FEMALE')

men = sum(users['gender'] == 'MALE')



female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100

male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100



# Bar width

width = 0.4



male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)

female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)



plt.legend()

plt.xlabel('Destination Country')

plt.ylabel('Percentage')



sns.despine()

plt.show()
destination_percentage = users.country_destination.value_counts() / users.shape[0] * 100

destination_percentage.plot(kind='bar',color='#FD5C64', rot=0)

# Using seaborn can also be plotted

# sns.countplot(x="country_destination", data=users, order=list(users.country_destination.value_counts().keys()))

plt.xlabel('Destination Country')

plt.ylabel('Percentage')

sns.despine()
age = 45



younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())

older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())



younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100

older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100



younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Youngers', rot=0)

older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Olders', rot=0)



plt.legend()

plt.xlabel('Destination Country')

plt.ylabel('Percentage')



sns.despine()

plt.show()
print((sum(users.language == 'en') / users.shape[0])*100)



En = sum(users.loc[users['language']=="en", 'country_destination'].value_counts());

No_En=sum(users.loc[users['language']!="en", 'country_destination'].value_counts());

En_destinations = users.loc[users['language']=="en" , 'country_destination'].value_counts() / En * 100

No_En_destinations = users.loc[users['language'] !="en", 'country_destination'].value_counts() / No_En * 100



younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='English', rot=0)

older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Non English', rot=0)



plt.legend()

plt.xlabel('Destination Country')

plt.ylabel('Percentage')



sns.despine()

plt.show()
import math

users["age"].isna().any()
users["gender"].isnull().values.any()

users["age"].isnull().values.any()

train_users = users.iloc[:213451 , :]

train_users
test_users = users.iloc[213451: , :]

test_users.drop(['country_destination'], axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import validation_curve



#Convert categorical variable into dummy/indicator variables.



y = train_users["country_destination"]

features = ["gender","age"]

X = pd.get_dummies(train_users[features])

X_test = pd.get_dummies(test_users[features])

X.head()
from sklearn.model_selection import cross_val_score



model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)

model_random_cv=cross_val_score(model, X, y, cv=5) 

print (model_random_cv.mean())


model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_users.id, 'country': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
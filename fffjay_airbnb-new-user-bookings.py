import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing, feature_extraction

%matplotlib inline

plt.style.use('seaborn')
test_users = pd.read_csv("../input/test_users.csv")
train_users = pd.read_csv("../input/train_users_2.csv")
print(train_users.shape)
print(test_users.shape)
train_users.head()
test_users.head()
all_users = pd.concat((train_users, test_users), axis = 0, ignore_index = True)
all_users.head()
all_users.shape
print(train_users.country_destination.value_counts())
import seaborn as sns
sns.set_style()
des_countries = train_users.country_destination.value_counts(dropna = False) / train_users.shape[0] * 100
des_countries.plot('bar', rot = 0)
plt.xlabel('Destination country')
plt.ylabel('Percentage of booking')
all_users.info()
test_users.info()
all_users.age.describe()
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))

axes[0].set_title('Age < 200')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
all_users[all_users.age < 200].age.hist(bins = 10, ax = axes[0])

axes[1].set_title('Age >= 200')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')
all_users[all_users.age >= 200].age.hist(bins = 10, ax = axes[1])
all_users.gender.value_counts(dropna = False)
female = sum(train_users.gender == 'FEMALE')
male = sum(train_users.gender == 'MALE')
other = sum(train_users.gender == 'OTHER')

female_destinations = train_users.loc[train_users.gender == 'FEMALE', 'country_destination'].value_counts() / female * 100
male_destinations = train_users.loc[train_users.gender == 'MALE', 'country_destination'].value_counts() / male * 100
other_destinations = train_users.loc[train_users.gender == 'OTHER', 'country_destination'].value_counts() / other * 100

female_destinations.plot('bar', width = 0.25, color = '#0b61a4', position = 0, label = 'Female', rot = 0)
male_destinations.plot('bar', width = 0.25, color = '#3f92d2', position = 1, label = 'Male', rot = 0)
other_destinations.plot('bar', width = 0.25, color = '#66a3d2', position = 2, label = 'Other', rot = 0)

plt.legend()
plt.xlabel('Destination country')
plt.ylabel('Percentage of booking')
plt.show()
fig = plt.figure(figsize = (12, 6))
all_users.date_account_created = pd.to_datetime(all_users.date_account_created)
all_users.date_account_created.value_counts().plot('line')
plt.xlabel('Year')
plt.ylabel('Count created')
fig = plt.figure(figsize = (12, 6))
all_users['date_first_active'] = pd.to_datetime(all_users.timestamp_first_active // 1000000, format = '%Y%m%d')
all_users.date_first_active.value_counts().plot('line')
plt.xlabel('Year')
plt.ylabel('Count first active')
all_users.signup_method.value_counts()
all_users.signup_app.value_counts()
all_users.first_device_type.value_counts()
all_users.first_browser.value_counts()
all_users.language.value_counts()
lang = all_users.language.value_counts() / all_users.shape[0] * 100
plt.figure(figsize = (12, 10))
plt.xlabel('User language')
plt.ylabel('Percentage')
lang.plot('bar', fontsize = 17, rot = 0)
all_users.affiliate_channel.value_counts()
all_users.affiliate_provider.value_counts()
sessions = pd.read_csv('../input/sessions.csv')
sessions.head()
sessions.shape
sessions.info()
len(sessions.user_id.unique())
df_sess = sessions.groupby(['user_id']).user_id.count().reset_index(name = 'session_count')
df_sess.head()
df_sess.session_count.describe()
secs = sessions.groupby(['user_id']).secs_elapsed.sum().reset_index()
secs.columns = ['user_id', 'secs_elapsed']
secs.describe()
sns.boxplot(x = secs.secs_elapsed)
sessions.action_type.value_counts()
at = sessions.action_type.value_counts(dropna = False) / sessions.shape[0] * 100
plt.figure(figsize = (12, 8))
plt.xlabel('Action type')
plt.ylabel('Percentage')
at.plot('bar', fontsize = 17)
sessions.action.value_counts()
sessions.action_detail.value_counts()
train_users_labels = train_users.loc[:, 'country_destination']
print(train_users_labels.head())
train_users_attrs = train_users.iloc[:, 0:15]
train_users_attrs.head()
train_users = train_users.drop(['date_first_booking'], axis = 1)
test_users = test_users.drop(['date_first_booking'], axis = 1)
date_acc_created_train = np.vstack(train_users.date_account_created.astype(str).apply(
    lambda x : list(map(int, x.split('-')))).values)

train_users['create_year'] = date_acc_created_train[:, 0]
train_users['create_month'] = date_acc_created_train[:, 1]
train_users['create_day'] = date_acc_created_train[:, 2]
train_users = train_users.drop(['date_account_created'], axis = 1)

date_acc_created_test = np.vstack(test_users.date_account_created.astype(str).apply(
    lambda x : list(map(int, x.split('-')))).values)

test_users['create_year'] = date_acc_created_test[:, 0]
test_users['create_month'] = date_acc_created_test[:, 1]
test_users['create_day'] = date_acc_created_test[:, 2]
test_users = test_users.drop(['date_account_created'], axis = 1)
train_users.head()
train_users.loc[train_users.gender == '-unknown-', 'gender'] = -1
train_users.loc[train_users.gender.isnull(), 'gender'] = -1
test_users.loc[test_users.gender == '-unknown-', 'gender'] = -1
test_users.loc[test_users.gender.isnull(), 'gender'] = -1
gender_enc = {'FEMALE' : 0,
             'MALE' : 1,
             'OTHER' : 2,
             -1 : -1}
for data in [train_users, test_users]:
    data.gender = data.gender.apply(lambda x : gender_enc[x])
train_users.head()
train_users.age.describe()
train_users.loc[train_users.age > 90, 'age'] = np.nan
train_users.loc[train_users.age < 16, 'age'] = np.nan
test_users.loc[test_users.age > 90, 'age'] = np.nan
test_users.loc[test_users.age < 16, 'age'] = np.nan
print(train_users.age.median())
print(test_users.age.median())
train_users.loc[train_users.age.isnull(), 'age'] = train_users.age.median()
test_users.loc[test_users.age.isnull(), 'age'] = test_users.age.median()
signup_enc = {'facebook' : 0,
             'google' : 1,
             'basic' : 2,
             'weibo' : 3}
for data in [train_users, test_users]:
    data.signup_method = data.signup_method.apply(lambda x : signup_enc[x])
train_users.head()
test_users.loc[test_users.language == '-unknown-', 'language'] = test_users.language.mode()[0]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_users.language = le.fit_transform(train_users.language)
test_users.language = le.fit_transform(test_users.language)
train_users.head()
train_users.affiliate_channel = le.fit_transform(train_users.affiliate_channel)
train_users.affiliate_provider = le.fit_transform(train_users.affiliate_provider)
test_users.affiliate_channel = le.fit_transform(test_users.affiliate_channel)
test_users.affiliate_provider = le.fit_transform(test_users.affiliate_provider)

train_users.loc[train_users.first_affiliate_tracked.isnull(), 'first_affiliate_tracked'] = 'untracked'
train_users.first_affiliate_tracked = le.fit_transform(train_users.first_affiliate_tracked)

test_users.loc[test_users.first_affiliate_tracked.isnull(), 'first_affiliate_tracked'] = 'untracked'
test_users.first_affiliate_tracked = le.fit_transform(test_users.first_affiliate_tracked)
train_users.head()
train_users.signup_app = le.fit_transform(train_users.signup_app)
train_users.first_device_type = le.fit_transform(train_users.first_device_type)
train_users.first_browser = le.fit_transform(train_users.first_browser)
test_users.signup_app = le.fit_transform(test_users.signup_app)
test_users.first_device_type = le.fit_transform(test_users.first_device_type)
test_users.first_browser = le.fit_transform(test_users.first_browser)
train_users.head()
test_users.head()
df = sessions.user_id.value_counts()
print(df.shape)
print(df.head())
df = df.to_frame()
df = df.rename(columns = {'user_id' : 'session_count'})
df['id'] = df.index
df.head()
train_users = pd.merge(train_users, df, how = 'left', on = ['id'])
test_users = pd.merge(test_users, df, how = 'left', on = ['id'])
train_users.session_count.fillna(0, inplace = True)
test_users.session_count.fillna(0, inplace = True)
train_users.session_count = train_users.session_count.astype(int)
test_users.session_count = test_users.session_count.astype(int)
label_df = train_users_labels.to_frame()
for data in [label_df]:
    data.country_destination = le.fit_transform(data.country_destination)

label_df.head()
train_users = train_users.drop(['id', 'timestamp_first_active', 'country_destination'], axis = 1)
train_users.head()
train_users.shape
from sklearn import preprocessing
ss = preprocessing.StandardScaler()
train_users_scaled = pd.DataFrame(ss.fit_transform(train_users))
train_users_scaled.head()
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

[train_data, test_data, train_label, test_label] = train_test_split(train_users_scaled, label_df, test_size = 0.3, random_state = 817)

gnb = GaussianNB()
gnb.fit(train_data, train_label.values.ravel())
print('Accuracy score for Navie Bayes:')
print(gnb.score(test_data, test_label))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(train_data, train_label.values.ravel())
print('Accuracy score for LDA:')
print(lda.score(test_data, test_label))
# from sklearn.ensemble import GradientBoostingClassifier

# gb = GradientBoostingClassifier(max_depth = 4, n_estimators = 100, random_state = 817)
# gb.fit(train_data, train_label.values.ravel())
# print('Accuracy score for GDBT:')
# print(gb.score(test_data, test_label))
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(max_depth = 4, n_estimators = 100, random_state = 817)
gb.fit(train_users_scaled, label_df.values.ravel())
test_users_scaled = pd.DataFrame(ss.fit_transform(test_users.drop(['id', 'timestamp_first_active'], axis = 1)))
prediction_proba = gb.predict_proba(test_users_scaled)
ids_test = test_users['id']

ids = []
countries = []

for i in range(len(ids_test)):
    idx = ids_test[i]
    ids += [idx] * 5
    countries += le.inverse_transform(np.argsort(prediction_proba[2])[::-1][:5]).tolist()
submission = pd.DataFrame({
    "id" : ids,
    "country" : countries
})
submission.to_csv('submission.csv', index = False)
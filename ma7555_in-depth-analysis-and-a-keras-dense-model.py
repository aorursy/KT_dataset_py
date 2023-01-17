# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import ComplementNB

from xgboost.sklearn import XGBClassifier

from wordcloud import WordCloud



import os

print(os.listdir("../input"))
# Load the data

train = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv')

test = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv')

train.tail()
# check for different columns between train/test datasets

print(np.setdiff1d(train.columns, test.columns, assume_unique=True))
# Concatenate both train & test datasets for further cleaning

all_data = pd.concat([train.drop('country_destination', axis=1), test])

all_data.dtypes



# Change countries names' to numerical labels

le = LabelEncoder()

train_labels = le.fit_transform(train.country_destination)

target_labels = pd.Series(train_labels, name='target')

list(le.classes_)
# Check numerical values

all_data.describe()
train.country_destination.value_counts().plot(kind='bar', color=plt.cm.tab20c(np.arange(len(train.country_destination.unique()))))

plt.show()
#Clean the data

all_data.replace(to_replace='-unknown-', value=np.nan, inplace=True)

#all_data.replace(to_replace='Other/Unknown', value=np.nan, inplace=True)

all_data.loc[all_data.age > 110, 'age'] = np.nan

all_data.loc[all_data.age < 18, 'age'] = np.nan

print('Number of rows: {}'.format(len(train)))

print('\nMissing Values as a percentage in training dataset')

print(all_data.iloc[:len(train)].isnull().sum().where(lambda x : x>0).dropna()/len(train))

print('\nMissing Values as a percentage in testing dataset')

print(all_data.iloc[len(train):].isnull().sum().where(lambda x : x>0).dropna()/len(test))
# fill language & first_affiliate_tracked with mode

# drop date_first_booking

all_data.language.fillna(all_data.language.mode()[0], inplace=True)

all_data.first_affiliate_tracked.fillna(all_data.first_affiliate_tracked.mode()[0], inplace=True)

all_data.drop('date_first_booking', axis=1, inplace=True)
# Convert dates to datetime format

all_data.date_account_created = pd.to_datetime(all_data.date_account_created)

all_data.timestamp_first_active = pd.to_datetime(all_data.timestamp_first_active, format='%Y%m%d%H%M%S')

all_data.head(10)
# Copy a new DF with fewer browser options

plt.figure(figsize=(15,5))

plt.subplot(121)

top_browsers = all_data.groupby('first_browser').id.count().nlargest(8).index

df = all_data.dropna(subset=['first_browser']).copy() # Drop Nulls to do the analysis

df.first_browser = df.first_browser.apply(lambda browser: 'Other' if browser not in top_browsers else browser)

sns.countplot(x='first_browser', data=df, hue='signup_app' ,order=df.first_browser.value_counts().index)

plt.xticks(rotation=30)

plt.subplot(122)

sns.countplot(x='first_browser',  data=df, hue='first_device_type' ,order=df.first_browser.value_counts().index)

plt.xticks(rotation=30)

plt.show()
#Mobile_Safari = all_data[all_data.first_device_type == 'iPad'].first_browser.value_counts().nlargest(1).index[0]

all_data.loc[(all_data.first_device_type.isin(['iPad', 'iPhone']) | all_data.signup_app.isin(['Moweb', 'iOS'])) & 

             (all_data.first_browser.isnull()), 'first_browser'] = 'Mobile Safari'

all_data.loc[(all_data.first_device_type == 'Windows Desktops') & (all_data.signup_app == 'Web') & 

             (all_data.first_browser.isnull()), 'first_browser'] = 'Chrome'

all_data.loc[(all_data.first_device_type == 'Mac Desktops') & (all_data.signup_app == 'Web') & 

             (all_data.first_browser.isnull()), 'first_browser'] = 'Safari'

all_data.loc[(all_data.first_device_type == 'Android Phone') & (all_data.first_browser.isnull()), 'first_browser'] = 'Chrome Mobile'

all_data.first_browser.fillna(all_data.first_browser.mode()[0], inplace=True) # If any left, fill with mode.. (3% were left)
print(all_data.isnull().sum().where(lambda x : x>0).dropna()/len(all_data))
all_data.fillna(-1, inplace=True)
fig = plt.figure(figsize=(15,15))

plt.subplot(221)

grouped_df = all_data.groupby([all_data.date_account_created.dt.year, all_data.date_account_created.dt.month]).count().id

grouped_df.plot(kind='line', xticks=range(1, len(grouped_df), 6), rot=45, color='g')

plt.xlabel('Creation Date (Y, M)')

plt.yticks([])

plt.title('Development of user base over years')



ax = fig.add_subplot(222)

all_data.groupby([all_data.timestamp_first_active.dt.year, 

                  all_data.timestamp_first_active.dt.month]).count().id.unstack().plot(

                    kind='line', ax=ax, 

                    xticks=range(all_data.timestamp_first_active.dt.year.min(), 

                                 all_data.timestamp_first_active.dt.year.max()+1),

                                 colormap='Paired')

plt.title('Monthly new users development over years (first activity)')

ax.legend(title='Month')



plt.subplot(223)

all_data.groupby([all_data.date_account_created.dt.month]).count().id.plot(kind='bar', color='g')

plt.title('New accounts by Month\nJul, Aug & Sep are on top')

plt.xlabel('Creation Month')



plt.subplot(224)

all_data.groupby([all_data.date_account_created.dt.year]).count().id.plot(kind='pie', shadow=True, autopct='%.1f%%', pctdistance=0.85)

plt.title('Creation year as percentage of the total user base')

plt.ylabel('User Base')



plt.subplots_adjust(hspace=0.35)

plt.show()

top_lang = all_data.language.value_counts(normalize=True).nlargest(1)

print('{} language is the most common used one, with percentage = {}% from the total users.'.format(top_lang.index[0].upper(), round(top_lang.values[0]*100, 2)))

plt.figure(figsize=(8,6))

plt.title('English is the most common language used')

plt.xlabel('Language')

plt.ylabel('Percentage of users')

all_data.language.value_counts(normalize=True).plot(kind='bar', color='orange', rot=0)

plt.show()
plt.figure(figsize=(18,15))

plt.subplot(221)

ax = sns.countplot(x='signup_method', data=all_data, hue='first_device_type' ,order=all_data.signup_method.value_counts().index)

ax.legend(loc=1)

plt.title('Signup method counts with different devices')

plt.ylabel('')

plt.subplot(222)

sns.countplot(x=all_data.signup_flow)

plt.ylabel('')

plt.title('Signup flow counts')



plt.subplot(223)

sns.countplot(x=all_data.signup_app, order=all_data.signup_app.value_counts().index)

plt.title('Signup application')

plt.subplot(224)

top_x = 7

sizes = np.append(all_data.first_browser.value_counts().iloc[:top_x].values, all_data.first_browser.value_counts().iloc[top_x+1:].values.sum())

labels = np.append(all_data.first_browser.value_counts().iloc[:top_x].index, 'Other')

plt.pie(sizes, labels=labels, autopct='%.1f%%',

        shadow=False, pctdistance=0.85, labeldistance=1.05, startangle=10, explode=[0.15 if (i == 0 or i == len(sizes)-1) else 0 for i in range(len(sizes))])

plt.title('First browser used')

plt.subplots_adjust(hspace=0.35)

plt.show()
plt.figure(figsize=(15,7))

plt.subplot(121)

top_x = 6

sizes = np.append(all_data.affiliate_provider.value_counts().iloc[:top_x].values, all_data.affiliate_provider.value_counts().iloc[top_x+1:].values.sum())

labels = np.append(all_data.affiliate_provider.value_counts().iloc[:top_x].index, 'Other')

plt.bar(x=range(top_x+1), height=sizes, tick_label=labels)

plt.xticks(rotation=45)

plt.title('Affiliate Providers')

plt.subplot(122)

grouped_df = all_data.groupby('affiliate_channel').count().id.nlargest(len(all_data.affiliate_channel.unique()))

explode_thr = 4

plt.pie(grouped_df, labels=grouped_df.index, autopct='%.1f%%', shadow=True, pctdistance=0.88, labeldistance=1.05, startangle=30, 

        explode = [0 if i < explode_thr else (i/len(grouped_df))-(explode_thr/len(grouped_df)) for i in range(len(grouped_df))])

plt.title('Affiliate Channels')

plt.show()
# Load user sessions and find the mean of each action along with most common actions..

sessions = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/sessions.csv')

sessions_grouped = sessions.groupby(['user_id', 'action']).secs_elapsed.mean()

crafted_features = pd.Series(sessions_grouped.index.get_level_values(1)).value_counts().nlargest().index

crafted_features
# Unfortunately we only have sessions data for about half the users only... crafting a feature with this is unlikely

print('You have session data for {} of users'.format(round(len(sessions.user_id.unique())/len(all_data), 2)))
sessions_df = sessions_grouped.unstack()[crafted_features]

print(sessions_df.head())
plt.figure(figsize=(11,8))

sessions[sessions.action.isin(crafted_features)].groupby('action').secs_elapsed.mean().plot(kind='bar', rot=0)

plt.ylabel('Average Elapsed Time In Seconds')

plt.title('Most common 5 actions')

plt.show()
all_data = all_data.join(sessions_grouped.unstack()[crafted_features], how='outer', on='id')

all_data.head()
# Attempts to find a feature to classify the unknown genders.... the header_userpic is the best one I could find, still not good enough

plt.figure(figsize=(10,5))

plt.subplot(121)

sns.barplot(x='gender', y='header_userpic', data=all_data)

plt.subplot(122)

sns.kdeplot(data=all_data[all_data.gender == 'MALE'].age.rename('MALE'))

sns.kdeplot(data=all_data[all_data.gender == 'FEMALE'].age.rename('FEMALE'))

plt.xlabel('Age')

plt.show()
age_gender_countries = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv')

plt.figure(figsize=(15,12))

plt.subplot(221)

sorted_df = age_gender_countries.groupby(["country_destination"]).population_in_thousands.sum().reset_index().sort_values('population_in_thousands', ascending=False)

sns.barplot(x="country_destination", y="population_in_thousands", hue="gender", order=sorted_df.country_destination, data=age_gender_countries, ci=None)

plt.title('Countries Visited By Gender')

plt.xlabel('Country')

plt.ylabel('Population in Thousands')

plt.subplot(222)

age_gender_countries.groupby(["age_bucket"]).population_in_thousands.sum().loc[age_gender_countries.age_bucket.iloc[:21].values[::-1]].plot(kind='bar', rot=45, color=plt.cm.tab20c(np.arange(len(age_gender_countries.age_bucket.unique()))))

plt.title('Age Buckets vs Number Of Users Who Made At Least a Booking in 2015')

plt.xlabel('Age Bucket')

plt.subplot(212)

plt.title('Age Buckets per Country vs Count')

buckets_count = len(age_gender_countries.age_bucket.unique())

ax = sns.barplot(x='country_destination', y="population_in_thousands", 

                 hue='age_bucket', hue_order=age_gender_countries.age_bucket.iloc[:buckets_count].values[::-1], 

                 data=age_gender_countries, ci=None, palette=sns.color_palette("tab20c", buckets_count))

ax.legend(title='Age Bucket', bbox_to_anchor=(1, 0.5), loc=6)

plt.xlabel('Country of Destination')

plt.ylabel('Population in Thousands')

plt.subplots_adjust(hspace=0.35)

plt.show()
countries = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/countries.csv')

countries.head()
# A comparison of countries area..

wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(zip(countries.country_destination.values, countries.destination_km2.values)))

fig, ax = plt.subplots(1,1,figsize=(10,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
# An engineered feature to see if people prefer to travel to countries close to them with minor language levenshtein distance

plt.figure(figsize=(10,5))

countries_lang_distance = train.merge(countries, on='country_destination', how='outer')[['language_levenshtein_distance',

                                                                                         'distance_km']]

countries_lang_distance=(countries_lang_distance-countries_lang_distance.min())/(countries_lang_distance.max()-countries_lang_distance.min()) #Normalize..

countries_lang_distance.dropna(how='all').mean(axis=1).plot(kind='hist')

plt.title('No relation could be seen')

plt.xlabel('Mean of Language Levenshtein Distance and Real Distance of Destination')

plt.ylabel('Number of travellers')

plt.show()
countries.head(20)
# One Hot Encoding of Categorial Features:

choosen_ohe_features = ['gender', 'language', 'affiliate_provider', 'first_affiliate_tracked', 'affiliate_channel',

                    'signup_app', 'first_device_type','first_browser']



ohe_df = pd.get_dummies(all_data[choosen_ohe_features])



numerical_features = [all_data.timestamp_first_active.dt.year.rename('first_active_year'), 

                      all_data.timestamp_first_active.dt.month.rename('first_active_month'), 

                      all_data.date_account_created.dt.year.rename('account_creation_year'), 

                      all_data.date_account_created.dt.month.rename('account_creation_month'), 

                      all_data.age]





num_df = pd.concat(numerical_features, axis=1)



# changing month to cyclical feature

num_df['firth_active_month_sin'] = np.sin((num_df.first_active_month-1)*(2.*np.pi/12))

num_df['firth_active_month_cos'] = np.cos((num_df.first_active_month-1)*(2.*np.pi/12))

num_df['account_creation_month_sin'] = np.sin((num_df.account_creation_month-1)*(2.*np.pi/12))

num_df['account_creation_month_cos'] = np.cos((num_df.account_creation_month-1)*(2.*np.pi/12))

num_df.drop(['first_active_month', 'account_creation_month'], axis=1, inplace=True)



num_df=(num_df-num_df.min())/(num_df.max()-num_df.min()) #Normalize..

num_df.head()
df = pd.concat([num_df, ohe_df], axis=1)

df.head()
# Resplitting into train/test and calculating accuracy for random guess

new_train = df.iloc[:len(train)]

new_test = df.iloc[len(new_train):]

print('If you randomly guess using the same distribution, you should have an accuracy around {}%'.format(round(target_labels.value_counts(normalize=True).nlargest(1).values[0]*100, 2)))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

import numpy as np



nclasses = len(np.unique(target_labels))

nfeatures = np.size(new_train, axis=1)



target_labels_keras = to_categorical(target_labels)



# keras model

model = Sequential()

model.add(Dense(nfeatures, activation='elu', kernel_initializer='he_normal', input_shape=(nfeatures,)))

model.add(Dense(150, activation='elu', kernel_initializer='he_normal'))

model.add(Dense(100, activation='elu', kernel_initializer='he_normal'))

model.add(Dense(50, activation='elu', kernel_initializer='he_normal'))

model.add(Dense(20, activation='elu', kernel_initializer='he_normal'))

model.add(Dense(15, activation='elu', kernel_initializer='he_normal'))

model.add(Dense(nclasses, activation='softmax', kernel_initializer='he_normal'))



# compile model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#fit model

model.fit(new_train, target_labels_keras, validation_split=0.15, epochs=15)
y_pred = model.predict(new_test)

y_pred = np.flip(np.argsort(y_pred), axis=1)[:, :5]
submission = pd.concat([test.id.repeat(5).reset_index(drop=True), pd.Series(y_pred.ravel()).rename('country')], axis=1)

submission['country'] = le.inverse_transform(submission.country)

submission.head()
# model = ComplementNB()

# model = MultinomialNB()

# model = XGBClassifier(n_estimators=25, max_depth=7, learning_rate=0.2, 

#                       objective='multi:softprob', seed=555, subsample=0.5, colsample_bytree=0.5)  

#model = RandomForestClassifier(n_estimators=25, random_state=555, max_depth=7, random_seed=555)

# model.fit(new_train, target_labels)

# y_pred = model.predict(new_test)
submission.to_csv('submission.csv', index=False)
%matplotlib inline

import pandas as pd

import numpy as np

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
df_agb = pd.read_csv('../input/age_gender_bkts.csv')

df_agb.head()
df_agb.isnull().values.any()
#Convert 100+ into a bin.

df_agb['age_bucket'] = df_agb['age_bucket'].apply(lambda x: '100-104' if x == '100+' else x)

#Define mean_age feature

df_agb['mean_age'] = df_agb['age_bucket'].apply(lambda x: (int(x.split('-')[0]) + int(x.split('-')[1]))/2)

df_agb = df_agb.drop('age_bucket', axis=1)

df_agb.head()
df_agb['country_destination'].value_counts()
df_agb['gender'].value_counts()
df_agb['gender'] = df_agb['gender'].apply(lambda x: 0 if x == 'male' else 1)

df_agb['gender'].value_counts()
df_agb['year'].value_counts()
df_agb = df_agb.drop('year', axis=1)

df_agb.head()
df_con = pd.read_csv('../input/countries.csv')

df_con
df_ses = pd.read_csv('../input/sessions.csv')

df_ses.head(15)
df_ses.shape
df_ses['action'] = df_ses['action'].replace('-unknown-', np.nan)

df_ses['action_type'] = df_ses['action_type'].replace('-unknown-', np.nan)

df_ses['action_detail'] = df_ses['action_detail'].replace('-unknown-', np.nan)
sns.distplot(df_ses[df_ses['secs_elapsed'].notnull()]['secs_elapsed'])
df_ses['secs_elapsed'].describe()
len(df_ses[df_ses['secs_elapsed'].isnull()])
median_secs = df_ses['secs_elapsed'].median()

df_ses['secs_elapsed'] = df_ses['secs_elapsed'].fillna(median_secs)
df_ses['secs_elapsed'].describe()
null_action = df_ses[(df_ses['action_type'].isnull()) | (df_ses['action_detail'].isnull()) | (df_ses['action'].isnull()) ]

null_action.head()
null_action.shape
len(null_action['action'].drop_duplicates())
df_ses[df_ses['device_type'].isnull()]
df_ses['device_type'] = df_ses['device_type'].replace('-unknown-', np.nan)
df_ses['device_type'].value_counts()
len(df_ses[df_ses['device_type'].isnull()])
df_train = pd.read_csv('../input/train_users_2.csv')

df_train.head()
df_train.shape
df_train['gender'] = df_train['gender'].replace('-unknown-', np.nan)

df_train['first_browser'] = df_train['first_browser'].replace('-unknown-', np.nan)
df_train[df_train['first_device_type'].isnull()]
df_train[df_train['age'] > 120].head()
df_train['age'] = df_train['age'].apply(lambda x: np.nan if x > 120 else x)
df_inf = df_train[(df_train['country_destination'] != 'NDF') & (df_train['country_destination'] != 'other') & (df_train['gender'] != 'OTHER') & (df_train['gender'].notnull())]

df_inf = df_inf[['id', 'gender', 'country_destination']]

df_inf.head()
df_inf['gender'].value_counts()
df_inf['country_destination'].value_counts()
observed = df_inf.pivot_table('id', ['gender'], 'country_destination', aggfunc='count').reset_index()

del observed.columns.name

observed = observed.set_index('gender')

observed
chi2, p, dof, expected = stats.chi2_contingency(observed)
chi2
p
df_signup = df_train[(df_train['signup_method'] != 'google')][['id', 'signup_method', 'signup_app']]

df_signup['device'] = df_signup['signup_app'].apply(lambda x: 'Computer' if x == 'Web' else 'Mobile')

df_signup.head()
df_signup['signup_method'].value_counts()
df_signup['device'].value_counts()
df_signup = df_signup.pivot_table('id', ['device'], 'signup_method', aggfunc='count')

df_signup.index = ['Computer', 'Mobile']

df_signup.columns = ['Basic', 'Facebook']

df_signup
chi2, p, dof, expected = stats.chi2_contingency(df_signup, correction=False)
chi2
p
df_signup.loc['Total'] = [ df_signup['Basic'].sum(), df_signup['Facebook'].sum()]

df_signup['Total'] = df_signup['Basic'] + df_signup['Facebook']

df_signup
fb_prop = df_signup.loc['Mobile', 'Facebook']/df_signup.loc['Total', 'Facebook']

fb_std = df_signup.loc['Mobile', 'Facebook'] * ((1 - fb_prop) ** 2) + df_signup.loc['Computer', 'Facebook'] * ((0 - fb_prop) ** 2)

fb_std = np.sqrt(fb_std/df_signup.loc['Total', 'Facebook'])



fb_prop, fb_std
basic_prop = df_signup.loc['Mobile', 'Basic']/df_signup.loc['Total', 'Basic']

basic_std = df_signup.loc['Mobile', 'Basic'] * ((1 - basic_prop) ** 2) + df_signup.loc['Computer', 'Basic'] * ((0 - basic_prop) ** 2)

basic_std = np.sqrt(basic_std/df_signup.loc['Total', 'Basic'])



basic_prop, basic_std
h0_prop = 0



prop_diff = fb_prop - basic_prop

p_hat = (df_signup.loc['Mobile', 'Basic'] + df_signup.loc['Mobile', 'Facebook'])/(df_signup.loc['Total', 'Basic'] + df_signup.loc['Total', 'Facebook']) 

var_diff = p_hat * (1- p_hat) * (1/df_signup.loc['Total', 'Basic'] + 1/df_signup.loc['Total', 'Facebook'])

sigma_diff = np.sqrt(var_diff)



prop_diff, sigma_diff
z = (prop_diff - h0_prop) / sigma_diff

z
p = (1-stats.norm.cdf(z))*2

p
plt.figure(figsize=(20,8))

sns.barplot(x='mean_age', y='population_in_thousands', hue='gender', data=df_agb, ci=None)
sns.set_style('whitegrid')

plt.figure(figsize=(10,7))

pop_stats = df_agb.groupby('country_destination')['population_in_thousands'].sum()

sns.barplot(x=pop_stats.index, y=pop_stats)
sns.set_style('whitegrid')

plt.figure(figsize=(10,7))

sns.barplot(x='country_destination', y='distance_km', data=df_con)
country_popularity = df_train[(df_train['country_destination'] != 'NDF') & (df_train['country_destination'] != 'other')]['country_destination'].value_counts()
country_distance = pd.Series(df_con['distance_km'])

country_distance.index = df_con['country_destination']
language_distance = pd.Series(df_con['language_levenshtein_distance'])

language_distance.index = df_con['country_destination']
country_area = pd.Series(df_con['destination_km2'])

country_area.index = df_con['country_destination']
df_dp = pd.concat([country_popularity, country_distance, language_distance, country_area], axis=1)

df_dp.columns = ['count', 'distance_km', 'language', 'area']

sns.jointplot(x='count', y='distance_km', data=df_dp)
sns.jointplot(x='count', y='distance_km', data=df_dp.drop('US'))
sns.jointplot(x='count', y='language', data=df_dp)
sns.jointplot(x='count', y='language', data=df_dp.drop('US'))
sns.jointplot(x='count', y='area', data=df_dp)
sns.jointplot(x='count', y='area', data=df_dp.drop('US'))
sns.distplot(df_ses[df_ses['secs_elapsed'].notnull()]['secs_elapsed'])
sns.distplot(df_ses[(df_ses['secs_elapsed'].notnull()) & (df_ses['secs_elapsed'] < 5000)]['secs_elapsed'])
len(df_ses[df_ses['secs_elapsed'] < 1000])/len(df_ses[df_ses['secs_elapsed'].notnull()])
plt.figure(figsize=(12,7))

sns.countplot(y='device_type', data=df_ses)
plt.figure(figsize=(10,5))

country_share = df_train['country_destination'].value_counts() / df_train.shape[0] * 100

country_share.plot(kind='bar',color='#FD5C64', rot=0)

plt.xlabel('Destination Country')

plt.ylabel('Percentage')

sns.despine()
classes = ['NDF','US','other','FR','IT','GB','ES','CA','DE','NL','AU','PT']
def stacked_bar(feature):

    ctab = pd.crosstab([df_train[feature].fillna('Unknown')], df_train.country_destination, dropna=False).apply(lambda x: x/x.sum(), axis=1)

    ctab[classes].plot(kind='bar', stacked=True, colormap='terrain', legend=False)
sns.countplot(df_train['gender'].fillna('Unknown'))
stacked_bar('gender')
sns.distplot(df_train['age'].dropna())
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

sns.boxplot(x='country_destination', y='age', data=df_train, palette="muted", ax =ax)

ax.set_ylim([10, 75])
def set_age_group(x):

    if x < 40:

        return 'Young'

    elif x >=40 and x < 60:

        return 'Middle'

    elif x >= 60 and x <= 125:

        return 'Old'

    else:

        return 'Unknown'
df_train['age_group'] = df_train['age'].apply(set_age_group)
stacked_bar('age_group')
stacked_bar('signup_method')
stacked_bar('language')
stacked_bar('affiliate_channel')
plt.figure(figsize=(8,4))

sns.countplot(df_train['affiliate_channel'])
stacked_bar('affiliate_provider')
plt.figure(figsize=(18,4))

sns.countplot(df_train['affiliate_provider'])
stacked_bar('first_affiliate_tracked')
stacked_bar('signup_flow')
stacked_bar('signup_app')
stacked_bar('first_device_type')
df_train['date_account_created'] = pd.to_datetime(df_train['date_account_created'])
sns.set_style("whitegrid", {'axes.edgecolor': '0'})

sns.set_context("poster", font_scale=1.1)

plt.figure(figsize=(12,6))



df_train[df_train['country_destination'] != 'NDF']['date_account_created'].value_counts().plot(kind='line', linewidth=1, color='green')

df_train[df_train['country_destination'] == 'NDF']['date_account_created'].value_counts().plot(kind='line', linewidth=1, color='red')
df_ses.head(2)
def session_features(df):

    df['total_seconds'] = df['id'].apply(lambda x: total_seconds[x] if x in total_seconds else 0)

    df['average_seconds'] = df['id'].apply(lambda x: average_seconds[x] if x in average_seconds else 0)

    df['total_sessions'] = df['id'].apply(lambda x: total_sessions[x] if x in total_sessions else 0)

    df['distinct_sessions'] = df['id'].apply(lambda x: distinct_sessions[x] if x in distinct_sessions else 0)

    df['num_short_sessions'] = df['id'].apply(lambda x: num_short_sessions[x] if x in num_short_sessions else 0)

    df['num_long_sessions'] = df['id'].apply(lambda x: num_long_sessions[x] if x in num_long_sessions else 0)

    df['num_devices'] = df['id'].apply(lambda x: num_devices[x] if x in num_devices else 0)

    return df
def browsers(df):

    df['first_browser'] = df['first_browser'].apply(lambda x: "Mobile_Safari" if x == "Mobile Safari" else x)

    major_browsers = ['Chrome', 'Safari', 'Firefox', 'IE', 'Mobile_Safari']

    df['first_browser'] = df['first_browser'].apply(lambda x: 'Other' if x not in major_browsers else x)

    return df
def classify_device(x):

    if x.find('Desktop') != -1:

        return 'Desktop'

    elif x.find('Tablet') != -1 or x.find('iPad') != -1:

        return 'Tablet'

    elif x.find('Phone') != -1:

        return 'Phone'

    else:

        return 'Unknown'
def devices(df):

    df['first_device_type'] = df['first_device_type'].apply(classify_device)

    return df
def affiliate_tracked(df):

    df['first_affiliate_tracked'] = df['first_affiliate_tracked'].fillna('Unknown')

    df['first_affiliate_tracked'] = df['first_affiliate_tracked'].apply(lambda x: 'Other' if x != 'Unknown' and x != 'untracked' else x)

    return df
def affiliate_provider(df):

    df['affiliate_provider'] = df['affiliate_provider'].apply(lambda x: 'rest' if x not in ['direct', 'google', 'other'] else x)

    return df
def affiliate_channel(df):

    df['affiliate_channel'] = df['affiliate_channel'].apply(lambda x: 'other' if x  not in ['direct', 'content'] else x)

    return df
def languages(df):

    df['language'] = df['language'].apply(lambda x: 'foreign' if x != 'en' else x)

    return df
def first_booking(df):

    df = df.drop('date_first_booking', axis=1)

    return df
def account_created(df):

    df = df.drop('date_account_created', axis=1)

    return df
def feature_engineering(df):

    df = session_features(df)

    df = df.drop('age', axis=1)

    df = browsers(df)

    df =devices(df)

    df =affiliate_tracked(df)

    df = affiliate_provider(df)

    df = affiliate_channel(df)

    df = languages(df)

    df['is_3'] = df['signup_flow'].apply(lambda x: 1 if x==3 else 0)

    df = first_booking(df)

    df = df.drop('timestamp_first_active', axis=1)

    df = account_created(df)

    df = df.set_index('id')

    df = pd.get_dummies(df, prefix='is')

    return df
total_seconds = df_ses.groupby('user_id')['secs_elapsed'].sum()
average_seconds = df_ses.groupby('user_id')['secs_elapsed'].mean()
total_sessions = df_ses.groupby('user_id')['action'].count()
distinct_sessions = df_ses.groupby('user_id')['action'].nunique()
num_short_sessions = df_ses[df_ses['secs_elapsed'] <= 300].groupby('user_id')['action'].count()

num_long_sessions = df_ses[df_ses['secs_elapsed'] >= 2000].groupby('user_id')['action'].count()
num_devices = df_ses.groupby('user_id')['device_type'].nunique()
df_train = session_features(df_train)
df_train = df_train.drop('age', axis=1)
df_train = browsers(df_train)
df_train = devices(df_train)
df_train = affiliate_tracked(df_train)
df_train = affiliate_provider(df_train)
df_train = affiliate_channel(df_train)
df_train = languages(df_train)
df_train['is_3'] = df_train['signup_flow'].apply(lambda x: 1 if x==3 else 0)

df_train['gender'] = df_train['gender'].fillna('Unknown')
df_train = first_booking(df_train)
df_train = df_train.drop('timestamp_first_active', axis=1)
df_train = account_created(df_train)
df_train = df_train.set_index('id')
class_dict = {

    'NDF': 0,

    'US': 1,

    'other': 2,

    'FR': 3,

    'CA': 4,

    'GB': 5,

    'ES': 6,

    'IT': 7,

    'PT': 8,

    'NL': 9,

    'DE': 10,

    'AU': 11

}
X, y = df_train.drop('country_destination', axis=1), df_train['country_destination'].apply(lambda x: class_dict[x])
X = pd.get_dummies(X, prefix='is')
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, stratify=y)
#classifiers = [RandomForestClassifier(verbose=1), LogisticRegression(verbose=1), GradientBoostingClassifier(verbose=True)]



#for classifier in classifiers:

#    classifier.fit(train_X, train_y)

#    print("Score: " + str(classifier.score(test_X, test_y)))
parameters = {

    'n_estimators': [100,200],

    'max_features': ['auto', 'log2'],

    'max_depth': [3,5]

}
#clf = GridSearchCV(GradientBoostingClassifier(), parameters, verbose=100)

#clf.fit(train_X, train_y)
#clf.best_params_
df_test = pd.read_csv('../input/test_users.csv')

df_test['gender'] = df_test['gender'].replace('-unknown-', 'Unknown')

df_test['age_group'] = df_test['age'].apply(set_age_group)

df_test.head()
#df_test = feature_engineering(df_test)

#df_test = df_test.drop('is_weibo', axis=1)
#df_test.columns
#X.columns
#pred_prob = clf.predict_proba(df_test)
#pred_prob = pd.DataFrame(pred_prob, index=df_test.index)

#pred_prob.head()
#inv_classes = {v: k for k, v in class_dict.items()}

#inv_classes
def get_top(s):

    indexes = [i for i in range(0,12)]

    lst = list(zip(indexes, s))

    top_five = sorted(lst, key=lambda x: x[1], reverse=True)[:5]

    top_five = [inv_classes[i[0]] for i in top_five]

    return str(top_five)
#pred_prob['get_top'] = pred_prob.apply(get_top, axis=1)

#pred_prob.head()
#import ast

#pred_prob['get_top'] = pred_prob['get_top'].apply(lambda x: ast.literal_eval(x))
#s = pred_prob.apply(lambda x: pd.Series(x['get_top']),axis=1).stack().reset_index(level=1, drop=True)

#s.name = 'country'
#submission = pred_prob.drop([i for i in range(0,12)] + ['get_top'], axis=1).join(s)

#submission.head()
#submission.to_csv('submission.csv')
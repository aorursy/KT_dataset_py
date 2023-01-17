# imports
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# list of data files available
print(os.listdir('../input'))
# training data
app_train = pd.read_csv('../input/application_train.csv')
print('training data shape: ', app_train.shape)
app_train.head()
# testing data
app_test = pd.read_csv('../input/application_test.csv')
print('testing data shape: ', app_test.shape)
app_test.head()
# review target variable
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();
(app_train['TARGET']).describe()
# summary of training data
app_train.describe()
# join training and testing data sets so we keep the same number of features in both
train_len = len(app_train)
dataset = pd.concat(objs=[app_train, app_test], axis=0).reset_index(drop=True)
# shape of combined dataset should be sum of rows in training and testing (307511 + 48744 = 356255) and 122 columns (testing data doesn't have target)
print('dataset data shape: ', dataset.shape)
# missing values
dataset.isnull().sum()
# types of data
dataset.dtypes.value_counts()
# categorical data - how many different categories for each variable
dataset.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
dataset.describe(include=[np.object])
# use label encoding for categorical variables with only two categories
le = LabelEncoder()
count = 0
le_vars = []
for col in dataset:
    if dataset[col].dtype == 'object':
        if len(list(dataset[col].unique())) == 2:
            le.fit(dataset[col])
            dataset[col] = le.transform(dataset[col])
            count += 1
            le_vars.append(col)
            
print('%d columns were label encoded' % count)
print(le_vars)
# use one-hot encoding for remaining categorical variables
dataset = pd.get_dummies(dataset)
print('dataset data shape: ', dataset.shape)
(dataset['CNT_CHILDREN']).describe()
dataset['CNT_CHILDREN'].plot.hist(title = 'CNT_CHILDREN Histogram');
plt.xlabel('CNT_CHILDREN')
# plot CNT_CHILDREN against the TARGET to better understand the data
g = sns.factorplot(x='CNT_CHILDREN', y='TARGET', data=app_train, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("default probability")
outlier_children = app_train[app_train['CNT_CHILDREN'] > 6]
print('count of outlier_children: ', len(outlier_children))
print('default probability of outlier_children: %0.2f%%' %(100 * outlier_children['TARGET'].mean()))
(outlier_children['CNT_CHILDREN']).describe()
# create a flag for outliers in the CNT_CHILDREN column, and then replace these values with nan
dataset['CNT_CHILDREN_outlier'] = dataset['CNT_CHILDREN'] > 6
for i in dataset['CNT_CHILDREN']:
    if i > 6:
        dataset['CNT_CHILDREN'].replace({i: np.nan}, inplace = True)
# review CNT_CHILDREN after our modifications
(dataset['CNT_CHILDREN']).describe()
dataset['CNT_CHILDREN'].plot.hist(title = 'CNT_CHILDREN Histogram');
plt.xlabel('CNT_CHILDREN')
(dataset['AMT_INCOME_TOTAL']).describe()
dataset['AMT_INCOME_TOTAL'].plot.hist(range = (1,1000000), title = 'AMT_INCOME_TOTAL Histogram');
plt.xlabel('AMT_INCOME_TOTAL')
(dataset['AMT_CREDIT']).describe()
dataset['AMT_CREDIT'].plot.hist(title = 'AMT_CREDIT Histogram');
plt.xlabel('AMT_CREDIT')
(dataset['AMT_ANNUITY']).describe()
dataset['AMT_ANNUITY'].plot.hist(title = 'AMT_ANNUITY Histogram');
plt.xlabel('AMT_ANNUITY')
(dataset['AMT_GOODS_PRICE']).describe()
dataset['AMT_GOODS_PRICE'].plot.hist(title = 'AMT_GOODS_PRICE Histogram');
plt.xlabel('AMT_GOODS_PRICE')
(dataset['REGION_POPULATION_RELATIVE']).describe()
dataset['REGION_POPULATION_RELATIVE'].plot.hist(title = 'REGION_POPULATION_RELATIVE Histogram');
plt.xlabel('REGION_POPULATION_RELATIVE')
# plot REGION_POPULATION_RELATIVE against the TARGET to better understand the data
g = sns.lineplot(x='REGION_POPULATION_RELATIVE', y='TARGET', data=app_train, palette = "muted")
g = sns.heatmap(app_train[['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
(dataset['DAYS_BIRTH']).describe()
# this variable appears to be equal to (date of birth) minus (date of application), which is producing negative numbers
# if we look again at the data transformed into positive numbers and into years (by dividing by -365.25) we get the following
(dataset['DAYS_BIRTH'] / -365.25).describe()
(dataset['DAYS_BIRTH'] / -365.25).plot.hist(title = 'DAYS_BIRTH Histogram');
plt.xlabel('DAYS_BIRTH')
(dataset['DAYS_EMPLOYED']).describe()
# this variable appears to be equal to (date of employment) minus (date of application), which is producing negative numbers
# if we look again at the data transformed into positive numbers and into years (by dividing by -365.25) we get the following
(dataset['DAYS_EMPLOYED'] / -365.25).describe()
# it appears that a dummy value was used, possibly for people who didn't have a date of employment to enter into the application
# this group had 365243 in the data, which is approximately -1000 years
# we should also look at the other side of the distribution - 49 years of employment is a long time
dataset['DAYS_EMPLOYED'].plot.hist(title = 'DAYS_EMPLOYED Histogram');
plt.xlabel('DAYS_EMPLOYED')
outlier_days_employed = app_train[app_train['DAYS_EMPLOYED'] == 365243]
print('count of outlier_days_employed: ', len(outlier_days_employed))
print('default probability of outlier_days_employed: %0.2f%%' %(100 * outlier_days_employed['TARGET'].mean()))
(outlier_days_employed['DAYS_EMPLOYED']).describe()
# create a flag for outliers in the DAYS_EMPLOYED column, and then replace these values with nan
dataset['DAYS_EMPLOYED_outlier'] = dataset['DAYS_EMPLOYED'] == 365243
dataset['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
# review DAYS_EMPLOYED after our modifications
(dataset['DAYS_EMPLOYED']).describe()
(dataset['DAYS_EMPLOYED'] / -365.25).plot.hist(title = 'DAYS_EMPLOYED Histogram');
plt.xlabel('DAYS_EMPLOYED')
(dataset['DAYS_REGISTRATION']).describe()
# this variable appears to be equal to (date of registration) minus (date of application), which is producing negative numbers
# if we look again at the data transformed into positive numbers and into years (by dividing by 365.25) we get the following
(dataset['DAYS_REGISTRATION'] / -365.25).describe()
(dataset['DAYS_REGISTRATION'] / -365.25).plot.hist(title = 'DAYS_REGISTRATION Histogram');
plt.xlabel('DAYS_REGISTRATION')
(dataset['DAYS_ID_PUBLISH']).describe()
# convert to positive years again
(dataset['DAYS_ID_PUBLISH'] / -365.25).describe()
(dataset['DAYS_ID_PUBLISH'] / -365.25).plot.hist(title = 'DAYS_ID_PUBLISH Histogram');
plt.xlabel('DAYS_ID_PUBLISH')
g = sns.heatmap(app_train[['TARGET','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
(dataset['OWN_CAR_AGE']).describe()
dataset['OWN_CAR_AGE'].plot.hist(title = 'OWN_CAR_AGE Histogram');
plt.xlabel('OWN_CAR_AGE')
# it seems that there are some outliers on the car age, as the max is 91
# let's get a better look at the values in the tail and whether these have a higher probability of default than average
outlier_car_age = app_train[app_train['OWN_CAR_AGE'] > 60]
print('count of outlier_car_age: ', len(outlier_car_age))
print('default probability of outlier_car_age: %0.2f%%' %(100 * outlier_car_age['TARGET'].mean()))
(outlier_car_age['OWN_CAR_AGE']).describe()
outlier_car_age['OWN_CAR_AGE'].plot.hist(title = 'OWN_CAR_AGE Outlier Histogram');
plt.xlabel('OWN_CAR_AGE')
# create a flag for outliers in the OWN_CAR_AGE column, and then replace these values with nan
dataset['OWN_CAR_AGE_outlier'] = dataset['OWN_CAR_AGE'] > 60
for i in dataset['OWN_CAR_AGE']:
    if i > 60:
        dataset['OWN_CAR_AGE'].replace({i: np.nan}, inplace = True)
# review OWN_CAR_AGE after our modifications
(dataset['OWN_CAR_AGE']).describe()
# now this data should look more like we expect
dataset['OWN_CAR_AGE'].plot.hist(title = 'OWN_CAR_AGE Histogram');
plt.xlabel('OWN_CAR_AGE')
(dataset['CNT_FAM_MEMBERS']).describe()
dataset['CNT_FAM_MEMBERS'].plot.hist(title = 'CNT_FAM_MEMBERS Histogram');
plt.xlabel('CNT_FAM_MEMBERS')
# it seems that there are some outliers on the count of family members, as the max is 21
# let's look at the 99th percentile
print(np.nanpercentile(dataset['CNT_FAM_MEMBERS'], 99))
# let's get a better look at the values in the tail and whether these have a higher probability of default than average
outlier_fam_mem = app_train[app_train['CNT_FAM_MEMBERS'] > 5]
print('count of outlier_fam_mem: ', len(outlier_fam_mem))
print('default probability of outlier_fam_mem: %0.2f%%' %(100 * outlier_fam_mem['TARGET'].mean()))
(outlier_fam_mem['CNT_FAM_MEMBERS']).describe()
# create a flag for outliers in the CNT_FAM_MEMBERS column, and then replace these values with nan
dataset['CNT_FAM_MEMBERS_outlier'] = dataset['CNT_FAM_MEMBERS'] > 5
for i in dataset['CNT_FAM_MEMBERS']:
    if i > 5:
        dataset['CNT_FAM_MEMBERS'].replace({i: np.nan}, inplace = True)
# review CNT_FAM_MEMBERS after our modifications
(dataset['CNT_FAM_MEMBERS']).describe()
dataset['CNT_FAM_MEMBERS'].plot.hist(title = 'CNT_FAM_MEMBERS Histogram');
plt.xlabel('CNT_FAM_MEMBERS')
(dataset['REGION_RATING_CLIENT']).describe()
dataset['REGION_RATING_CLIENT'].plot.hist(title = 'REGION_RATING_CLIENT Histogram');
plt.xlabel('REGION_RATING_CLIENT')
(dataset['REGION_RATING_CLIENT_W_CITY']).describe()
dataset['REGION_RATING_CLIENT_W_CITY'].plot.hist(title = 'REGION_RATING_CLIENT_W_CITY Histogram');
plt.xlabel('REGION_RATING_CLIENT_W_CITY')
# how many are equal to -1 in the dataset?
dataset['REGION_RATING_CLIENT_W_CITY'].map(lambda s: 1 if s == -1 else 0).sum()
# this appears to be a data entry error
# let's set the value of -1 equal to 1 instead
for i in dataset['REGION_RATING_CLIENT_W_CITY']:
    if i == -1:
        dataset['REGION_RATING_CLIENT_W_CITY'].replace({i: 1}, inplace = True)
(dataset['REGION_RATING_CLIENT_W_CITY']).describe()
(dataset['HOUR_APPR_PROCESS_START']).describe()
dataset['HOUR_APPR_PROCESS_START'].plot.hist(title = 'HOUR_APPR_PROCESS_START Histogram');
plt.xlabel('HOUR_APPR_PROCESS_START')
g = sns.heatmap(app_train[['TARGET','OWN_CAR_AGE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','HOUR_APPR_PROCESS_START']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
(dataset['EXT_SOURCE_1']).describe()
# this data looks pretty good from the above, check the histogram
dataset['EXT_SOURCE_1'].plot.hist(title = 'EXT_SOURCE_1 Histogram');
plt.xlabel('EXT_SOURCE_1')
(dataset['EXT_SOURCE_2']).describe()
# this data also looks pretty good from the above, check the histogram
dataset['EXT_SOURCE_2'].plot.hist(title = 'EXT_SOURCE_2 Histogram');
plt.xlabel('EXT_SOURCE_2')
(dataset['EXT_SOURCE_3']).describe()
# this doesn't appear to have issues either, check the histogram
dataset['EXT_SOURCE_3'].plot.hist(title = 'EXT_SOURCE_3 Histogram');
plt.xlabel('EXT_SOURCE_3')
g = sns.heatmap(app_train[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
(dataset['OBS_30_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])
# there appear to be some outliers here we may need to deal with (max = 354??)
dataset['OBS_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_30_CNT_SOCIAL_CIRCLE')
# let's look more at these outliers and whether these have a higher probability of default than average
outlier_obs_30_social = app_train[app_train['OBS_30_CNT_SOCIAL_CIRCLE'] > 17]
print('count of outlier_obs_30_social: ', len(outlier_obs_30_social))
print('default probability of outlier_obs_30_social: %0.2f%%' %(100 * outlier_obs_30_social['TARGET'].mean()))
(outlier_obs_30_social['OBS_30_CNT_SOCIAL_CIRCLE']).describe()
# create a flag for outliers in the OBS_30_CNT_SOCIAL_CIRCLE column, and then replace these values with nan
dataset['OBS_30_CNT_SOCIAL_CIRCLE_outlier'] = dataset['OBS_30_CNT_SOCIAL_CIRCLE'] > 17
for i in dataset['OBS_30_CNT_SOCIAL_CIRCLE']:
    if i > 17:
        dataset['OBS_30_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)
# review OBS_30_CNT_SOCIAL_CIRCLE after our modifications
(dataset['OBS_30_CNT_SOCIAL_CIRCLE']).describe()
dataset['OBS_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_30_CNT_SOCIAL_CIRCLE')
(dataset['DEF_30_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])
dataset['DEF_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_30_CNT_SOCIAL_CIRCLE')
outlier_def_30_social = app_train[app_train['DEF_30_CNT_SOCIAL_CIRCLE'] > 5]
print('count of outlier_def_30_social: ', len(outlier_def_30_social))
print('default probability of outlier_def_30_social: %0.2f%%' %(100 * outlier_def_30_social['TARGET'].mean()))
(outlier_def_30_social['DEF_30_CNT_SOCIAL_CIRCLE']).describe()
dataset['DEF_30_CNT_SOCIAL_CIRCLE_outlier'] = dataset['DEF_30_CNT_SOCIAL_CIRCLE'] > 5
for i in dataset['DEF_30_CNT_SOCIAL_CIRCLE']:
    if i > 5:
        dataset['DEF_30_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)
(dataset['DEF_30_CNT_SOCIAL_CIRCLE']).describe()
dataset['DEF_30_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_30_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_30_CNT_SOCIAL_CIRCLE')
(dataset['OBS_60_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])
dataset['OBS_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_60_CNT_SOCIAL_CIRCLE')
outlier_obs_60_social = app_train[app_train['OBS_60_CNT_SOCIAL_CIRCLE'] > 16]
print('count of outlier_obs_60_social: ', len(outlier_obs_60_social))
print('default probability of outlier_obs_60_social: %0.2f%%' %(100 * outlier_obs_60_social['TARGET'].mean()))
(outlier_obs_60_social['OBS_60_CNT_SOCIAL_CIRCLE']).describe()
dataset['OBS_60_CNT_SOCIAL_CIRCLE_outlier'] = dataset['OBS_60_CNT_SOCIAL_CIRCLE'] > 16
for i in dataset['OBS_60_CNT_SOCIAL_CIRCLE']:
    if i > 16:
        dataset['OBS_60_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)
(dataset['OBS_60_CNT_SOCIAL_CIRCLE']).describe()
dataset['OBS_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'OBS_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('OBS_60_CNT_SOCIAL_CIRCLE')
(dataset['DEF_60_CNT_SOCIAL_CIRCLE']).describe(percentiles=[0.25,0.5,0.75,0.99,0.999,0.9999,0.99999])
dataset['DEF_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_60_CNT_SOCIAL_CIRCLE')
outlier_def_60_social = app_train[app_train['DEF_60_CNT_SOCIAL_CIRCLE'] > 4]
print('count of outlier_def_60_social: ', len(outlier_def_60_social))
print('default probability of outlier_def_60_social: %0.2f%%' %(100 * outlier_def_60_social['TARGET'].mean()))
(outlier_def_60_social['DEF_60_CNT_SOCIAL_CIRCLE']).describe()
dataset['DEF_60_CNT_SOCIAL_CIRCLE_outlier'] = dataset['DEF_60_CNT_SOCIAL_CIRCLE'] > 4
for i in dataset['DEF_60_CNT_SOCIAL_CIRCLE']:
    if i > 4:
        dataset['DEF_60_CNT_SOCIAL_CIRCLE'].replace({i: np.nan}, inplace = True)
(dataset['DEF_60_CNT_SOCIAL_CIRCLE']).describe()
dataset['DEF_60_CNT_SOCIAL_CIRCLE'].plot.hist(title = 'DEF_60_CNT_SOCIAL_CIRCLE Histogram');
plt.xlabel('DEF_60_CNT_SOCIAL_CIRCLE')
g = sns.heatmap(app_train[['TARGET','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
(dataset['DAYS_LAST_PHONE_CHANGE']).describe()
# let's transform this into positive years, as we did with the other DAYS_ variables above
(dataset['DAYS_LAST_PHONE_CHANGE'] / -365.25).describe()
(dataset['DAYS_LAST_PHONE_CHANGE'] / -365.25).plot.hist(title = 'DAYS_LAST_PHONE_CHANGE Histogram');
plt.xlabel('DAYS_LAST_PHONE_CHANGE')
(dataset['AMT_REQ_CREDIT_BUREAU_HOUR']).describe()
dataset['AMT_REQ_CREDIT_BUREAU_HOUR'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_HOUR Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_HOUR')
dataset['AMT_REQ_CREDIT_BUREAU_HOUR_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_HOUR'] > 1
for i in dataset['AMT_REQ_CREDIT_BUREAU_HOUR']:
    if i > 1:
        dataset['AMT_REQ_CREDIT_BUREAU_HOUR'].replace({i: np.nan}, inplace = True)
(dataset['AMT_REQ_CREDIT_BUREAU_DAY']).describe()
dataset['AMT_REQ_CREDIT_BUREAU_DAY'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_DAY Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_DAY')
dataset['AMT_REQ_CREDIT_BUREAU_DAY_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_DAY'] > 2
for i in dataset['AMT_REQ_CREDIT_BUREAU_DAY']:
    if i > 2:
        dataset['AMT_REQ_CREDIT_BUREAU_DAY'].replace({i: np.nan}, inplace = True)
(dataset['AMT_REQ_CREDIT_BUREAU_WEEK']).describe()
dataset['AMT_REQ_CREDIT_BUREAU_WEEK'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_WEEK Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_WEEK')
dataset['AMT_REQ_CREDIT_BUREAU_WEEK_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_WEEK'] > 2
for i in dataset['AMT_REQ_CREDIT_BUREAU_WEEK']:
    if i > 2:
        dataset['AMT_REQ_CREDIT_BUREAU_WEEK'].replace({i: np.nan}, inplace = True)
(dataset['AMT_REQ_CREDIT_BUREAU_MON']).describe()
dataset['AMT_REQ_CREDIT_BUREAU_MON'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_MON Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_MON')
dataset['AMT_REQ_CREDIT_BUREAU_MON_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_MON'] > 5
for i in dataset['AMT_REQ_CREDIT_BUREAU_MON']:
    if i > 5:
        dataset['AMT_REQ_CREDIT_BUREAU_MON'].replace({i: np.nan}, inplace = True)
(dataset['AMT_REQ_CREDIT_BUREAU_QRT']).describe()
dataset['AMT_REQ_CREDIT_BUREAU_QRT'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_QRT Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_QRT')
dataset['AMT_REQ_CREDIT_BUREAU_QRT_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_QRT'] > 5
for i in dataset['AMT_REQ_CREDIT_BUREAU_QRT']:
    if i > 5:
        dataset['AMT_REQ_CREDIT_BUREAU_QRT'].replace({i: np.nan}, inplace = True)
(dataset['AMT_REQ_CREDIT_BUREAU_YEAR']).describe()
dataset['AMT_REQ_CREDIT_BUREAU_YEAR'].plot.hist(title = 'AMT_REQ_CREDIT_BUREAU_YEAR Histogram');
plt.xlabel('AMT_REQ_CREDIT_BUREAU_YEAR')
dataset['AMT_REQ_CREDIT_BUREAU_YEAR_outlier'] = dataset['AMT_REQ_CREDIT_BUREAU_YEAR'] > 10
for i in dataset['AMT_REQ_CREDIT_BUREAU_YEAR']:
    if i > 10:
        dataset['AMT_REQ_CREDIT_BUREAU_YEAR'].replace({i: np.nan}, inplace = True)
g = sns.heatmap(app_train[['TARGET','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# create new variables
dataset['EMPLOY_AGE'] = dataset['DAYS_EMPLOYED'] / dataset['DAYS_BIRTH']
dataset['INCOME_AGE'] = dataset['AMT_INCOME_TOTAL'] / dataset['DAYS_BIRTH']
dataset['CREDIT_AGE'] = dataset['AMT_CREDIT'] / dataset['DAYS_BIRTH']
dataset['CREDIT_INCOME'] = dataset['AMT_CREDIT'] / dataset['AMT_INCOME_TOTAL']
dataset['ANNUITY_INCOME'] = dataset['AMT_ANNUITY'] / dataset['AMT_INCOME_TOTAL']
dataset['ANNUITY_CREDIT'] = dataset['AMT_ANNUITY'] / dataset['AMT_CREDIT']
# let's look at the correlations of the new variables we created along with TARGET
g = sns.heatmap(dataset[['TARGET','EMPLOY_AGE','INCOME_AGE','CREDIT_AGE','CREDIT_INCOME','ANNUITY_INCOME','ANNUITY_CREDIT']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# EMPLOY_AGE seems to be most correlated with TARGET of our new variables
# we can plot EMPLOY_AGE relative to TARGET using KDE
plt.figure(figsize = (8, 6))
sns.kdeplot(dataset.loc[dataset['TARGET'] == 0, 'EMPLOY_AGE'], label = 'TARGET == 0')
sns.kdeplot(dataset.loc[dataset['TARGET'] == 1, 'EMPLOY_AGE'], label = 'TARGET == 1')
plt.xlabel('EMPLOY_AGE'); plt.ylabel('Density'); plt.title('KDE of EMPLOY_AGE');
# the first file we will investigate is bureau
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()
print('bureau data shape: ', bureau.shape)
bureau.describe()
# the first item to look at is how many records are in this for each applicant
BUREAU_count = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'bureau_count'})
(BUREAU_count['bureau_count']).describe()
dataset = dataset.merge(BUREAU_count, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_count'])
print('%.4f' % corr)
# review the data, and divide by -365.25 to turn this into positive years
(bureau['DAYS_CREDIT'] / -365.25).describe()
(bureau['DAYS_CREDIT'] / -365.25).plot.hist(title = 'DAYS_CREDIT Histogram');
plt.xlabel('DAYS_CREDIT')
DAYS_CREDIT_max = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_CREDIT'].max().rename(columns = {'DAYS_CREDIT': 'bureau_DAYS_CREDIT_max'})
DAYS_CREDIT_max.head()
# merge with the dataset
dataset = dataset.merge(DAYS_CREDIT_max, on = 'SK_ID_CURR', how = 'left')
# what is the correlation of our new variable with TARGET
corr = dataset['TARGET'].corr(dataset['bureau_DAYS_CREDIT_max'])
print('%.4f' % corr)
# evaluate the new variable with a KDE plot
plt.figure(figsize = (8, 6))
sns.kdeplot(dataset.loc[dataset['TARGET'] == 0, 'bureau_DAYS_CREDIT_max'], label = 'TARGET == 0')
sns.kdeplot(dataset.loc[dataset['TARGET'] == 1, 'bureau_DAYS_CREDIT_max'], label = 'TARGET == 1')
plt.xlabel('bureau_DAYS_CREDIT_max'); plt.ylabel('Density'); plt.title('KDE of bureau_DAYS_CREDIT_max');
(bureau['CREDIT_DAY_OVERDUE']).describe([.25, .5, .75, .9, .99, .999])
# this looks like virtually all are zero, but there are some outliers
bureau['CREDIT_DAY_OVERDUE'].plot.hist(title = 'CREDIT_DAY_OVERDUE Histogram');
plt.xlabel('CREDIT_DAY_OVERDUE')
# let's take the max of this variable
CREDIT_DAY_OVERDUE_max = bureau.groupby('SK_ID_CURR', as_index=False)['CREDIT_DAY_OVERDUE'].max().rename(columns = {'CREDIT_DAY_OVERDUE': 'bureau_CREDIT_DAY_OVERDUE_max'})
(CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max']).describe([.25, .5, .75, .9, .99, .999])
# most of the data in this column is zero
# how many non-zero items exist?
CREDIT_DAY_OVERDUE_max[CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max'] > 0].count()
# let's turn this into a flag, since 99% of the data is zero
CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max_flag'] = CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max'].where(CREDIT_DAY_OVERDUE_max['bureau_CREDIT_DAY_OVERDUE_max']==0,other=1)
# drop the max variable and merge in the flag
CREDIT_DAY_OVERDUE_max = CREDIT_DAY_OVERDUE_max.drop('bureau_CREDIT_DAY_OVERDUE_max', axis=1)
dataset = dataset.merge(CREDIT_DAY_OVERDUE_max, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_CREDIT_DAY_OVERDUE_max_flag'])
print('%.4f' % corr)
(bureau['DAYS_CREDIT_ENDDATE']).describe()
bureau['DAYS_CREDIT_ENDDATE'].plot.hist(title = 'DAYS_CREDIT_ENDDATE Histogram');
plt.xlabel('DAYS_CREDIT_ENDDATE')
# let's take the max of this variable
DAYS_CREDIT_ENDDATE_max = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_CREDIT_ENDDATE'].max().rename(columns = {'DAYS_CREDIT_ENDDATE': 'bureau_DAYS_CREDIT_ENDDATE_max'})
(DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max']).describe()
# it appears that we have a few outliers around -41875
DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max_outlier'] = DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max'] < -10000
for i in DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max']:
    if i < -10000:
        DAYS_CREDIT_ENDDATE_max['bureau_DAYS_CREDIT_ENDDATE_max'].replace({i: np.nan}, inplace = True)
# merge both our max variable and outlier flag into the dataset
dataset = dataset.merge(DAYS_CREDIT_ENDDATE_max, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_DAYS_CREDIT_ENDDATE_max'])
print('%.4f' % corr)
(bureau['DAYS_ENDDATE_FACT'] / -365.25).describe()
(bureau['DAYS_ENDDATE_FACT'] / -365.25).plot.hist(title = 'DAYS_ENDDATE_FACT Histogram');
plt.xlabel('DAYS_ENDDATE_FACT')
# let's take the average of this variable
DAYS_ENDDATE_FACT_mean = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_ENDDATE_FACT'].mean().rename(columns = {'DAYS_ENDDATE_FACT': 'bureau_DAYS_ENDDATE_FACT_mean'})
(DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean']).describe()
(DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean']).plot.hist(title = 'bureau_DAYS_ENDDATE_FACT_mean Histogram');
plt.xlabel('bureau_DAYS_ENDDATE_FACT_mean')
# it appears that we have a few outliers around -8000 days that we can handle
DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean_outlier'] = DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean'] < -4000
for i in DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean']:
    if i < -4000:
        DAYS_ENDDATE_FACT_mean['bureau_DAYS_ENDDATE_FACT_mean'].replace({i: np.nan}, inplace = True)
# merge both our mean variable and outlier flag into the dataset
dataset = dataset.merge(DAYS_ENDDATE_FACT_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_DAYS_ENDDATE_FACT_mean'])
print('%.4f' % corr)
# evaluate the new variable with a KDE plot
plt.figure(figsize = (8, 6))
sns.kdeplot(dataset.loc[dataset['TARGET'] == 0, 'bureau_DAYS_ENDDATE_FACT_mean'], label = 'TARGET == 0')
sns.kdeplot(dataset.loc[dataset['TARGET'] == 1, 'bureau_DAYS_ENDDATE_FACT_mean'], label = 'TARGET == 1')
plt.xlabel('bureau_DAYS_ENDDATE_FACT_mean'); plt.ylabel('Density'); plt.title('KDE of bureau_DAYS_ENDDATE_FACT_mean');
(bureau['AMT_CREDIT_MAX_OVERDUE']).describe()
# let's take the max of this variable
AMT_CREDIT_MAX_OVERDUE_max = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_MAX_OVERDUE'].max().rename(columns = {'AMT_CREDIT_MAX_OVERDUE': 'bureau_AMT_CREDIT_MAX_OVERDUE_max'})
(AMT_CREDIT_MAX_OVERDUE_max['bureau_AMT_CREDIT_MAX_OVERDUE_max']).describe()
# I'm also curious on the average of this variable
AMT_CREDIT_MAX_OVERDUE_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_MAX_OVERDUE'].mean().rename(columns = {'AMT_CREDIT_MAX_OVERDUE': 'bureau_AMT_CREDIT_MAX_OVERDUE_mean'})
(AMT_CREDIT_MAX_OVERDUE_mean['bureau_AMT_CREDIT_MAX_OVERDUE_mean']).describe()
# I'm not sure which of these two variables may work better in this case, so let's bring them both into the dataset for now
dataset = dataset.merge(AMT_CREDIT_MAX_OVERDUE_max, on = 'SK_ID_CURR', how = 'left')
dataset = dataset.merge(AMT_CREDIT_MAX_OVERDUE_mean, on = 'SK_ID_CURR', how = 'left')
corr_max = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_MAX_OVERDUE_max'])
corr_mean = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_MAX_OVERDUE_mean'])
print('correlation for max variable: %.4f' % corr_max)
print('correlation for mean variable: %.4f' % corr_mean)
(bureau['CNT_CREDIT_PROLONG']).describe()
# since these are counts, let's sum this variable
CNT_CREDIT_PROLONG_sum = bureau.groupby('SK_ID_CURR', as_index=False)['CNT_CREDIT_PROLONG'].sum().rename(columns = {'CNT_CREDIT_PROLONG': 'bureau_CNT_CREDIT_PROLONG_sum'})
(CNT_CREDIT_PROLONG_sum['bureau_CNT_CREDIT_PROLONG_sum']).describe()
# merge into our dataset
dataset = dataset.merge(CNT_CREDIT_PROLONG_sum, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_CNT_CREDIT_PROLONG_sum'])
print('%.4f' % corr)
(bureau['AMT_CREDIT_SUM']).describe()
# since these are amounts, let's average this variable
AMT_CREDIT_SUM_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM'].mean().rename(columns = {'AMT_CREDIT_SUM': 'bureau_AMT_CREDIT_SUM_mean'})
(AMT_CREDIT_SUM_mean['bureau_AMT_CREDIT_SUM_mean']).describe()
# merge into our dataset
dataset = dataset.merge(AMT_CREDIT_SUM_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_mean'])
print('%.4f' % corr)
(bureau['AMT_CREDIT_SUM_DEBT']).describe()
# since these are amounts, let's average this variable
AMT_CREDIT_SUM_DEBT_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM_DEBT'].mean().rename(columns = {'AMT_CREDIT_SUM_DEBT': 'bureau_AMT_CREDIT_SUM_DEBT_mean'})
(AMT_CREDIT_SUM_DEBT_mean['bureau_AMT_CREDIT_SUM_DEBT_mean']).describe()
# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_CREDIT_SUM_DEBT_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_DEBT_mean'])
print('%.4f' % corr)
(bureau['AMT_CREDIT_SUM_LIMIT']).describe()
# since these are amounts, let's average this variable
AMT_CREDIT_SUM_LIMIT_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM_LIMIT'].mean().rename(columns = {'AMT_CREDIT_SUM_LIMIT': 'bureau_AMT_CREDIT_SUM_LIMIT_mean'})
(AMT_CREDIT_SUM_LIMIT_mean['bureau_AMT_CREDIT_SUM_LIMIT_mean']).describe()
# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_CREDIT_SUM_LIMIT_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_LIMIT_mean'])
print('%.4f' % corr)
(bureau['AMT_CREDIT_SUM_OVERDUE']).describe()
# since these are amounts, let's average this variable
AMT_CREDIT_SUM_OVERDUE_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_CREDIT_SUM_OVERDUE'].mean().rename(columns = {'AMT_CREDIT_SUM_OVERDUE': 'bureau_AMT_CREDIT_SUM_OVERDUE_mean'})
(AMT_CREDIT_SUM_OVERDUE_mean['bureau_AMT_CREDIT_SUM_OVERDUE_mean']).describe()
# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_CREDIT_SUM_OVERDUE_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_CREDIT_SUM_OVERDUE_mean'])
print('%.4f' % corr)
# divide by -365.25 to turn this into positive years
(bureau['DAYS_CREDIT_UPDATE'] / -365.25).describe()
# since this is a days variable, let's use max
DAYS_CREDIT_UPDATE_max = bureau.groupby('SK_ID_CURR', as_index=False)['DAYS_CREDIT_UPDATE'].max().rename(columns = {'DAYS_CREDIT_UPDATE': 'bureau_DAYS_CREDIT_UPDATE_max'})
(DAYS_CREDIT_UPDATE_max['bureau_DAYS_CREDIT_UPDATE_max']).describe()
# merge into our dataset and look at the correlation
dataset = dataset.merge(DAYS_CREDIT_UPDATE_max, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_DAYS_CREDIT_UPDATE_max'])
print('%.4f' % corr)
(bureau['AMT_ANNUITY']).describe()
# since these are amounts, let's average this variable
AMT_ANNUITY_mean = bureau.groupby('SK_ID_CURR', as_index=False)['AMT_ANNUITY'].mean().rename(columns = {'AMT_ANNUITY': 'bureau_AMT_ANNUITY_mean'})
(AMT_ANNUITY_mean['bureau_AMT_ANNUITY_mean']).describe()
# merge into our dataset and look at the correlation
dataset = dataset.merge(AMT_ANNUITY_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_AMT_ANNUITY_mean'])
print('%.4f' % corr)
bureau.describe(include=[np.object])
# let's use one-hot encoding on these variables
bureau_cats = pd.get_dummies(bureau.select_dtypes('object'))
bureau_cats['SK_ID_CURR'] = bureau['SK_ID_CURR']
bureau_cats.head()
bureau_cats_grouped = bureau_cats.groupby('SK_ID_CURR').agg('sum')
bureau_cats_grouped.head()
#merge into our dataset
dataset = dataset.merge(bureau_cats_grouped, on = 'SK_ID_CURR', right_index = True, how = 'left')
dataset.head()
# the next file we will investigate is bureau_balance
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
bureau_balance.head()
# this appears to be the number of months of balance relative to the application date
# let's start with the count
MONTHS_BALANCE_count = bureau_balance.groupby('SK_ID_BUREAU', as_index=False)['MONTHS_BALANCE'].count().rename(columns = {'MONTHS_BALANCE': 'bureau_bal_MONTHS_BALANCE_count'})
(MONTHS_BALANCE_count['bureau_bal_MONTHS_BALANCE_count']).describe()
# let's also look at the mean
MONTHS_BALANCE_mean = bureau_balance.groupby('SK_ID_BUREAU', as_index=False)['MONTHS_BALANCE'].mean().rename(columns = {'MONTHS_BALANCE': 'bureau_bal_MONTHS_BALANCE_mean'})
(MONTHS_BALANCE_mean['bureau_bal_MONTHS_BALANCE_mean']).describe()
MONTHS_BAL = MONTHS_BALANCE_mean.merge(MONTHS_BALANCE_count, on = 'SK_ID_BUREAU', right_index = True, how = 'inner')
MONTHS_BAL.head()
# now let's get our categoricals
bureau_bal_cats = pd.get_dummies(bureau_balance.select_dtypes('object'))
bureau_bal_cats['SK_ID_BUREAU'] = bureau_balance['SK_ID_BUREAU']
bureau_bal_cats.head()
bureau_bal_cats_grouped = bureau_bal_cats.groupby('SK_ID_BUREAU').agg('sum')
bureau_bal_cats_grouped.head()
# now let's merge the MONTHS_BAL with our categoricals by SK_ID_BUREAU, then merge with bureau to add in SK_ID_CURR
bureau_bal_merged = MONTHS_BAL.merge(bureau_bal_cats_grouped, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
bureau_bal_merged = bureau_bal_merged.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')
bureau_bal_merged.head()
bureau_bal_MONTHS_BALANCE_mean_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['bureau_bal_MONTHS_BALANCE_mean'].mean().rename(columns = {'bureau_bal_MONTHS_BALANCE_mean': 'bureau_bal_MONTHS_BALANCE_mean_mean'})
dataset = dataset.merge(bureau_bal_MONTHS_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_MONTHS_BALANCE_mean_mean'])
print('%.4f' % corr)
bureau_bal_MONTHS_BALANCE_count_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['bureau_bal_MONTHS_BALANCE_count'].mean().rename(columns = {'bureau_bal_MONTHS_BALANCE_count': 'bureau_bal_MONTHS_BALANCE_count_mean'})
dataset = dataset.merge(bureau_bal_MONTHS_BALANCE_count_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_MONTHS_BALANCE_count_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_0_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_0'].mean().rename(columns = {'STATUS_0': 'bureau_bal_STATUS_0_mean'})
dataset = dataset.merge(bureau_bal_STATUS_0_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_0_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_1_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_1'].mean().rename(columns = {'STATUS_1': 'bureau_bal_STATUS_1_mean'})
dataset = dataset.merge(bureau_bal_STATUS_1_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_1_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_2_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_2'].mean().rename(columns = {'STATUS_2': 'bureau_bal_STATUS_2_mean'})
dataset = dataset.merge(bureau_bal_STATUS_2_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_2_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_3_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_3'].mean().rename(columns = {'STATUS_3': 'bureau_bal_STATUS_3_mean'})
dataset = dataset.merge(bureau_bal_STATUS_3_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_3_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_4_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_4'].mean().rename(columns = {'STATUS_4': 'bureau_bal_STATUS_4_mean'})
dataset = dataset.merge(bureau_bal_STATUS_4_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_4_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_5_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_5'].mean().rename(columns = {'STATUS_5': 'bureau_bal_STATUS_5_mean'})
dataset = dataset.merge(bureau_bal_STATUS_5_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_5_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_C_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_C'].mean().rename(columns = {'STATUS_C': 'bureau_bal_STATUS_C_mean'})
dataset = dataset.merge(bureau_bal_STATUS_C_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_C_mean'])
print('%.4f' % corr)
bureau_bal_STATUS_X_mean = bureau_bal_merged.groupby('SK_ID_CURR', as_index=False)['STATUS_X'].mean().rename(columns = {'STATUS_X': 'bureau_bal_STATUS_X_mean'})
dataset = dataset.merge(bureau_bal_STATUS_X_mean, on = 'SK_ID_CURR', how = 'left')
corr = dataset['TARGET'].corr(dataset['bureau_bal_STATUS_X_mean'])
print('%.4f' % corr)
dataset.head()
# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del bureau, BUREAU_count, DAYS_CREDIT_max, CREDIT_DAY_OVERDUE_max, DAYS_CREDIT_ENDDATE_max, DAYS_ENDDATE_FACT_mean, AMT_CREDIT_MAX_OVERDUE_max, 
AMT_CREDIT_MAX_OVERDUE_mean, CNT_CREDIT_PROLONG_sum, AMT_CREDIT_SUM_mean, AMT_CREDIT_SUM_DEBT_mean, AMT_CREDIT_SUM_LIMIT_mean, AMT_CREDIT_SUM_OVERDUE_mean, 
DAYS_CREDIT_UPDATE_max, AMT_ANNUITY_mean, bureau_cats, bureau_cats_grouped, bureau_balance, MONTHS_BALANCE_count, MONTHS_BALANCE_mean, MONTHS_BAL, bureau_bal_cats, 
bureau_bal_cats_grouped, bureau_bal_merged, bureau_bal_MONTHS_BALANCE_mean_mean, bureau_bal_MONTHS_BALANCE_count_mean, bureau_bal_STATUS_0_mean, 
bureau_bal_STATUS_1_mean, bureau_bal_STATUS_2_mean, bureau_bal_STATUS_3_mean, bureau_bal_STATUS_4_mean, bureau_bal_STATUS_5_mean, bureau_bal_STATUS_C_mean, 
bureau_bal_STATUS_X_mean
gc.collect()
# the next file we will investigate is credit_card_balance
credit = pd.read_csv('../input/credit_card_balance.csv')
credit.head()
credit_stats_by_prev = credit[['SK_ID_PREV', 'SK_ID_CURR']]
credit_MONTHS_BALANCE_count = credit.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].count().rename(columns = {'MONTHS_BALANCE': 'credit_MONTHS_BALANCE_count'})
credit_MONTHS_BALANCE_mean = credit.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].mean().rename(columns = {'MONTHS_BALANCE': 'credit_MONTHS_BALANCE_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_MONTHS_BALANCE_count, on = 'SK_ID_PREV', how = 'left')
credit_stats_by_prev = credit_stats_by_prev.merge(credit_MONTHS_BALANCE_mean, on = 'SK_ID_PREV', how = 'left')
credit_stats_by_prev.head()
gc.enable()
del credit_MONTHS_BALANCE_count, credit_MONTHS_BALANCE_mean
gc.collect()
credit_AMT_BALANCE_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_BALANCE'].mean().rename(columns = {'AMT_BALANCE': 'credit_AMT_BALANCE_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_BALANCE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_BALANCE_mean
gc.collect()
credit_AMT_CREDIT_LIMIT_ACTUAL_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_CREDIT_LIMIT_ACTUAL'].mean().rename(columns = {'AMT_CREDIT_LIMIT_ACTUAL': 'credit_AMT_CREDIT_LIMIT_ACTUAL_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_CREDIT_LIMIT_ACTUAL_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_CREDIT_LIMIT_ACTUAL_mean
gc.collect()
credit_AMT_DRAWINGS_ATM_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_ATM_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_ATM_CURRENT': 'credit_AMT_DRAWINGS_ATM_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_ATM_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_ATM_CURRENT_mean
gc.collect()
credit_AMT_DRAWINGS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_CURRENT': 'credit_AMT_DRAWINGS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_CURRENT_mean
gc.collect()
credit_AMT_DRAWINGS_OTHER_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_OTHER_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_OTHER_CURRENT': 'credit_AMT_DRAWINGS_OTHER_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_OTHER_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_OTHER_CURRENT_mean
gc.collect()
credit_AMT_DRAWINGS_POS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_DRAWINGS_POS_CURRENT'].mean().rename(columns = {'AMT_DRAWINGS_POS_CURRENT': 'credit_AMT_DRAWINGS_POS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_DRAWINGS_POS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_POS_CURRENT_mean
gc.collect()
credit_AMT_INST_MIN_REGULARITY_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_INST_MIN_REGULARITY'].mean().rename(columns = {'AMT_INST_MIN_REGULARITY': 'credit_AMT_INST_MIN_REGULARITY_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_INST_MIN_REGULARITY_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_INST_MIN_REGULARITY_mean
gc.collect()
credit_AMT_PAYMENT_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_PAYMENT_CURRENT'].mean().rename(columns = {'AMT_PAYMENT_CURRENT': 'credit_AMT_PAYMENT_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_PAYMENT_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_CURRENT_mean
gc.collect()
credit_AMT_PAYMENT_TOTAL_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_PAYMENT_TOTAL_CURRENT'].mean().rename(columns = {'AMT_PAYMENT_TOTAL_CURRENT': 'credit_AMT_PAYMENT_TOTAL_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_PAYMENT_TOTAL_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_TOTAL_CURRENT_mean
gc.collect()
credit_AMT_RECEIVABLE_PRINCIPAL_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_RECEIVABLE_PRINCIPAL'].mean().rename(columns = {'AMT_RECEIVABLE_PRINCIPAL': 'credit_AMT_RECEIVABLE_PRINCIPAL_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_RECEIVABLE_PRINCIPAL_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_RECEIVABLE_PRINCIPAL_mean
gc.collect()
credit_AMT_RECIVABLE_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_RECIVABLE'].mean().rename(columns = {'AMT_RECIVABLE': 'credit_AMT_RECIVABLE_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_RECIVABLE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_RECIVABLE_mean
gc.collect()
credit_AMT_TOTAL_RECEIVABLE_mean = credit.groupby('SK_ID_PREV', as_index=False)['AMT_TOTAL_RECEIVABLE'].mean().rename(columns = {'AMT_TOTAL_RECEIVABLE': 'credit_AMT_TOTAL_RECEIVABLE_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_AMT_TOTAL_RECEIVABLE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_AMT_TOTAL_RECEIVABLE_mean
gc.collect()
credit_CNT_DRAWINGS_ATM_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_ATM_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_ATM_CURRENT': 'credit_CNT_DRAWINGS_ATM_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_ATM_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_ATM_CURRENT_mean
gc.collect()
credit_CNT_DRAWINGS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_CURRENT': 'credit_CNT_DRAWINGS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_CURRENT_mean
gc.collect()
credit_CNT_DRAWINGS_OTHER_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_OTHER_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_OTHER_CURRENT': 'credit_CNT_DRAWINGS_OTHER_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_OTHER_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_OTHER_CURRENT_mean
gc.collect()
credit_CNT_DRAWINGS_POS_CURRENT_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_DRAWINGS_POS_CURRENT'].mean().rename(columns = {'CNT_DRAWINGS_POS_CURRENT': 'credit_CNT_DRAWINGS_POS_CURRENT_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_DRAWINGS_POS_CURRENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_POS_CURRENT_mean
gc.collect()
credit_CNT_INSTALMENT_MATURE_CUM_mean = credit.groupby('SK_ID_PREV', as_index=False)['CNT_INSTALMENT_MATURE_CUM'].mean().rename(columns = {'CNT_INSTALMENT_MATURE_CUM': 'credit_CNT_INSTALMENT_MATURE_CUM_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_CNT_INSTALMENT_MATURE_CUM_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_CNT_INSTALMENT_MATURE_CUM_mean
gc.collect()
credit_SK_DPD_mean = credit.groupby('SK_ID_PREV', as_index=False)['SK_DPD'].mean().rename(columns = {'SK_DPD': 'credit_SK_DPD_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_SK_DPD_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_SK_DPD_mean
gc.collect()
credit_SK_DPD_DEF_mean = credit.groupby('SK_ID_PREV', as_index=False)['SK_DPD_DEF'].mean().rename(columns = {'SK_DPD_DEF': 'credit_SK_DPD_DEF_mean'})
credit_stats_by_prev = credit_stats_by_prev.merge(credit_SK_DPD_DEF_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_SK_DPD_DEF_mean
gc.collect()
# now let's deal with our one categorical variable, NAME_CONTRACT_STATUS
credit_cats = pd.get_dummies(credit.select_dtypes('object'))
credit_cats['SK_ID_PREV'] = credit['SK_ID_PREV']
credit_cats.head()
credit_cats_grouped = credit_cats.groupby('SK_ID_PREV').agg('sum')
credit_cats_grouped.head()
credit_stats_by_prev = credit_stats_by_prev.merge(credit_cats_grouped, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del credit_cats_grouped, credit_cats
gc.collect()
credit_stats_by_prev.head()
credit_MONTHS_BALANCE_count_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_MONTHS_BALANCE_count'].mean().rename(columns = {'credit_MONTHS_BALANCE_count': 'credit_MONTHS_BALANCE_count_mean'})
dataset = dataset.merge(credit_MONTHS_BALANCE_count_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_MONTHS_BALANCE_count_mean
gc.collect()
credit_MONTHS_BALANCE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_MONTHS_BALANCE_mean'].mean().rename(columns = {'credit_MONTHS_BALANCE_mean': 'credit_MONTHS_BALANCE_mean_mean'})
dataset = dataset.merge(credit_MONTHS_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_MONTHS_BALANCE_mean_mean
gc.collect()
credit_AMT_BALANCE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_BALANCE_mean'].mean().rename(columns = {'credit_AMT_BALANCE_mean': 'credit_AMT_BALANCE_mean_mean'})
dataset = dataset.merge(credit_AMT_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_BALANCE_mean_mean
gc.collect()
credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_CREDIT_LIMIT_ACTUAL_mean'].mean().rename(columns = {'credit_AMT_CREDIT_LIMIT_ACTUAL_mean': 'credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean'})
dataset = dataset.merge(credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_CREDIT_LIMIT_ACTUAL_mean_mean
gc.collect()
credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_ATM_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_ATM_CURRENT_mean': 'credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_ATM_CURRENT_mean_mean
gc.collect()
credit_AMT_DRAWINGS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_CURRENT_mean': 'credit_AMT_DRAWINGS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_CURRENT_mean_mean
gc.collect()
credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_OTHER_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_OTHER_CURRENT_mean': 'credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_OTHER_CURRENT_mean_mean
gc.collect()
credit_AMT_DRAWINGS_POS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_DRAWINGS_POS_CURRENT_mean'].mean().rename(columns = {'credit_AMT_DRAWINGS_POS_CURRENT_mean': 'credit_AMT_DRAWINGS_POS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_DRAWINGS_POS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_DRAWINGS_POS_CURRENT_mean_mean
gc.collect()
credit_AMT_INST_MIN_REGULARITY_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_INST_MIN_REGULARITY_mean'].mean().rename(columns = {'credit_AMT_INST_MIN_REGULARITY_mean': 'credit_AMT_INST_MIN_REGULARITY_mean_mean'})
dataset = dataset.merge(credit_AMT_INST_MIN_REGULARITY_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_INST_MIN_REGULARITY_mean_mean
gc.collect()
credit_AMT_PAYMENT_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_PAYMENT_CURRENT_mean'].mean().rename(columns = {'credit_AMT_PAYMENT_CURRENT_mean': 'credit_AMT_PAYMENT_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_PAYMENT_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_CURRENT_mean_mean
gc.collect()
credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_PAYMENT_TOTAL_CURRENT_mean'].mean().rename(columns = {'credit_AMT_PAYMENT_TOTAL_CURRENT_mean': 'credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean'})
dataset = dataset.merge(credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_PAYMENT_TOTAL_CURRENT_mean_mean
gc.collect()
credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_RECEIVABLE_PRINCIPAL_mean'].mean().rename(columns = {'credit_AMT_RECEIVABLE_PRINCIPAL_mean': 'credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean'})
dataset = dataset.merge(credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_RECEIVABLE_PRINCIPAL_mean_mean
gc.collect()
credit_AMT_RECIVABLE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_RECIVABLE_mean'].mean().rename(columns = {'credit_AMT_RECIVABLE_mean': 'credit_AMT_RECIVABLE_mean_mean'})
dataset = dataset.merge(credit_AMT_RECIVABLE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_RECIVABLE_mean_mean
gc.collect()
credit_AMT_TOTAL_RECEIVABLE_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_AMT_TOTAL_RECEIVABLE_mean'].mean().rename(columns = {'credit_AMT_TOTAL_RECEIVABLE_mean': 'credit_AMT_TOTAL_RECEIVABLE_mean_mean'})
dataset = dataset.merge(credit_AMT_TOTAL_RECEIVABLE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_AMT_TOTAL_RECEIVABLE_mean_mean
gc.collect()
credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_ATM_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_ATM_CURRENT_mean': 'credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_ATM_CURRENT_mean_mean
gc.collect()
credit_CNT_DRAWINGS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_CURRENT_mean': 'credit_CNT_DRAWINGS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_CURRENT_mean_mean
gc.collect()
credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_OTHER_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_OTHER_CURRENT_mean': 'credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_OTHER_CURRENT_mean_mean
gc.collect()
credit_CNT_DRAWINGS_POS_CURRENT_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_DRAWINGS_POS_CURRENT_mean'].mean().rename(columns = {'credit_CNT_DRAWINGS_POS_CURRENT_mean': 'credit_CNT_DRAWINGS_POS_CURRENT_mean_mean'})
dataset = dataset.merge(credit_CNT_DRAWINGS_POS_CURRENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_DRAWINGS_POS_CURRENT_mean_mean
gc.collect()
credit_CNT_INSTALMENT_MATURE_CUM_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_CNT_INSTALMENT_MATURE_CUM_mean'].mean().rename(columns = {'credit_CNT_INSTALMENT_MATURE_CUM_mean': 'credit_CNT_INSTALMENT_MATURE_CUM_mean_mean'})
dataset = dataset.merge(credit_CNT_INSTALMENT_MATURE_CUM_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_CNT_INSTALMENT_MATURE_CUM_mean_mean
gc.collect()
credit_SK_DPD_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_SK_DPD_mean'].mean().rename(columns = {'credit_SK_DPD_mean': 'credit_SK_DPD_mean_mean'})
dataset = dataset.merge(credit_SK_DPD_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_SK_DPD_mean_mean
gc.collect()
credit_SK_DPD_DEF_mean_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['credit_SK_DPD_DEF_mean'].mean().rename(columns = {'credit_SK_DPD_DEF_mean': 'credit_SK_DPD_DEF_mean_mean'})
dataset = dataset.merge(credit_SK_DPD_DEF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_SK_DPD_DEF_mean_mean
gc.collect()
credit_NAME_CONTRACT_STATUS_Active_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Active'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Active': 'credit_NAME_CONTRACT_STATUS_Active_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Active_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Active_mean
gc.collect()
credit_NAME_CONTRACT_STATUS_Approved_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Approved'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Approved': 'credit_NAME_CONTRACT_STATUS_Approved_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Approved_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Approved_mean
gc.collect()
credit_NAME_CONTRACT_STATUS_Completed_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Completed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Completed': 'credit_NAME_CONTRACT_STATUS_Completed_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Completed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Completed_mean
gc.collect()
credit_NAME_CONTRACT_STATUS_Demand_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Demand'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Demand': 'credit_NAME_CONTRACT_STATUS_Demand_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Demand_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Demand_mean
gc.collect()
credit_NAME_CONTRACT_STATUS_Refused_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Refused'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Refused': 'credit_NAME_CONTRACT_STATUS_Refused_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Refused_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Refused_mean
gc.collect()
credit_NAME_CONTRACT_STATUS_Sent_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Sent proposal'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Sent proposal': 'credit_NAME_CONTRACT_STATUS_Sent_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Sent_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Sent_mean
gc.collect()
credit_NAME_CONTRACT_STATUS_Signed_mean = credit_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Signed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Signed': 'credit_NAME_CONTRACT_STATUS_Signed_mean'})
dataset = dataset.merge(credit_NAME_CONTRACT_STATUS_Signed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del credit_NAME_CONTRACT_STATUS_Signed_mean
gc.collect()
print('dataset data shape: ', dataset.shape)
dataset.head()
# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del credit, credit_stats_by_prev
gc.collect()
# the next file we will investigate is installments_payments
install = pd.read_csv('../input/installments_payments.csv')
install.head()
# create the additional variables
install['DAYS_DIFF'] = install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']
install['AMT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
install.head()
install_stats_by_prev = install[['SK_ID_PREV', 'SK_ID_CURR']]
install_NUM_INSTALMENT_VERSION_count = install.groupby('SK_ID_PREV', as_index=False)['NUM_INSTALMENT_VERSION'].count().rename(columns = {'NUM_INSTALMENT_VERSION': 'install_NUM_INSTALMENT_VERSION_count'})
install_NUM_INSTALMENT_VERSION_max = install.groupby('SK_ID_PREV', as_index=False)['NUM_INSTALMENT_VERSION'].max().rename(columns = {'NUM_INSTALMENT_VERSION': 'install_NUM_INSTALMENT_VERSION_max'})
install_stats_by_prev = install_stats_by_prev.merge(install_NUM_INSTALMENT_VERSION_count, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_NUM_INSTALMENT_VERSION_max, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_NUM_INSTALMENT_VERSION_count, install_NUM_INSTALMENT_VERSION_max
gc.collect()
install_DAYS_INSTALMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['DAYS_INSTALMENT'].mean().rename(columns = {'DAYS_INSTALMENT': 'install_DAYS_INSTALMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_INSTALMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_DAYS_INSTALMENT_mean
gc.collect()
install_DAYS_ENTRY_PAYMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['DAYS_ENTRY_PAYMENT'].mean().rename(columns = {'DAYS_ENTRY_PAYMENT': 'install_DAYS_ENTRY_PAYMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_ENTRY_PAYMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_DAYS_ENTRY_PAYMENT_mean
gc.collect()
install_AMT_INSTALMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['AMT_INSTALMENT'].mean().rename(columns = {'AMT_INSTALMENT': 'install_AMT_INSTALMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_INSTALMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_AMT_INSTALMENT_mean
gc.collect()
install_AMT_PAYMENT_mean = install.groupby('SK_ID_PREV', as_index=False)['AMT_PAYMENT'].mean().rename(columns = {'AMT_PAYMENT': 'install_AMT_PAYMENT_mean'})
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_PAYMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_AMT_PAYMENT_mean
gc.collect()
# capture the mean, max, and min for DAYS_DIFF
install_DAYS_DIFF_mean = install.groupby('SK_ID_PREV', as_index=False)['DAYS_DIFF'].mean().rename(columns = {'DAYS_DIFF': 'install_DAYS_DIFF_mean'})
install_DAYS_DIFF_max = install.groupby('SK_ID_PREV', as_index=False)['DAYS_DIFF'].max().rename(columns = {'DAYS_DIFF': 'install_DAYS_DIFF_max'})
install_DAYS_DIFF_min = install.groupby('SK_ID_PREV', as_index=False)['DAYS_DIFF'].min().rename(columns = {'DAYS_DIFF': 'install_DAYS_DIFF_min'})
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_DIFF_mean, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_DIFF_max, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_DAYS_DIFF_min, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_DAYS_DIFF_mean, install_DAYS_DIFF_max, install_DAYS_DIFF_min
gc.collect()
# capture the mean, max, and min for AMT_DIFF
install_AMT_DIFF_mean = install.groupby('SK_ID_PREV', as_index=False)['AMT_DIFF'].mean().rename(columns = {'AMT_DIFF': 'install_AMT_DIFF_mean'})
install_AMT_DIFF_max = install.groupby('SK_ID_PREV', as_index=False)['AMT_DIFF'].max().rename(columns = {'AMT_DIFF': 'install_AMT_DIFF_max'})
install_AMT_DIFF_min = install.groupby('SK_ID_PREV', as_index=False)['AMT_DIFF'].min().rename(columns = {'AMT_DIFF': 'install_AMT_DIFF_min'})
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_DIFF_mean, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_DIFF_max, on = 'SK_ID_PREV', how = 'left')
install_stats_by_prev = install_stats_by_prev.merge(install_AMT_DIFF_min, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del install_AMT_DIFF_mean, install_AMT_DIFF_max, install_AMT_DIFF_min
gc.collect()
install_stats_by_prev.head()
install_NUM_INSTALMENT_VERSION_count_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_NUM_INSTALMENT_VERSION_count'].mean().rename(columns = {'install_NUM_INSTALMENT_VERSION_count': 'install_NUM_INSTALMENT_VERSION_count_mean'})
dataset = dataset.merge(install_NUM_INSTALMENT_VERSION_count_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_NUM_INSTALMENT_VERSION_count_mean
gc.collect()
install_NUM_INSTALMENT_VERSION_max_max = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_NUM_INSTALMENT_VERSION_max'].max().rename(columns = {'install_NUM_INSTALMENT_VERSION_max': 'install_NUM_INSTALMENT_VERSION_max_max'})
dataset = dataset.merge(install_NUM_INSTALMENT_VERSION_max_max, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_NUM_INSTALMENT_VERSION_max_max
gc.collect()
install_DAYS_INSTALMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_INSTALMENT_mean'].mean().rename(columns = {'install_DAYS_INSTALMENT_mean': 'install_DAYS_INSTALMENT_mean_mean'})
dataset = dataset.merge(install_DAYS_INSTALMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_INSTALMENT_mean_mean
gc.collect()
install_DAYS_ENTRY_PAYMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_ENTRY_PAYMENT_mean'].mean().rename(columns = {'install_DAYS_ENTRY_PAYMENT_mean': 'install_DAYS_ENTRY_PAYMENT_mean_mean'})
dataset = dataset.merge(install_DAYS_ENTRY_PAYMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_ENTRY_PAYMENT_mean_mean
gc.collect()
install_AMT_INSTALMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_INSTALMENT_mean'].mean().rename(columns = {'install_AMT_INSTALMENT_mean': 'install_AMT_INSTALMENT_mean_mean'})
dataset = dataset.merge(install_AMT_INSTALMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_INSTALMENT_mean_mean
gc.collect()
install_AMT_PAYMENT_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_PAYMENT_mean'].mean().rename(columns = {'install_AMT_PAYMENT_mean': 'install_AMT_PAYMENT_mean_mean'})
dataset = dataset.merge(install_AMT_PAYMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_PAYMENT_mean_mean
gc.collect()
install_DAYS_DIFF_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_DIFF_mean'].mean().rename(columns = {'install_DAYS_DIFF_mean': 'install_DAYS_DIFF_mean_mean'})
dataset = dataset.merge(install_DAYS_DIFF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_DIFF_mean_mean
gc.collect()
install_DAYS_DIFF_max_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_DIFF_max'].mean().rename(columns = {'install_DAYS_DIFF_max': 'install_DAYS_DIFF_max_mean'})
dataset = dataset.merge(install_DAYS_DIFF_max_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_DIFF_max_mean
gc.collect()
install_DAYS_DIFF_min_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_DAYS_DIFF_min'].mean().rename(columns = {'install_DAYS_DIFF_min': 'install_DAYS_DIFF_min_mean'})
dataset = dataset.merge(install_DAYS_DIFF_min_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_DAYS_DIFF_min_mean
gc.collect()
install_AMT_DIFF_mean_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_DIFF_mean'].mean().rename(columns = {'install_AMT_DIFF_mean': 'install_AMT_DIFF_mean_mean'})
dataset = dataset.merge(install_AMT_DIFF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_DIFF_mean_mean
gc.collect()
install_AMT_DIFF_max_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_DIFF_max'].mean().rename(columns = {'install_AMT_DIFF_max': 'install_AMT_DIFF_max_mean'})
dataset = dataset.merge(install_AMT_DIFF_max_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_DIFF_max_mean
gc.collect()
install_AMT_DIFF_min_mean = install_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['install_AMT_DIFF_min'].mean().rename(columns = {'install_AMT_DIFF_min': 'install_AMT_DIFF_min_mean'})
dataset = dataset.merge(install_AMT_DIFF_min_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del install_AMT_DIFF_min_mean
gc.collect()
print('dataset data shape: ', dataset.shape)
dataset.head()
# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del install, install_stats_by_prev
gc.collect()
# the next file we will investigate is POS_CASH_balance
cash = pd.read_csv('../input/POS_CASH_balance.csv')
cash.head()
cash_stats_by_prev = cash[['SK_ID_PREV', 'SK_ID_CURR']]
cash_MONTHS_BALANCE_count = cash.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].count().rename(columns = {'MONTHS_BALANCE': 'cash_MONTHS_BALANCE_count'})
cash_MONTHS_BALANCE_mean = cash.groupby('SK_ID_PREV', as_index=False)['MONTHS_BALANCE'].mean().rename(columns = {'MONTHS_BALANCE': 'cash_MONTHS_BALANCE_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_MONTHS_BALANCE_count, on = 'SK_ID_PREV', how = 'left')
cash_stats_by_prev = cash_stats_by_prev.merge(cash_MONTHS_BALANCE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_MONTHS_BALANCE_count, cash_MONTHS_BALANCE_mean
gc.collect()
cash_CNT_INSTALMENT_mean = cash.groupby('SK_ID_PREV', as_index=False)['CNT_INSTALMENT'].mean().rename(columns = {'CNT_INSTALMENT': 'cash_CNT_INSTALMENT_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_CNT_INSTALMENT_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_mean
gc.collect()
cash_CNT_INSTALMENT_FUTURE_mean = cash.groupby('SK_ID_PREV', as_index=False)['CNT_INSTALMENT_FUTURE'].mean().rename(columns = {'CNT_INSTALMENT_FUTURE': 'cash_CNT_INSTALMENT_FUTURE_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_CNT_INSTALMENT_FUTURE_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_FUTURE_mean
gc.collect()
cash_SK_DPD_mean = cash.groupby('SK_ID_PREV', as_index=False)['SK_DPD'].mean().rename(columns = {'SK_DPD': 'cash_SK_DPD_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_SK_DPD_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_SK_DPD_mean
gc.collect()
cash_SK_DPD_DEF_mean = cash.groupby('SK_ID_PREV', as_index=False)['SK_DPD_DEF'].mean().rename(columns = {'SK_DPD_DEF': 'cash_SK_DPD_DEF_mean'})
cash_stats_by_prev = cash_stats_by_prev.merge(cash_SK_DPD_DEF_mean, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_SK_DPD_DEF_mean
gc.collect()
# now let's deal with our one categorical variable, NAME_CONTRACT_STATUS, in this cash file
cash_cats = pd.get_dummies(cash.select_dtypes('object'))
cash_cats['SK_ID_PREV'] = cash['SK_ID_PREV']
cash_cats.head()
cash_cats_grouped = cash_cats.groupby('SK_ID_PREV').agg('sum')
cash_cats_grouped.head()
cash_stats_by_prev = cash_stats_by_prev.merge(cash_cats_grouped, on = 'SK_ID_PREV', how = 'left')
gc.enable()
del cash_cats_grouped, cash_cats
gc.collect()
cash_stats_by_prev.head()
cash_MONTHS_BALANCE_count_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_MONTHS_BALANCE_count'].mean().rename(columns = {'cash_MONTHS_BALANCE_count': 'cash_MONTHS_BALANCE_count_mean'})
dataset = dataset.merge(cash_MONTHS_BALANCE_count_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_MONTHS_BALANCE_count_mean
gc.collect()
cash_MONTHS_BALANCE_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_MONTHS_BALANCE_mean'].mean().rename(columns = {'cash_MONTHS_BALANCE_mean': 'cash_MONTHS_BALANCE_mean_mean'})
dataset = dataset.merge(cash_MONTHS_BALANCE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_MONTHS_BALANCE_mean_mean
gc.collect()
cash_CNT_INSTALMENT_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_CNT_INSTALMENT_mean'].mean().rename(columns = {'cash_CNT_INSTALMENT_mean': 'cash_CNT_INSTALMENT_mean_mean'})
dataset = dataset.merge(cash_CNT_INSTALMENT_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_mean_mean
gc.collect()
cash_CNT_INSTALMENT_FUTURE_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_CNT_INSTALMENT_FUTURE_mean'].mean().rename(columns = {'cash_CNT_INSTALMENT_FUTURE_mean': 'cash_CNT_INSTALMENT_FUTURE_mean_mean'})
dataset = dataset.merge(cash_CNT_INSTALMENT_FUTURE_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_CNT_INSTALMENT_FUTURE_mean_mean
gc.collect()
cash_SK_DPD_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_SK_DPD_mean'].mean().rename(columns = {'cash_SK_DPD_mean': 'cash_SK_DPD_mean_mean'})
dataset = dataset.merge(cash_SK_DPD_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_SK_DPD_mean_mean
gc.collect()
cash_SK_DPD_DEF_mean_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['cash_SK_DPD_DEF_mean'].mean().rename(columns = {'cash_SK_DPD_DEF_mean': 'cash_SK_DPD_DEF_mean_mean'})
dataset = dataset.merge(cash_SK_DPD_DEF_mean_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_SK_DPD_DEF_mean_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Active_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Active'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Active': 'cash_NAME_CONTRACT_STATUS_Active_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Active_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Active_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Amortized_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Amortized debt'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Amortized debt': 'cash_NAME_CONTRACT_STATUS_Amortized_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Amortized_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Amortized_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Approved_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Approved'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Approved': 'cash_NAME_CONTRACT_STATUS_Approved_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Approved_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Approved_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Canceled_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Canceled'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Canceled': 'cash_NAME_CONTRACT_STATUS_Canceled_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Canceled_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Canceled_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Completed_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Completed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Completed': 'cash_NAME_CONTRACT_STATUS_Completed_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Completed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Completed_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Demand_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Demand'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Demand': 'cash_NAME_CONTRACT_STATUS_Demand_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Demand_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Demand_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Returned_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Returned to the store'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Returned to the store': 'cash_NAME_CONTRACT_STATUS_Returned_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Returned_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Returned_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_Signed_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_Signed'].mean().rename(columns = {'NAME_CONTRACT_STATUS_Signed': 'cash_NAME_CONTRACT_STATUS_Signed_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_Signed_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_Signed_mean
gc.collect()
cash_NAME_CONTRACT_STATUS_XNA_mean = cash_stats_by_prev.groupby('SK_ID_CURR', as_index=False)['NAME_CONTRACT_STATUS_XNA'].mean().rename(columns = {'NAME_CONTRACT_STATUS_XNA': 'cash_NAME_CONTRACT_STATUS_XNA_mean'})
dataset = dataset.merge(cash_NAME_CONTRACT_STATUS_XNA_mean, on = 'SK_ID_CURR', how = 'left')
gc.enable()
del cash_NAME_CONTRACT_STATUS_XNA_mean
gc.collect()
# let's free up some memory by deleting some of the dataframes we are done with
gc.enable()
del cash, cash_stats_by_prev
gc.collect()
print('dataset data shape: ', dataset.shape)
dataset.head()
# let's review our dataset data types
dataset.dtypes.value_counts()
# it looks like we have a couple of objects still in our data?
dataset.describe(include=[np.object])
# our outlier variables appear to still be in the format of True or False, so we need to fix this before continuing.
dataset['bureau_DAYS_CREDIT_ENDDATE_max_outlier'] = dataset['bureau_DAYS_CREDIT_ENDDATE_max_outlier'].map({False:0, True:1})
dataset['bureau_DAYS_ENDDATE_FACT_mean_outlier'] = dataset['bureau_DAYS_ENDDATE_FACT_mean_outlier'].map({False:0, True:1})
# check again
dataset['bureau_DAYS_CREDIT_ENDDATE_max_outlier'].describe()
dataset['bureau_DAYS_ENDDATE_FACT_mean_outlier'].describe()
# because the dataset file is so large, let's use a subsample of the data to evaluate the collinear variables
y_temp = dataset[['TARGET']]
X_temp = dataset.drop(['TARGET'], axis=1)
X_big, X_small, y_big, y_small = train_test_split(X_temp, y_temp, test_size=0.2, random_state=1)
# let's first make the correlation matrix
corr = X_small.drop(['SK_ID_CURR'], axis=1)
corr_matrix = corr.corr().abs()
upper_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper_corr.head()
# let's drop any columns with correlations above 0.9
drop_cols = [column for column in upper_corr.columns if any(upper_corr[column] > 0.9)]
print('Columns to remove: ', len(drop_cols))
# now we can drop these columns from the full dataset file
dataset = dataset.drop(columns = drop_cols)
print('dataset data shape: ', dataset.shape)
# delete the dataframes we don't need anymore
gc.enable()
del X_temp, X_big, X_small, y_temp, y_big, y_small, corr, corr_matrix, upper_corr, drop_cols
gc.collect()
# missing values (in percent)
dataset_missing = (dataset.isnull().sum() / len(dataset)).sort_values(ascending = False)
dataset_missing.head(10)
# let's remove columns with more than 75% missing data
dataset_missing = dataset_missing.index[dataset_missing > 0.75]
print('Columns with more than 75% missing values: ', len(dataset_missing))
# let's drop these columns
dataset = dataset.drop(columns = dataset_missing)
print('dataset data shape: ', dataset.shape)
# separate training and testing data for modeling
train = dataset[:train_len]
x_test = dataset[train_len:]
train_ids = train['SK_ID_CURR']
test_ids = x_test['SK_ID_CURR']
train.drop(columns=['SK_ID_CURR'], axis = 1, inplace=True)
x_test.drop(columns=['TARGET', 'SK_ID_CURR'], axis = 1, inplace=True)
# separate training data
train['TARGET'] = train['TARGET'].astype(int)
y_train = train['TARGET']
x_train = train.drop(columns=['TARGET'], axis = 1)
print('x_train data shape: ', x_train.shape)
print('y_train data shape: ', y_train.shape)
print('x_test data shape: ', x_test.shape)
# create a dataframe of all zeroes to hold feature importance calculations
feature_imp = np.zeros(x_train.shape[1])
# create the model to use
# for the parameters, objective is binary (as this is either default or no default that we are predicting),
# boosting type is gradient-based one-side sampling (larger gradients contribute more to information gain so this keeps those 
# with larger gradients and only randomly drops those with smaller), class weight is balanced
# (automatically adjust the weights to be inversely proportional to the frequencies)

model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')
# we will fit the model twice and record the feature importances each time
# note that we will use auc (area under the curve) for evaluation, as on this is what our model will be judged

for i in range(2):
    train_x1, train_x2, train_y1, train_y2 = train_test_split(x_train, y_train, test_size = 0.25, random_state = i)
    model.fit(train_x1, train_y1, early_stopping_rounds=100, eval_set = [(train_x2, train_y2)], eval_metric = 'auc', verbose = 200)
    feature_imp += model.feature_importances_
# review features with most importance
feature_imp = feature_imp / 2
feature_imp = pd.DataFrame({'feature': list(x_train.columns), 'importance': feature_imp}).sort_values('importance', ascending = False)
feature_imp.head(10)
# review features with zero importance
zero_imp = list(feature_imp[feature_imp['importance'] == 0.0]['feature'])
print('count of features with 0 importance: ', len(zero_imp))
feature_imp.tail(10)
# let's drop the features with zero importance
x_train = x_train.drop(columns = zero_imp)
x_test = x_test.drop(columns = zero_imp)
print('x_train data shape: ', x_train.shape)
print('x_test data shape: ', x_test.shape)
# dataframe to hold predictions
test_predictions = np.zeros(x_test.shape[0])
# dataframe for out of fold validation predictions
out_of_fold = np.zeros(x_train.shape[0])
# lists for validation and training scores
valid_scores = []
train_scores = []
k_fold = KFold(n_splits = 5, shuffle = False, random_state = 50)
x_train = np.array(x_train)
x_test = np.array(x_test)
# iterate through each of the five folds
for train_indices, valid_indices in k_fold.split(x_train):
    train_features, train_labels = x_train[train_indices], y_train[train_indices]
    valid_features, valid_labels = x_train[valid_indices], y_train[valid_indices]
    
    # create the model, similar to the one used above for feature importances
    model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', boosting_type='goss',class_weight = 'balanced', 
                               learning_rate = 0.05, reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1, random_state = 50)
    
    # train the model
    model.fit(train_features, train_labels, eval_metric = 'auc',
              eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
              eval_names = ['valid', 'train'], early_stopping_rounds = 100, verbose = 200)
    
    # record the best iteration
    best_iteration = model.best_iteration_
    
    # test predictions
    test_predictions += model.predict_proba(x_test, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
    
    # out of fold predictions
    out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
    
    # record scores
    valid_score = model.best_score_['valid']['auc']
    train_score = model.best_score_['train']['auc']
    valid_scores.append(valid_score)
    train_scores.append(train_score)
    
    # Clean up memory
    gc.enable()
    del model, train_features, valid_features
    gc.collect()
# scores
valid_auc = roc_auc_score(y_train, out_of_fold)

valid_scores.append(valid_auc)
train_scores.append(np.mean(train_scores))

fold_names = list(range(5))
fold_names.append('overall')

metrics = pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores}) 
metrics
# make submission file
submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
submission.to_csv('submission.csv', index = False)
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

import os

pd.set_option('display.max_columns', 500)

%matplotlib inline

color = sns.color_palette()

sns.set_style('darkgrid')

print(os.listdir("../input"))

rcParams['figure.figsize'] = 10, 6

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

lbl = LabelEncoder()

pd.set_option('display.max_columns', 200)



pd.set_option('display.max_rows', 200)
path = "../input"

train_df  = pd.read_csv(path+'/application_train.csv')

test_df = pd.read_csv(path+'/application_test.csv')

sample_submission = pd.read_csv(path+'/sample_submission.csv')
train_df.shape
train_df.head(5)
#train_df = train_df.sample(frac=0.05, random_state=10)
train_df.loc[train_df['TARGET'] == 0, ['TARGET']].count()
train_df.loc[train_df['TARGET'] == 1, ['TARGET']].count()
train_df.describe()
train_df['NAME_CONTRACT_TYPE'].unique()
train_df[["NAME_CONTRACT_TYPE", "TARGET"]].groupby(['NAME_CONTRACT_TYPE'], as_index=False).sum().sort_values(by='TARGET', ascending=False)
train_df[["NAME_CONTRACT_TYPE", "TARGET"]].groupby(['NAME_CONTRACT_TYPE'], as_index=False).sum().sort_values(by='TARGET', ascending=False).plot(kind='bar')
train_df['CNT_CHILDREN'].unique()
train_df[["CNT_CHILDREN", "TARGET"]].groupby(['CNT_CHILDREN'], as_index=False).sum().sort_values(by='TARGET', ascending=False)
train_df['NAME_INCOME_TYPE'].unique()
train_df[["NAME_INCOME_TYPE", "TARGET"]].groupby(['NAME_INCOME_TYPE'], as_index=False).sum().sort_values(by='TARGET', ascending=False)
print(pd.crosstab(train_df['NAME_INCOME_TYPE'], train_df['TARGET'], normalize='index'))

sns.countplot(x='NAME_INCOME_TYPE', hue='TARGET', data=train_df, palette='hls')

plt.xticks(rotation=45)

plt.show()
train_df[["NAME_FAMILY_STATUS", "TARGET"]].groupby(['NAME_FAMILY_STATUS'], as_index=False).sum().sort_values(by='TARGET', ascending=False)
print(pd.crosstab(train_df['NAME_FAMILY_STATUS'], train_df['TARGET'], normalize='index'))

sns.countplot(x='NAME_FAMILY_STATUS', hue='TARGET', data=train_df, palette='hls')

plt.xticks(rotation=45)

plt.show()
train_df['NAME_HOUSING_TYPE'].unique()
train_df[["NAME_HOUSING_TYPE", "TARGET"]].groupby(['NAME_HOUSING_TYPE'], as_index=False).sum().sort_values(by='TARGET', ascending=False)
print(pd.crosstab(train_df['NAME_HOUSING_TYPE'], train_df['TARGET'], normalize='index'))

sns.countplot(x='NAME_HOUSING_TYPE', hue='TARGET', data=train_df, palette='hls')

plt.xticks(rotation=45)

plt.show()
print(pd.crosstab(train_df['NAME_EDUCATION_TYPE'], train_df['TARGET'], normalize='index'))

sns.countplot(x='NAME_EDUCATION_TYPE', hue='TARGET', data=train_df, palette='hls')

plt.xticks(rotation=45)

plt.show()
train_df['OCCUPATION_TYPE'].unique()
train_df[["OCCUPATION_TYPE", "TARGET"]].groupby(['OCCUPATION_TYPE'], as_index=False).sum().sort_values(by='TARGET', ascending=False)
train_df['HOUSETYPE_MODE'].unique()
train_df[["HOUSETYPE_MODE", "TARGET"]].groupby(['HOUSETYPE_MODE'], as_index=False).sum().sort_values(by='TARGET', ascending=False)
train_df['ORGANIZATION_TYPE'].unique()
 #   a = pd.DataFrame(0, index=np.arange(len(train_df)), columns=['docs'])
train_df['docs'] = pd.DataFrame(0, index=np.arange(len(train_df)), columns=['docs'])

for i in range(2, 21):

    cols_name = 'FLAG_DOCUMENT_'+str(i)

    #train_df[cols_name][0:5]

    train_df['docs'] = train_df['docs']+train_df[cols_name]
train_df.head(5)
train_df['AMT_INCOME_TOTAL'].apply(np.log).plot(kind='hist', bins=50) 

plt.show()
train_df['AMT_CREDIT'].apply(np.log).plot(kind='hist', bins=50) 

plt.show()
train_df['AMT_ANNUITY'].apply(np.log).plot(kind='hist', bins=50) 

plt.show()
train_df['AMT_GOODS_PRICE'].apply(np.log).plot(kind='hist', bins=50) 

plt.show()
props = train_df.groupby("HOUSETYPE_MODE")['TARGET'].value_counts(normalize=True).unstack()

props.plot(kind='bar', stacked='True', rot=45)

plt.show()
props = train_df.groupby("OCCUPATION_TYPE")['TARGET'].value_counts(normalize=True).unstack()

props.plot(kind='bar', stacked='True', rot=45,figsize=[12,6])

plt.show()
props = train_df.groupby("NAME_INCOME_TYPE")['TARGET'].value_counts(normalize=True).unstack()

props.plot(kind='bar', stacked='True', rot=45)

plt.show()
props = train_df.groupby("NAME_FAMILY_STATUS")['TARGET'].value_counts(normalize=True).unstack()

props.plot(kind='bar', stacked='True', rot=45)

plt.show()
f, axes = plt.subplots(1, 3, figsize=(12, 6))

sns.countplot(x='TARGET', data=train_df, ax=axes[0])

sns.countplot(x='HOUSETYPE_MODE', hue='TARGET', data=train_df, ax=axes[1] )

sns.countplot(x='NAME_FAMILY_STATUS', hue='TARGET', data=train_df, ax=axes[2])



plt.tight_layout()
print(pd.crosstab(train_df['FLAG_MOBIL'], train_df['TARGET']))
print(pd.crosstab(train_df['FLAG_EMAIL'], train_df['TARGET']))

sns.countplot(x='FLAG_EMAIL', hue='TARGET', data=train_df, palette='hls')
Mob_Loc = train_df.columns.get_loc('FLAG_MOBIL')

Email_Loc = train_df.columns.get_loc('FLAG_EMAIL')



train_df.iloc[:,Mob_Loc:Email_Loc+1].head(5)
print(pd.crosstab(train_df['FLAG_EMP_PHONE'], train_df['TARGET']))

sns.countplot(x='FLAG_EMP_PHONE', hue='TARGET', data=train_df, palette='hls')
print(pd.crosstab(train_df['FLAG_WORK_PHONE'], train_df['TARGET']))

sns.countplot(x='FLAG_WORK_PHONE', hue='TARGET', data=train_df, palette='hls')
print(pd.crosstab(train_df['FLAG_CONT_MOBILE'], train_df['TARGET']))

sns.countplot(x='FLAG_CONT_MOBILE', hue='TARGET', data=train_df, palette='hls')
print(pd.crosstab(train_df['FLAG_PHONE'], train_df['TARGET']))

sns.countplot(x='FLAG_PHONE', hue='TARGET', data=train_df, palette='hls')
del train_df['FLAG_MOBIL']

del test_df['FLAG_MOBIL']
del train_df['WEEKDAY_APPR_PROCESS_START']

del train_df['HOUR_APPR_PROCESS_START']



del test_df['WEEKDAY_APPR_PROCESS_START']

del test_df['HOUR_APPR_PROCESS_START']
train_df.head(5)
print(pd.crosstab(train_df['docs'], train_df['TARGET']))

sns.countplot(x='docs', hue='TARGET', data=train_df, palette='hls')
train_df['DAYS_BIRTH'] = np.ceil(train_df['DAYS_BIRTH']/365)

train_df['Age'] = train_df['DAYS_BIRTH'].abs()
test_df['DAYS_BIRTH'] = np.ceil(test_df['DAYS_BIRTH']/365)

test_df['Age'] = test_df['DAYS_BIRTH'].abs()
train_df['DAYS_LAST_PHONE_CHANGE'] = np.ceil(train_df['DAYS_LAST_PHONE_CHANGE']/365)

train_df['Mob_change'] = train_df['DAYS_LAST_PHONE_CHANGE'].abs()

test_df['DAYS_LAST_PHONE_CHANGE'] = np.ceil(test_df['DAYS_LAST_PHONE_CHANGE']/365)

test_df['Mob_change'] = test_df['DAYS_LAST_PHONE_CHANGE'].abs()
del train_df['DAYS_BIRTH']


del test_df['DAYS_BIRTH']
train_df.head(5)
age_target_zero = train_df[(train_df['Age'] > 0) & (train_df['TARGET'] == 0)]



age_target_one = train_df[(train_df['Age'] > 0) & (train_df['TARGET'] == 1)]



sns.distplot(age_target_zero['Age'], bins =24, color='g')

sns.distplot(age_target_one['Age'], bins =24, color='r')
g = sns.FacetGrid(train_df, col='TARGET', size=5)

g = g.map(sns.distplot, 'Age')

plt.show()
print(pd.crosstab(train_df['REGION_RATING_CLIENT'], train_df['TARGET']))

sns.countplot(x='REGION_RATING_CLIENT', hue='TARGET', data=train_df)

plt.show()
print(pd.crosstab(train_df['REGION_RATING_CLIENT_W_CITY'], train_df['TARGET']))

sns.countplot(x='REGION_RATING_CLIENT_W_CITY', hue='TARGET', data=train_df)

plt.show()
print(pd.crosstab(train_df['REG_REGION_NOT_LIVE_REGION'], train_df['TARGET']))

sns.countplot(x='REG_REGION_NOT_LIVE_REGION', hue='TARGET', data=train_df)

plt.show()
print(pd.crosstab(train_df['REG_REGION_NOT_WORK_REGION'], train_df['TARGET']))

sns.countplot(x='REG_REGION_NOT_WORK_REGION', hue='TARGET', data=train_df)

plt.show()
train_df.head(5)
print(pd.crosstab(train_df['REG_CITY_NOT_LIVE_CITY'], train_df['TARGET']))

sns.countplot(x='REG_CITY_NOT_LIVE_CITY', hue='TARGET', data=train_df)

plt.show()
print(pd.crosstab(train_df['REG_CITY_NOT_WORK_CITY'], train_df['TARGET']))

sns.countplot(x='REG_CITY_NOT_WORK_CITY', hue='TARGET', data=train_df)

plt.show()
print(pd.crosstab(train_df['LIVE_CITY_NOT_WORK_CITY'], train_df['TARGET']))

sns.countplot(x='LIVE_CITY_NOT_WORK_CITY', hue='TARGET', data=train_df)

plt.show()
group = pd.DataFrame()

group['Orgtype_count'] =train_df.groupby(['ORGANIZATION_TYPE'])['ORGANIZATION_TYPE'].count()

group['Orgtype_index'] = group.index



group_top = group.sort_values(by='Orgtype_count', ascending=False).head(15)

plt.figure(figsize=(25, 10))

sns.barplot(x='Orgtype_index', y='Orgtype_count', data=group_top)

plt.xlabel('ORGANIZATION TYPE')

plt.ylabel('Number of Organization')

plt.xticks(rotation=45)

plt.tight_layout()
f, axes = plt.subplots(3, 1, figsize=(12, 6))

c1 = sns.distplot(train_df.loc[(train_df['EXT_SOURCE_1'] >0) & (train_df['TARGET'] == 1),['EXT_SOURCE_1']], kde = True, color='g', ax=axes[0],bins=50)

c1a = sns.distplot(train_df.loc[(train_df['EXT_SOURCE_1'] >0)& (train_df['TARGET'] == 0),['EXT_SOURCE_1']], kde = True, color='r', ax=axes[0],bins=50)



c2 = sns.distplot(train_df.loc[(train_df['EXT_SOURCE_2'] >0) & (train_df['TARGET'] == 1), ['EXT_SOURCE_2']], kde=True, color='g', ax=axes[1],bins=50)

c2a = sns.distplot(train_df.loc[(train_df['EXT_SOURCE_2'] >0) & (train_df['TARGET'] == 0), ['EXT_SOURCE_2']], kde=True, color='r', ax=axes[1],bins=50)



c3 = sns.distplot(train_df.loc[(train_df['EXT_SOURCE_3'] >0) & (train_df['TARGET'] == 1), ['EXT_SOURCE_3']], kde=True, color='g', ax=axes[2],bins=50)

c3a = sns.distplot(train_df.loc[(train_df['EXT_SOURCE_3'] >0)& (train_df['TARGET'] == 0), ['EXT_SOURCE_3']], kde=True,  color='r', ax=axes[2],bins=50)
train_df.head()
f, axes = plt.subplots(16, 1, figsize=(12, 50))

c1 = sns.distplot(train_df.loc[(train_df['APARTMENTS_AVG'] >0) & (train_df['TARGET'] == 1),['APARTMENTS_AVG']], color='g' ,kde = True, ax=axes[0],bins=50)

c1a = sns.distplot(train_df.loc[(train_df['APARTMENTS_AVG'] >0) & (train_df['TARGET'] == 0),['APARTMENTS_AVG']], color='r', kde = True, ax=axes[0],bins=50)





c2 = sns.distplot(train_df.loc[(train_df['BASEMENTAREA_AVG'] >0) & (train_df['TARGET'] == 1),['BASEMENTAREA_AVG']], kde=True, color='g', ax=axes[1],bins=50)

c2a = sns.distplot(train_df.loc[(train_df['BASEMENTAREA_AVG'] >0) & (train_df['TARGET'] == 0),['BASEMENTAREA_AVG']], kde=True, color='r', ax=axes[1],bins=50)



c3 = sns.distplot(train_df.loc[(train_df['YEARS_BEGINEXPLUATATION_AVG'] >0) & (train_df['TARGET'] == 1),['YEARS_BEGINEXPLUATATION_AVG']], color='g', kde=True, ax=axes[2],bins=50)

c3a = sns.distplot(train_df.loc[(train_df['YEARS_BEGINEXPLUATATION_AVG'] >0) & (train_df['TARGET'] == 0),['YEARS_BEGINEXPLUATATION_AVG']], color='r', kde=True, ax=axes[2],bins=50)



c4 = sns.distplot(train_df.loc[(train_df['YEARS_BUILD_AVG'] >0) & (train_df['TARGET'] == 1),['YEARS_BUILD_AVG']], kde=True,color='g', ax=axes[3],bins=50)

c4a = sns.distplot(train_df.loc[(train_df['YEARS_BUILD_AVG'] >0) & (train_df['TARGET'] == 0),['YEARS_BUILD_AVG']], kde=True, color='r', ax=axes[3],bins=50)



c5 = sns.distplot(train_df.loc[(train_df['COMMONAREA_AVG'] >0) & (train_df['TARGET'] == 1), ['COMMONAREA_AVG']], kde=True, color='g', ax=axes[4],bins=50)

c5a = sns.distplot(train_df.loc[(train_df['COMMONAREA_AVG'] >0)& (train_df['TARGET'] == 0), ['COMMONAREA_AVG']], kde=True, color='r', ax=axes[4],bins=50)



c6 = sns.distplot(train_df.loc[(train_df['ELEVATORS_AVG'] >0)& (train_df['TARGET'] == 1), ['ELEVATORS_AVG']], kde=True, color='g', ax=axes[5],bins=50)

c6a = sns.distplot(train_df.loc[(train_df['ELEVATORS_AVG'] >0)& (train_df['TARGET'] == 0), ['ELEVATORS_AVG']], kde=True, color='r', ax=axes[5],bins=50)



c7 = sns.distplot(train_df.loc[(train_df['ENTRANCES_AVG'] >0)& (train_df['TARGET'] == 1),['ENTRANCES_AVG']], kde=True, color='g', ax=axes[6],bins=50)

c7a = sns.distplot(train_df.loc[(train_df['ENTRANCES_AVG'] >0)& (train_df['TARGET'] == 0),['ENTRANCES_AVG']], kde=True, color='r', ax=axes[6],bins=50)



c8 = sns.distplot(train_df.loc[(train_df['FLOORSMAX_AVG'] >0)& (train_df['TARGET'] == 1),['FLOORSMAX_AVG']], kde=True, color='g', ax=axes[7],bins=50)

c8a = sns.distplot(train_df.loc[(train_df['FLOORSMAX_AVG'] >0)& (train_df['TARGET'] == 0),['FLOORSMAX_AVG']], kde=True, color='r', ax=axes[7],bins=50)



c9 = sns.distplot(train_df.loc[(train_df['FLOORSMIN_AVG'] >0)& (train_df['TARGET'] == 1),['FLOORSMIN_AVG']], kde=True, color='g', ax=axes[8],bins=50)

c9a = sns.distplot(train_df.loc[(train_df['FLOORSMIN_AVG'] >0)& (train_df['TARGET'] == 0),['FLOORSMIN_AVG']], kde=True, color='r', ax=axes[8],bins=50)



c10 = sns.distplot(train_df.loc[(train_df['LANDAREA_AVG'] >0)& (train_df['TARGET'] == 1),['LANDAREA_AVG']], kde=True, color='g', ax=axes[9],bins=50)

c10a = sns.distplot(train_df.loc[(train_df['LANDAREA_AVG'] >0)& (train_df['TARGET'] == 0),['LANDAREA_AVG']], kde=True, color='r',  ax=axes[9],bins=50)



c11 = sns.distplot(train_df.loc[(train_df['LIVINGAPARTMENTS_AVG'] >0)& (train_df['TARGET'] == 1),['LIVINGAPARTMENTS_AVG']], kde=True,color='g', ax=axes[10],bins=50)

c11a = sns.distplot(train_df.loc[(train_df['LIVINGAPARTMENTS_AVG'] >0)& (train_df['TARGET'] == 0),['LIVINGAPARTMENTS_AVG']], kde=True,color='r', ax=axes[10],bins=50)





c12 = sns.distplot(train_df.loc[(train_df['LIVINGAREA_AVG'] >0)& (train_df['TARGET'] == 1),['LIVINGAREA_AVG']], kde=True, color='g', ax=axes[11],bins=50)

c12a = sns.distplot(train_df.loc[(train_df['LIVINGAREA_AVG'] >0)& (train_df['TARGET'] == 0),['LIVINGAREA_AVG']], kde=True, color='r', ax=axes[11],bins=50)



c13 = sns.distplot(train_df.loc[(train_df['NONLIVINGAPARTMENTS_AVG'] >0)& (train_df['TARGET'] == 1),['NONLIVINGAPARTMENTS_AVG']], kde=True, color='g', ax=axes[12],bins=50)

c13a = sns.distplot(train_df.loc[(train_df['NONLIVINGAPARTMENTS_AVG'] >0)& (train_df['TARGET'] == 0),['NONLIVINGAPARTMENTS_AVG']], kde=True, color='r', ax=axes[12],bins=50)



c14 = sns.distplot(train_df.loc[(train_df['NONLIVINGAREA_AVG'] >0)& (train_df['TARGET'] == 1),['NONLIVINGAREA_AVG']], kde=True,color='g', ax=axes[13],bins=50)

c14a = sns.distplot(train_df.loc[(train_df['NONLIVINGAREA_AVG'] >0)& (train_df['TARGET'] == 0),['NONLIVINGAREA_AVG']], kde=True,  color='r',ax=axes[13],bins=50)



c15 = sns.distplot(train_df.loc[(train_df['LIVINGAPARTMENTS_AVG'] >0)& (train_df['TARGET'] == 1),['LIVINGAPARTMENTS_AVG']], kde=True,color='g', ax=axes[14],bins=50)

c15a = sns.distplot(train_df.loc[(train_df['LIVINGAPARTMENTS_AVG'] >0)& (train_df['TARGET'] == 0),['LIVINGAPARTMENTS_AVG']], kde=True,  color='r',ax=axes[14],bins=50)



c16 = sns.distplot(train_df.loc[(train_df['LIVINGAREA_AVG'] >0)& (train_df['TARGET'] == 1),['LIVINGAREA_AVG']], kde=True, color='g',ax=axes[15],bins=50)

c16a = sns.distplot(train_df.loc[(train_df['LIVINGAREA_AVG'] >0)& (train_df['TARGET'] == 0),['LIVINGAREA_AVG']], kde=True,  color='r', ax=axes[15],bins=50)
train_df.loc[:,train_df.dtypes =='object'].columns
train_df['WALLSMATERIAL_MODE'].unique()
train_df['EMERGENCYSTATE_MODE'].unique()
print(pd.crosstab(train_df['WALLSMATERIAL_MODE'], train_df['TARGET']))

sns.countplot(x='WALLSMATERIAL_MODE', hue='TARGET', data=train_df)

plt.xticks(rotation=45)

plt.show()
print(pd.crosstab(train_df['EMERGENCYSTATE_MODE'], train_df['TARGET']))

sns.countplot(x='EMERGENCYSTATE_MODE', hue='TARGET', data=train_df)

plt.xticks(rotation=45)

plt.show()
train_df.head(5)
null_values = pd.DataFrame({'col_names':pd.isnull(train_df).sum().index,'col_count':pd.isnull(train_df).sum().values}) 

print('Total {} columns hava a null values.'.format(null_values.loc[null_values['col_count'] > 0, ].shape[0]))
null_values.loc[null_values['col_count'] > 0, ].sort_values(by='col_count', ascending=False)
train_df.shape
train_df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(train_df['AMT_REQ_CREDIT_BUREAU_YEAR'].min(), inplace=True)

train_df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(train_df['AMT_REQ_CREDIT_BUREAU_QRT'].min(), inplace=True)

train_df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(train_df['AMT_REQ_CREDIT_BUREAU_MON'].min(), inplace=True)



train_df['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(train_df['AMT_REQ_CREDIT_BUREAU_WEEK'].min(), inplace=True)

train_df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(train_df['AMT_REQ_CREDIT_BUREAU_DAY'].min(), inplace=True)

train_df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(train_df['AMT_REQ_CREDIT_BUREAU_HOUR'].min(), inplace=True)





train_df['EXT_SOURCE_3'].fillna(train_df['EXT_SOURCE_3'].min(), inplace=True)

train_df['TOTALAREA_MODE'].fillna(train_df['TOTALAREA_MODE'].min(), inplace=True)

train_df['YEARS_BEGINEXPLUATATION_AVG'].fillna(train_df['YEARS_BEGINEXPLUATATION_AVG'].min(), inplace=True)

train_df['YEARS_BEGINEXPLUATATION_MODE'].fillna(train_df['YEARS_BEGINEXPLUATATION_MODE'].min(), inplace=True)



train_df['FLOORSMAX_AVG'].fillna(train_df['FLOORSMAX_AVG'].min(), inplace=True)

train_df['FLOORSMAX_MODE'].fillna(train_df['FLOORSMAX_MODE'].min(), inplace=True)

train_df['FLOORSMAX_MEDI'].fillna(train_df['FLOORSMAX_MEDI'].min(), inplace=True)



train_df['LIVINGAREA_AVG'].fillna(train_df['LIVINGAREA_AVG'].min(), inplace=True)

train_df['LIVINGAREA_MODE'].fillna(train_df['LIVINGAREA_MODE'].min(), inplace=True)

train_df['LIVINGAREA_MEDI'].fillna(train_df['LIVINGAREA_MEDI'].min(), inplace=True)



train_df['ENTRANCES_MODE'].fillna(train_df['ENTRANCES_MODE'].min(), inplace=True)

train_df['ENTRANCES_MEDI'].fillna(train_df['ENTRANCES_MEDI'].min(), inplace=True)

train_df['ENTRANCES_AVG'].fillna(train_df['ENTRANCES_AVG'].min(), inplace=True)



train_df['APARTMENTS_AVG'].fillna(train_df['APARTMENTS_AVG'].min(), inplace=True)

train_df['APARTMENTS_MEDI'].fillna(train_df['APARTMENTS_MEDI'].min(), inplace=True)

train_df['APARTMENTS_MODE'].fillna(train_df['APARTMENTS_MODE'].min(), inplace=True)



train_df['ELEVATORS_AVG'].fillna(train_df['ELEVATORS_AVG'].min(), inplace=True)

train_df['ELEVATORS_MODE'].fillna(train_df['ELEVATORS_MODE'].min(), inplace=True)

train_df['ELEVATORS_MEDI'].fillna(train_df['ELEVATORS_MEDI'].min(), inplace=True)



train_df['NONLIVINGAREA_AVG'].fillna(train_df['NONLIVINGAREA_AVG'].min(), inplace=True)

train_df['NONLIVINGAREA_MODE'].fillna(train_df['NONLIVINGAREA_MODE'].min(), inplace=True)

train_df['NONLIVINGAREA_MEDI'].fillna(train_df['NONLIVINGAREA_MEDI'].min(), inplace=True)



train_df['BASEMENTAREA_AVG'].fillna(train_df['BASEMENTAREA_AVG'].min(), inplace=True)

train_df['BASEMENTAREA_MEDI'].fillna(train_df['BASEMENTAREA_MEDI'].min(), inplace=True)

train_df['BASEMENTAREA_MODE'].fillna(train_df['BASEMENTAREA_MODE'].min(), inplace=True)



train_df['LANDAREA_MODE'].fillna(train_df['LANDAREA_MODE'].min(), inplace=True)

train_df['LANDAREA_AVG'].fillna(train_df['LANDAREA_AVG'].min(), inplace=True)

train_df['LANDAREA_MEDI'].fillna(train_df['LANDAREA_MEDI'].min(), inplace=True)





train_df['YEARS_BUILD_MEDI'].fillna(train_df['YEARS_BUILD_MEDI'].min(), inplace=True)

train_df['YEARS_BUILD_AVG'].fillna(train_df['YEARS_BUILD_AVG'].min(), inplace=True)

train_df['YEARS_BUILD_MODE'].fillna(train_df['YEARS_BUILD_MODE'].min(), inplace=True)





train_df['FLOORSMIN_MODE'].fillna(train_df['FLOORSMIN_MODE'].min(), inplace=True)

train_df['FLOORSMIN_AVG'].fillna(train_df['FLOORSMIN_AVG'].min(), inplace=True)

train_df['FLOORSMIN_MEDI'].fillna(train_df['FLOORSMIN_MEDI'].min(), inplace=True)





train_df['LIVINGAPARTMENTS_MODE'].fillna(train_df['LIVINGAPARTMENTS_MODE'].min(), inplace=True)

train_df['LIVINGAPARTMENTS_AVG'].fillna(train_df['LIVINGAPARTMENTS_AVG'].min(), inplace=True)

train_df['LIVINGAPARTMENTS_MEDI'].fillna(train_df['LIVINGAPARTMENTS_MEDI'].min(), inplace=True)





train_df['NONLIVINGAPARTMENTS_MODE'].fillna(train_df['NONLIVINGAPARTMENTS_MODE'].min(), inplace=True)

train_df['NONLIVINGAPARTMENTS_AVG'].fillna(train_df['NONLIVINGAPARTMENTS_AVG'].min(), inplace=True)

train_df['NONLIVINGAPARTMENTS_MEDI'].fillna(train_df['NONLIVINGAPARTMENTS_MEDI'].min(), inplace=True)



train_df['COMMONAREA_AVG'].fillna(train_df['COMMONAREA_AVG'].min(), inplace=True)

train_df['COMMONAREA_MODE'].fillna(train_df['COMMONAREA_MODE'].min(), inplace=True)

train_df['COMMONAREA_MEDI'].fillna(train_df['COMMONAREA_MEDI'].min(), inplace=True)



train_df['OWN_CAR_AGE'].fillna(train_df['OWN_CAR_AGE'].min(), inplace=True)

train_df['EXT_SOURCE_1'].fillna(train_df['EXT_SOURCE_1'].min(), inplace=True)



train_df['YEARS_BEGINEXPLUATATION_MEDI'].fillna(train_df['YEARS_BEGINEXPLUATATION_MEDI'].min(), inplace=True)



train_df['AMT_ANNUITY'].fillna(train_df['AMT_ANNUITY'].min(), inplace=True)

train_df['CNT_FAM_MEMBERS'].fillna(train_df['CNT_FAM_MEMBERS'].min(), inplace=True)
test_df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(test_df['AMT_REQ_CREDIT_BUREAU_YEAR'].min(), inplace=True)

test_df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(test_df['AMT_REQ_CREDIT_BUREAU_QRT'].min(), inplace=True)

test_df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(test_df['AMT_REQ_CREDIT_BUREAU_MON'].min(), inplace=True)



test_df['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(test_df['AMT_REQ_CREDIT_BUREAU_WEEK'].min(), inplace=True)

test_df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(test_df['AMT_REQ_CREDIT_BUREAU_DAY'].min(), inplace=True)

test_df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(test_df['AMT_REQ_CREDIT_BUREAU_HOUR'].min(), inplace=True)





test_df['EXT_SOURCE_3'].fillna(test_df['EXT_SOURCE_3'].min(), inplace=True)

test_df['TOTALAREA_MODE'].fillna(test_df['TOTALAREA_MODE'].min(), inplace=True)

test_df['YEARS_BEGINEXPLUATATION_AVG'].fillna(test_df['YEARS_BEGINEXPLUATATION_AVG'].min(), inplace=True)

test_df['YEARS_BEGINEXPLUATATION_MODE'].fillna(test_df['YEARS_BEGINEXPLUATATION_MODE'].min(), inplace=True)



test_df['FLOORSMAX_AVG'].fillna(test_df['FLOORSMAX_AVG'].min(), inplace=True)

test_df['FLOORSMAX_MODE'].fillna(test_df['FLOORSMAX_MODE'].min(), inplace=True)

test_df['FLOORSMAX_MEDI'].fillna(test_df['FLOORSMAX_MEDI'].min(), inplace=True)



test_df['LIVINGAREA_AVG'].fillna(test_df['LIVINGAREA_AVG'].min(), inplace=True)

test_df['LIVINGAREA_MODE'].fillna(test_df['LIVINGAREA_MODE'].min(), inplace=True)

test_df['LIVINGAREA_MEDI'].fillna(test_df['LIVINGAREA_MEDI'].min(), inplace=True)



test_df['ENTRANCES_MODE'].fillna(test_df['ENTRANCES_MODE'].min(), inplace=True)

test_df['ENTRANCES_MEDI'].fillna(test_df['ENTRANCES_MEDI'].min(), inplace=True)

test_df['ENTRANCES_AVG'].fillna(test_df['ENTRANCES_AVG'].min(), inplace=True)



test_df['APARTMENTS_AVG'].fillna(test_df['APARTMENTS_AVG'].min(), inplace=True)

test_df['APARTMENTS_MEDI'].fillna(test_df['APARTMENTS_MEDI'].min(), inplace=True)

test_df['APARTMENTS_MODE'].fillna(test_df['APARTMENTS_MODE'].min(), inplace=True)



test_df['ELEVATORS_AVG'].fillna(test_df['ELEVATORS_AVG'].min(), inplace=True)

test_df['ELEVATORS_MODE'].fillna(test_df['ELEVATORS_MODE'].min(), inplace=True)

test_df['ELEVATORS_MEDI'].fillna(test_df['ELEVATORS_MEDI'].min(), inplace=True)



test_df['NONLIVINGAREA_AVG'].fillna(test_df['NONLIVINGAREA_AVG'].min(), inplace=True)

test_df['NONLIVINGAREA_MODE'].fillna(test_df['NONLIVINGAREA_MODE'].min(), inplace=True)

test_df['NONLIVINGAREA_MEDI'].fillna(test_df['NONLIVINGAREA_MEDI'].min(), inplace=True)



test_df['BASEMENTAREA_AVG'].fillna(test_df['BASEMENTAREA_AVG'].min(), inplace=True)

test_df['BASEMENTAREA_MEDI'].fillna(test_df['BASEMENTAREA_MEDI'].min(), inplace=True)

test_df['BASEMENTAREA_MODE'].fillna(test_df['BASEMENTAREA_MODE'].min(), inplace=True)



test_df['LANDAREA_MODE'].fillna(test_df['LANDAREA_MODE'].min(), inplace=True)

test_df['LANDAREA_AVG'].fillna(test_df['LANDAREA_AVG'].min(), inplace=True)

test_df['LANDAREA_MEDI'].fillna(test_df['LANDAREA_MEDI'].min(), inplace=True)





test_df['YEARS_BUILD_MEDI'].fillna(test_df['YEARS_BUILD_MEDI'].min(), inplace=True)

test_df['YEARS_BUILD_AVG'].fillna(test_df['YEARS_BUILD_AVG'].min(), inplace=True)

test_df['YEARS_BUILD_MODE'].fillna(test_df['YEARS_BUILD_MODE'].min(), inplace=True)





test_df['FLOORSMIN_MODE'].fillna(test_df['FLOORSMIN_MODE'].min(), inplace=True)

test_df['FLOORSMIN_AVG'].fillna(test_df['FLOORSMIN_AVG'].min(), inplace=True)

test_df['FLOORSMIN_MEDI'].fillna(test_df['FLOORSMIN_MEDI'].min(), inplace=True)





test_df['LIVINGAPARTMENTS_MODE'].fillna(test_df['LIVINGAPARTMENTS_MODE'].min(), inplace=True)

test_df['LIVINGAPARTMENTS_AVG'].fillna(test_df['LIVINGAPARTMENTS_AVG'].min(), inplace=True)

test_df['LIVINGAPARTMENTS_MEDI'].fillna(test_df['LIVINGAPARTMENTS_MEDI'].min(), inplace=True)





test_df['NONLIVINGAPARTMENTS_MODE'].fillna(test_df['NONLIVINGAPARTMENTS_MODE'].min(), inplace=True)

test_df['NONLIVINGAPARTMENTS_AVG'].fillna(test_df['NONLIVINGAPARTMENTS_AVG'].min(), inplace=True)

test_df['NONLIVINGAPARTMENTS_MEDI'].fillna(test_df['NONLIVINGAPARTMENTS_MEDI'].min(), inplace=True)



test_df['COMMONAREA_AVG'].fillna(test_df['COMMONAREA_AVG'].min(), inplace=True)

test_df['COMMONAREA_MODE'].fillna(test_df['COMMONAREA_MODE'].min(), inplace=True)

test_df['COMMONAREA_MEDI'].fillna(test_df['COMMONAREA_MEDI'].min(), inplace=True)





test_df['OWN_CAR_AGE'].fillna(test_df['OWN_CAR_AGE'].min(), inplace=True)



test_df['EXT_SOURCE_1'].fillna(test_df['EXT_SOURCE_1'].min(), inplace=True)



test_df['YEARS_BEGINEXPLUATATION_MEDI'].fillna(test_df['YEARS_BEGINEXPLUATATION_MEDI'].min(), inplace=True)



test_df['AMT_ANNUITY'].fillna(test_df['AMT_ANNUITY'].min(), inplace=True)

test_df['CNT_FAM_MEMBERS'].fillna(test_df['CNT_FAM_MEMBERS'].min(), inplace=True)
print(train_df['NAME_TYPE_SUITE'].mode()[0])

train_df['NAME_TYPE_SUITE'].fillna(train_df['NAME_TYPE_SUITE'].mode()[0], inplace=True)

test_df['NAME_TYPE_SUITE'].fillna(train_df['NAME_TYPE_SUITE'].mode()[0], inplace=True)
print(train_df['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0])

train_df['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(train_df['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)

test_df['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(train_df['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
print(train_df['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0])

train_df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(train_df['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)

test_df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(train_df['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
print(train_df['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0])

train_df['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(train_df['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)

test_df['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(train_df['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
print(train_df['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0])

train_df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(train_df['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)

test_df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(train_df['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
print(train_df['EXT_SOURCE_2'].mode()[0])

train_df['EXT_SOURCE_2'].fillna(train_df['EXT_SOURCE_2'].mode()[0], inplace=True)

test_df['EXT_SOURCE_2'].fillna(train_df['EXT_SOURCE_2'].mode()[0], inplace=True)
print(train_df['AMT_GOODS_PRICE'].mode()[0])

train_df['AMT_GOODS_PRICE'].fillna(train_df['AMT_GOODS_PRICE'].mode()[0], inplace=True)

test_df['AMT_GOODS_PRICE'].fillna(train_df['AMT_GOODS_PRICE'].mode()[0], inplace=True)
null_values = pd.DataFrame({'col_names':pd.isnull(train_df).sum().index,'col_count':pd.isnull(train_df).sum().values}) 

print('Total {} columns hava a null values.'.format(null_values.loc[null_values['col_count'] > 0, ].shape[0]))
null_values.loc[null_values['col_count'] > 0, ]
obj_col = train_df.loc[:, train_df.dtypes == object].columns
for colname in obj_col.values:

    train_df[colname]= lbl.fit_transform(list(train_df[colname].astype(str)))

    test_df[colname]= lbl.fit_transform(list(test_df[colname].astype(str)))
train_df.head(10)
null_values = pd.DataFrame({'col_names':pd.isnull(train_df).sum().index,'col_count':pd.isnull(train_df).sum().values}) 

print('Total {} columns hava a null values.'.format(null_values.loc[null_values['col_count'] > 0, ].shape[0]))

test_df['CNT_FAM_MEMBERS'].head()
null_values.loc[null_values['col_count'] > 0, ]
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
feature_col = ['NAME_CONTRACT_TYPE',

               'CODE_GENDER',

               'FLAG_OWN_CAR',

               'FLAG_OWN_REALTY',

               'CNT_CHILDREN',

               'AMT_INCOME_TOTAL',

               'AMT_INCOME_TOTAL',

'AMT_CREDIT',

'AMT_ANNUITY',

'AMT_GOODS_PRICE',

'NAME_TYPE_SUITE',

'NAME_INCOME_TYPE',

'NAME_EDUCATION_TYPE',

'NAME_FAMILY_STATUS',

'NAME_HOUSING_TYPE',

'REGION_POPULATION_RELATIVE',

'DAYS_EMPLOYED',

'DAYS_REGISTRATION',

'DAYS_ID_PUBLISH',

'OWN_CAR_AGE',

'FLAG_EMP_PHONE',

'FLAG_WORK_PHONE',

'FLAG_CONT_MOBILE',

'FLAG_PHONE',

'FLAG_EMAIL',

'OCCUPATION_TYPE',

'CNT_FAM_MEMBERS',

'REGION_RATING_CLIENT',

'REGION_RATING_CLIENT_W_CITY',

'REG_REGION_NOT_LIVE_REGION',

'REG_REGION_NOT_WORK_REGION',

'LIVE_REGION_NOT_WORK_REGION',

'REG_CITY_NOT_LIVE_CITY',

'REG_CITY_NOT_WORK_CITY',

'LIVE_CITY_NOT_WORK_CITY',

'ORGANIZATION_TYPE',

'EXT_SOURCE_1',

'EXT_SOURCE_2',

'EXT_SOURCE_3',

'APARTMENTS_AVG',

'BASEMENTAREA_AVG',

'YEARS_BEGINEXPLUATATION_AVG',

'YEARS_BUILD_AVG',

'COMMONAREA_AVG',

'ELEVATORS_AVG',

'ENTRANCES_AVG',

'FLOORSMAX_AVG',

'FLOORSMIN_AVG',

'LANDAREA_AVG',

'LIVINGAPARTMENTS_AVG',

'LIVINGAREA_AVG',

'NONLIVINGAPARTMENTS_AVG',

'NONLIVINGAREA_AVG',

'APARTMENTS_MODE',

'BASEMENTAREA_MODE',

'YEARS_BEGINEXPLUATATION_MODE',

'YEARS_BUILD_MODE',

'COMMONAREA_MODE',

'ELEVATORS_MODE',

'ENTRANCES_MODE',

'FLOORSMAX_MODE',

'FLOORSMIN_MODE',

'LANDAREA_MODE',

'LIVINGAPARTMENTS_MODE',

'LIVINGAREA_MODE',

'NONLIVINGAPARTMENTS_MODE',

'NONLIVINGAREA_MODE',

'APARTMENTS_MEDI',

'BASEMENTAREA_MEDI',

'YEARS_BEGINEXPLUATATION_MEDI',

'YEARS_BUILD_MEDI',

'COMMONAREA_MEDI',

'ELEVATORS_MEDI',

'ENTRANCES_MEDI',

'FLOORSMAX_MEDI',

'FLOORSMIN_MEDI',

'LANDAREA_MEDI',

'LIVINGAPARTMENTS_MEDI',

'LIVINGAREA_MEDI',

'NONLIVINGAPARTMENTS_MEDI',

'NONLIVINGAREA_MEDI',

'FONDKAPREMONT_MODE',

'HOUSETYPE_MODE',

'TOTALAREA_MODE',

'WALLSMATERIAL_MODE',

'EMERGENCYSTATE_MODE',

'OBS_30_CNT_SOCIAL_CIRCLE',

'DEF_30_CNT_SOCIAL_CIRCLE',

'OBS_60_CNT_SOCIAL_CIRCLE',

'DEF_60_CNT_SOCIAL_CIRCLE',

'DAYS_LAST_PHONE_CHANGE',

'FLAG_DOCUMENT_2',

'FLAG_DOCUMENT_3',

'FLAG_DOCUMENT_4',

'FLAG_DOCUMENT_5',

'FLAG_DOCUMENT_6',

'FLAG_DOCUMENT_7',

'FLAG_DOCUMENT_8',

'FLAG_DOCUMENT_9',

'FLAG_DOCUMENT_10',

'FLAG_DOCUMENT_11',

'FLAG_DOCUMENT_12',

'FLAG_DOCUMENT_13',

'FLAG_DOCUMENT_14',

'FLAG_DOCUMENT_15',

'FLAG_DOCUMENT_16',

'FLAG_DOCUMENT_17',

'FLAG_DOCUMENT_18',

'FLAG_DOCUMENT_19',

'FLAG_DOCUMENT_20',

'FLAG_DOCUMENT_21',

'AMT_REQ_CREDIT_BUREAU_HOUR',

'AMT_REQ_CREDIT_BUREAU_DAY',

'AMT_REQ_CREDIT_BUREAU_WEEK',

'AMT_REQ_CREDIT_BUREAU_MON',

'AMT_REQ_CREDIT_BUREAU_QRT',

'AMT_REQ_CREDIT_BUREAU_YEAR',

'Age',

'Mob_change']

label_col = ['TARGET']
X_train, X_test, Y_train, Y_test = train_test_split(train_df[feature_col],train_df[label_col], test_size=0.2, random_state=52)
model_name = []

model_score =[]
X_train = np.array(X_train)

X_test = np.array(X_test)

Y_train = np.array(Y_train)

Y_test = np.array(Y_test)
'''linsvc = LinearSVC()

linsvc.fit(X_train, Y_train)

linsvc_score = round(linsvc.score(X_train, Y_train)*100, 2)

model_name.append('LinearSVC')

model_score.append(linsvc_score)

linsvc_score'''
'''svc = SVC()

svc.fit(X_train,Y_train)

svc_score = round(svc.score(X_train,Y_train)*100, 2)

model_name.append('SVC')

model_score.append(svc_score)

svc_score'''
'''kneighbors = KNeighborsClassifier()

kneighbors.fit(X_train,Y_train)

kneighbors_score = round(kneighbors.score(X_train,Y_train)*100, 2)

model_name.append('KNeighborsClassifier')

model_score.append(kneighbors_score)

kneighbors_score'''
randomforest = RandomForestClassifier()

randomforest.fit(X_train,Y_train)

randomforest_score = round(randomforest.score(X_train,Y_train)*100, 2)

model_name.append('RandomForestClassifier')

model_score.append(randomforest_score)

randomforest_score
'''gradient = GradientBoostingClassifier()

gradient.fit(X_train,Y_train)

gradient_score = round(gradient.score(X_train,Y_train)*100, 2)

model_name.append('GradientBoostingClassifier')

model_score.append(gradient_score)

gradient_score'''
all_score = pd.DataFrame({'model_name':model_name, 'model_score':model_score})

all_score
predict_result = randomforest.predict(train_df[feature_col])
my_submission = pd.DataFrame({'SK_ID_CURR':train_df['SK_ID_CURR'], 'TARGET':predict_result})
my_submission.to_csv('my_submission.csv', index=False)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import gc

import warnings

import time

warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

#app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
app_num_basic_col = [

'SK_ID_CURR',

'TARGET',

'CNT_CHILDREN',

'AMT_INCOME_TOTAL',

'AMT_CREDIT',

'AMT_ANNUITY',

'AMT_GOODS_PRICE',

'REGION_POPULATION_RELATIVE',

'DAYS_BIRTH',

'DAYS_EMPLOYED',

'DAYS_REGISTRATION',

'DAYS_ID_PUBLISH',

'CNT_FAM_MEMBERS',

'REGION_RATING_CLIENT',

'REGION_RATING_CLIENT_W_CITY',

'REG_REGION_NOT_LIVE_REGION',

'REG_REGION_NOT_WORK_REGION',

'LIVE_REGION_NOT_WORK_REGION',

'REG_CITY_NOT_LIVE_CITY',

'REG_CITY_NOT_WORK_CITY',

'LIVE_CITY_NOT_WORK_CITY']
app_cat_basic_col = ['NAME_CONTRACT_TYPE',

'FLAG_OWN_CAR',

'FLAG_OWN_REALTY',

'CODE_GENDER',

'NAME_TYPE_SUITE',

'NAME_INCOME_TYPE',

'NAME_EDUCATION_TYPE',

'NAME_FAMILY_STATUS',

'NAME_HOUSING_TYPE',

'OCCUPATION_TYPE',

'ORGANIZATION_TYPE']
len(app_num_basic_col)
len(app_cat_basic_col)
df = df[app_num_basic_col + app_cat_basic_col]
df.shape
df['TARGET'].value_counts()
def find_missing(data):

    ## Number of missing values

    missing_cnt = data.isnull().sum().values

    ## Total

    total = data.shape[0]

    ##Percentage of Missing values

    percentage = missing_cnt/total * 100

    missing_df = pd.DataFrame(data={'Total': total, 'Missing Count' : missing_cnt,'Percentage' : percentage}, 

                              index=data.columns.values)

    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    return missing_df
find_missing(df[app_num_basic_col])
df['AMT_GOODS_PRICE']=df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())

df['AMT_ANNUITY']=df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())

df['CNT_FAM_MEMBERS']=df['CNT_FAM_MEMBERS'].fillna(df['CNT_FAM_MEMBERS'].median())
find_missing(df[app_num_basic_col])
find_missing(df[app_cat_basic_col])
df.OCCUPATION_TYPE.unique()
df.NAME_TYPE_SUITE.unique()
app_cat_basic_col.remove('OCCUPATION_TYPE')
df.drop('OCCUPATION_TYPE',inplace=True, axis=1)
df.shape
df['NAME_TYPE_SUITE']=df['NAME_TYPE_SUITE'].fillna('NTS_XNA')
basic_features = app_num_basic_col + app_cat_basic_col 
len(basic_features)
find_missing(df[basic_features])
sns.boxplot(data=df['DAYS_EMPLOYED'])
df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
# Create an anomalous flag column

#df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243



# Replace the anomalous values with nan

df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)



df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())



df['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
len(basic_features)
#basic_features.append('DAYS_EMPLOYED_ANOM')
len(basic_features)
df[df['DAYS_EMPLOYED'] / -365 > 8]['DAYS_EMPLOYED'].count()
(df['DAYS_BIRTH'] / -365).describe()
df[df['CODE_GENDER'] == 'XNA']
df = df[df['CODE_GENDER'] != 'XNA']
#df.drop('DAYS_EMPLOYED_ANOM',inplace=True,axis=1)

df.shape

df[['SK_ID_CURR','CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'DAYS_EMPLOYED']].head(10)
# Categorical features with Binary encode (0 or 1; two categories)

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'DAYS_EMPLOYED']:

    df[bin_feature], uniques = pd.factorize(df[bin_feature])
df[['SK_ID_CURR','CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'DAYS_EMPLOYED']].head(10)
one_hot_encode_col = ['NAME_CONTRACT_TYPE',

'NAME_TYPE_SUITE',

'NAME_INCOME_TYPE',

'NAME_EDUCATION_TYPE',

'NAME_FAMILY_STATUS',

'NAME_HOUSING_TYPE',

'ORGANIZATION_TYPE']
dummy_df = pd.get_dummies(df[one_hot_encode_col], dummy_na=False, drop_first=True)
len(dummy_df.columns)
df.shape
len(basic_features)
df.drop(one_hot_encode_col, axis=1,inplace=True)
for f in one_hot_encode_col:

    basic_features.remove(f)
len(basic_features)
df.shape
len(df[basic_features].columns)
len(dummy_df.columns)
df = pd.concat([df[basic_features], dummy_df], axis=1)
del dummy_df

gc.collect()
df.shape

df.head()

df.describe()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix
X = df.drop('TARGET',axis=1)

y = df['TARGET']

print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=27)

print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)


dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

print(classification_report(y_test,predictions))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=27)

print(X_train.shape[1])

print(X_train.shape[0])
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

print(classification_report(y_test,predictions))


rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)



rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))

print(classification_report(y_test,rfc_pred))
pd.value_counts(y_train).plot.bar()

plt.title('histogram')

plt.xlabel('TARGET')

plt.ylabel('Frequency')

df['TARGET'].value_counts()
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=27)

X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

pd.value_counts(y_train_sm).plot.bar()

plt.title('histogram')

plt.xlabel('TARGET')

plt.ylabel('Frequency')



print("Number transactions X_train dataset: ", X_train_sm.shape)

print("Number transactions y_train dataset: ", y_train_sm.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
dtree.fit(X_train_sm,y_train_sm)

predictions = dtree.predict(X_test)

print(classification_report(y_test,predictions))
rfc.fit(X_train_sm, y_train_sm)



rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))

print(classification_report(y_test,rfc_pred))
#Standardization

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_std=sc.fit_transform(X_train)

X_test_std=sc.transform(X_test)

df_x = pd.DataFrame(X_test_std)

df_x.head()
sm = SMOTE(random_state=27)

X_train_sm, y_train_sm = sm.fit_sample(X_train_std, y_train)

pd.value_counts(y_train_sm).plot.bar()

plt.title('histogram')

plt.xlabel('TARGET')

plt.ylabel('Frequency')
dtree.fit(X_train_sm,y_train_sm)

predictions = dtree.predict(X_test_std)

print(classification_report(y_test,predictions))
rfc.fit(X_train_sm, y_train_sm)



rfc_pred = rfc.predict(X_test_std)

print(confusion_matrix(y_test,rfc_pred))

print(classification_report(y_test,rfc_pred))
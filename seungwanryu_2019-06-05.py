# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 



# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('../input/application_train.csv')

print('Training data shape: ', app_train.shape)

app_train.head()
app_test = pd.read_csv('../input/application_test.csv')

print('Testing data shape: ', app_test.shape)

app_test.head()
app_train['TARGET'].value_counts()
app_train.dtypes.value_counts()
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
le = LabelEncoder()

le_count = 0
for col in app_train:

    if app_train[col].dtype == 'object':

        if len(list(app_train[col].unique())) <= 2:

            le.fit(app_train[col])

            app_train[col] = le.transform(app_train[col])

            app_test[col] = le.transform(app_test[col])

            

            le_count += 1
app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)
app_train.shape
app_test.shape
train_labels = app_train['TARGET']



app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
app_train['TARGET'] = train_labels
print(app_train.shape)

print(app_test.shape)
(app_train['DAYS_BIRTH'] / -365).describe()
app_train['DAYS_EMPLOYED'].describe()
app_train['DAYS_EMPLOYES_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == 365243

app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

print(app_test['DAYS_EMPLOYED_ANOM'].sum(), len(app_test))
correlations = app_train.corr()['TARGET'].sort_values()

print(correlations.tail(15), correlations.head(15))
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
plt.style.use('fivethirtyeight')

plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)

plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
age_data = app_train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365



age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_data.head()
age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups
plt.figure(figsize = (8, 8))

plt.bar(age_groups.index.astype(str), 100*age_groups['TARGET'])

plt.xticks(rotation = 75);

plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay(%)')

plt.title('Failure to Repay by age Group')
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

ext_data_corrs = ext_data.corr()

ext_data_corrs
poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]

poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy = 'median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])
poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(poly_features)



poly_features = poly_transformer.transform(poly_features)

poly_features_test = poly_transformer.transform(poly_features_test)
poly_features.shape
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
poly_features = pd.DataFrame(poly_features, 

                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

poly_features['TARGET'] = poly_target



poly_corrs = poly_features.corr()['TARGET'].sort_values()
print(poly_corrs.head(10))

print(poly_corrs.tail(10))
poly_features_test = pd.DataFrame(poly_features_test, 

                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 

                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))



poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']

app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')



poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')



app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)
print(app_train_poly.shape)

print(app_test_poly.shape)
app_train_domain = app_train.copy()

app_test_domain = app_test.copy()



app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']

app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']

app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']

app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
from sklearn.preprocessing import MinMaxScaler, Imputer
if 'TARGET' in app_train:

    train = app_train.drop(columns = ['TARGET'])

else:

    train = app_train.copy()
features = list(train.columns)

test = app_test.copy()

imputer = Imputer(strategy = 'median')

scaler = MinMaxScaler(feature_range = (0, 1))

imputer.fit(train)
train = imputer.transform(train)

test = imputer.transform(app_test)
scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)
print(train.shape)

print(test.shape)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C = 0.0001)

log_reg.fit(train, train_labels)
log_reg_pred = log_reg.predict_proba(test)[:, 1]
submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred



submit.head()
submit.to_csv('log_reg_baseline.csv', index = False)
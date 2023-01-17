%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path = '../input/'
! ls {path}
# Importing the dataset
train_df = pd.read_csv('../input/application_train.csv')
test_df = pd.read_csv('../input/application_test.csv')
train = train_df.copy()
test = test_df.copy()
print(train.shape)
print(test.shape)
train.head()
train['TARGET'].value_counts().plot.bar()
train.describe()
train.describe(exclude=np.number)
train.dtypes.value_counts()
def missing_values_table(data):
    total_missing = data.isnull().sum().sort_values(ascending=False)
    percentage_missing = (100*data.isnull().sum()/len(data)).sort_values(ascending=False)
    missing_table = pd.DataFrame({'missing values':total_missing,'% missing':percentage_missing})
    return missing_table
missing_values = missing_values_table(train)
missing_values.head(20)
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in train.columns:
    if train[c].dtype == 'object':
        if len(list(train[c].unique())) <= 2:
            train[c] = le.fit_transform(train[c])
            test[c] = le.transform(test[c])
print(train.shape)
print(test.shape)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape)
print(test.shape)
train_label = train['TARGET']
# Align the training and testing data, keep only columns present in both dataframes
train, test = train.align(test, join = 'inner', axis = 1)

# Add the target back in
train['TARGET'] = train_label

print(train.shape)
print(test.shape)
(train['DAYS_BIRTH']/-365).describe()
(train['DAYS_EMPLOYED']).describe()
train['DAYS_EMPLOYED'].plot.hist()
outlier = train[train['DAYS_EMPLOYED'] == 365243]
non_out = train[train['DAYS_EMPLOYED'] != 365243]
print('The non-outlier default on %0.2f%% of loans' % (100 * non_out['TARGET'].mean()))
print('The outlier default on %0.2f%% of loans' % (100 * outlier['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(outlier))
train['DAYS_EMPLOYED_OUTLIER'] = train['DAYS_EMPLOYED'] == 365243
train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
train['DAYS_EMPLOYED'].plot.hist()
test['DAYS_EMPLOYED_OUTLIER'] = test['DAYS_EMPLOYED'] == 365243
test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
correlation = train.corr()['TARGET'].sort_values()
print(correlation.head(15))
print(correlation.tail(15))
train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])
train['DAYS_BIRTH'].corr(train['TARGET'])
plt.style.use('fivethirtyeight')
plt.hist(train['DAYS_BIRTH']/365,bins = 25,edgecolor='k')
plt.xlabel('AGE (years)')
plt.ylabel('COUNT')
plt.figure(figsize=(10,8))
sns.kdeplot(train.loc[train['TARGET']==0,'DAYS_BIRTH']/365,label='target==0')
sns.kdeplot(train.loc[train['TARGET']==1,'DAYS_BIRTH']/365,label='target==1')
age_data = train[['TARGET','DAYS_BIRTH']]
age_data['YEAR_BIRTH'] = age_data['DAYS_BIRTH'] / 365
# BINS
age_data['YEAR_BINNED'] = pd.cut(age_data['YEAR_BIRTH'],bins=np.linspace(20,70,11))
age_data.head(10)
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEAR_BINNED').mean()
age_groups
age_groups.TARGET.plot.bar(figsize=(8,8))
plt.title('Failure to repay the loan')
plt.xlabel('age groups (years)')
plt.ylabel('faulure to repay %')
# Extract the EXT_SOURCE variables and show correlations
ext_data = train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs
plt.figure(figsize=(8,6))
sns.heatmap(ext_data_corrs,cmap=plt.cm.YlGnBu_r,annot=True,vmin=.2,vmax=.6)
plt.figure(figsize = (10,12))
for i , source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
    plt.subplot(3,1,i+1)
    sns.kdeplot(train.loc[train['TARGET']==0, source],label='TARGET = 0')
    sns.kdeplot(train.loc[train['TARGET']==1, source],label='TARGET = 0')
    plt.title('Destribution of {} by Target value'.format(source))
    plt.xlabel('{}'.format(source))
    plt.ylabel('Density')
plt.tight_layout(h_pad=2)
# Make a new dataframe for polynomial features
poly_features = train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# imputer for handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])

# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
# Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                     'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
# Create a dataframe of the features 
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))
# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = train['SK_ID_CURR']
train_poly = train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
test_poly = test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
train_poly, test_poly = train_poly.align(test_poly, join = 'inner', axis = 1)

# Print out the new shapes
print('Training data with polynomial features shape: ', train_poly.shape)
print('Testing data with polynomial features shape:  ', test_poly.shape)
train_domain = train.copy()
test_domain = test.copy()
train_domain['CREDIT_INCOME_PERCENT'] = train_domain['AMT_CREDIT'] / train_domain['AMT_INCOME_TOTAL']
train_domain['ANNUITY_INCOME_PERCENT'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_INCOME_TOTAL']
train_domain['CREDIT_TERM'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_CREDIT']
train_domain['DAYS_EMPLOYED_PERCENT'] = train_domain['DAYS_EMPLOYED'] / train_domain['DAYS_BIRTH']
test_domain['CREDIT_INCOME_PERCENT'] = test_domain['AMT_CREDIT'] / test_domain['AMT_INCOME_TOTAL']
test_domain['ANNUITY_INCOME_PERCENT'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_INCOME_TOTAL']
test_domain['CREDIT_TERM'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_CREDIT']
test_domain['DAYS_EMPLOYED_PERCENT'] = test_domain['DAYS_EMPLOYED'] / test_domain['DAYS_BIRTH']
plt.figure(figsize = (8, 6))
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 0, 'CREDIT_INCOME_PERCENT'], label = 'target == 0')
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 1, 'CREDIT_INCOME_PERCENT'], label = 'target == 1')
plt.figure(figsize = (8, 6))
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 0, 'ANNUITY_INCOME_PERCENT'], label = 'target == 0')
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 1, 'ANNUITY_INCOME_PERCENT'], label = 'target == 1')
plt.figure(figsize = (8, 6))
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 0, 'CREDIT_TERM'], label = 'target == 0')
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 1, 'CREDIT_TERM'], label = 'target == 1')
plt.figure(figsize = (8, 6))
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 0, 'DAYS_EMPLOYED_PERCENT'], label = 'target == 0')
sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 1, 'DAYS_EMPLOYED_PERCENT'], label = 'target == 1')
from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in train:
    train = train.drop(columns = ['TARGET'])
else:
    train = train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
testing = test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)
# Transform both training and testing data
train = imputer.transform(train)
testing = imputer.transform(test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
testing = scaler.transform(testing)

print('Training data shape: ', train.shape)
print('Testing data shape: ', testing.shape)
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, train_label)
# Make predictions
y_pred = log_reg.predict_proba(testing)[:, 1]
# # Submission dataframe
# submit = test[['SK_ID_CURR']]
# submit['TARGET'] = y_pred

# submit.head(),submit.shape
# # Save the submission to a csv file
# submit.to_csv('log_reg_baseline.csv', index = False)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1,
                                       n_jobs = -1)
# Train on the training data
random_forest.fit(train, train_label)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = random_forest.predict_proba(testing)[:, 1]
# # Make a submission dataframe
# submit = test[['SK_ID_CURR']]
# submit['TARGET'] = predictions

# # Save the submission dataframe
# submit.to_csv('random_forest_baseline.csv', index = False)
poly_features_names = list(train_poly.columns)

# Impute the polynomial features
imputer = Imputer(strategy = 'median')

poly_features = imputer.fit_transform(train_poly)
poly_features_test = imputer.transform(test_poly)

# Scale the polynomial features
scaler = MinMaxScaler(feature_range = (0, 1))

poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

random_forest_poly = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
# Train on the training data
random_forest_poly.fit(poly_features, train_label)

# Make predictions on the test data
predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]
# Make a submission dataframe
submit = test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline_engineered.csv', index = False)
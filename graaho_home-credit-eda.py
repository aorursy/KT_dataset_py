# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# numpy and pandas for data manipulation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# File system manangement

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns
# List files available

print(os.listdir("../input/home-credit-default-risk"))
dir_name = "../input/home-credit-default-risk"
app_train = pd.read_csv(os.path.join(dir_name,'application_train.csv'))

print("Training data shape: ", app_train.shape)

app_train.head()
print("Application train column names: ")

print("------------------------------")

for _ in app_train.columns.values:

    print(_, end=' , ')
app_test = pd.read_csv(os.path.join(dir_name,'application_test.csv'))

print("Test data shape: ", app_test.shape)

app_test.head()
print("Application test column names: ")

print("------------------------------")

for _ in app_test.columns.values:

    print(_, end=' , ')
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist()
app_train.isnull().sum()
def missing_values_table(df):

    # Total missing values

    missing_values = df.isnull().sum()

#     print(missing_values)

    # Percentage of missing values

    missing_values_percentage = 100 * missing_values / len(df)

#     print(missing_values_percentage)

    missing_values_table = pd.concat([missing_values,missing_values_percentage],axis=1)

    print("Missing values and percentage shape: ",missing_values_table.shape)

#     print(missing_values_table)

    # Renaming the column names

    missing_values_rename_columns = missing_values_table.rename(columns = {0: 'Missing Values', 1: 'Missing % of total values'})

#     print(missing_values_rename_columns)

    # Sorting the table by percentage of missing descendents

    missing_values_rename_columns = missing_values_rename_columns[missing_values_rename_columns.iloc[:,1] != 0].sort_values('Missing % of total values', ascending = False).round(1)

#     print(missing_values_rename_columns)

    print(f'This dataframe has {df.shape[1]} columns. There are {missing_values_rename_columns.shape[0]} columns that have missing values.')

    

    return missing_values_rename_columns

    
app_train_missing_values = missing_values_table(app_train)

app_train_missing_values.head(10)
app_train.dtypes.value_counts()
app_train.select_dtypes('object').apply(pd.Series.nunique,axis = 0) # here axis = 1 means row in the dataframe and 0 means column
#Create a label encoder object

def label_encoder_train_test(train, test):



    le = LabelEncoder()

    le_count = 0



    # Iterate through the columns

    for col in train:

        if train[col].dtype == 'object':

            if len(list(train[col].unique())) <= 2:

                le.fit(train[col])

                train[col] = le.transform(train[col])

                test[col] = le.transform(test[col])

                le_count += 1



    print(f'{le_count} columns were encoded.')
label_encoder_train_test(app_train,app_test)
# One hot encoding for more than two categorical values

def one_hot_encoding_train_test(train, test):

    train = pd.get_dummies(train)

    test = pd.get_dummies(test)

    

    print(f"Training feature shape: {train.shape}")

    print(f"Testing feature shape: {test.shape}")
one_hot_encoding_train_test(app_train,app_test)
train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes

app_train, app_test = app_train.align(app_test, join='inner', axis=1) # axis = 1 for column based alignment

app_train['TARGET'] = train_labels



print(f'Training feature shape: {app_train.shape}')

print(f'Testing feature shape: {app_test.shape}')

# The numbers in the DAYS_BIRTH column are negative because they are recorded relative to the current loan application

age = (app_train['DAYS_BIRTH']/ -365).describe()

age
app_train['DAYS_EMPLOYED'].describe()
app_train.DAYS_EMPLOYED.plot.hist(title='Days employment histogram');

plt.xlabel('Days employment');
anomalous = app_train[app_train.DAYS_EMPLOYED == 365243]

non_anomalous = app_train[app_train.DAYS_EMPLOYED != 365243]



print(f'The non-anomalous default on {non_anomalous.TARGET.mean() * 100}% of loans')

print(f'The anomalous default on {anomalous.TARGET.mean() * 100}% of loans')

print(f'There are {len(anomalous)} anomalous days of employment')
app_train.DAYS_EMPLOYED_ANOM = app_train.DAYS_EMPLOYED == 365243

app_train.DAYS_EMPLOYED.replace({365243: np.nan}, inplace = True)

app_train.DAYS_EMPLOYED.plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
app_test.DAYS_EMPLOYED_ANOM = app_test.DAYS_EMPLOYED == 365243

app_test.DAYS_EMPLOYED.replace({365243: np.nan}, inplace = True)

print(f'There are {app_test.DAYS_EMPLOYED_ANOM.sum()} anomalies in the test data out of {len(app_test)} entries.')
correlations = app_train.corr()['TARGET'].sort_values()



print('Most positive correlations:\n', correlations.tail(15))

print('\nMost negative correlations:\n', correlations.head(15))
app_train.DAYS_BIRTH = abs(app_train.DAYS_BIRTH)

app_train.DAYS_BIRTH.corr(app_train.TARGET)
plt.style.use('fivethirtyeight')

plt.hist(app_train['DAYS_BIRTH']/365, edgecolor='k', bins=25)

plt.title('Age of client');plt.xlabel('Age (years)');plt.ylabel('Count')
plt.figure(figsize = (10, 8))

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
# Age information into a separate dataframe

age_data = app_train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365



# Bin the age data

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_data.head(10)

# ( = "excluding", ] = "including"
type(age_data['YEARS_BINNED']) #special case of excluding and including
# Group by the bin and calculate averages



age_groups  = age_data.groupby('YEARS_BINNED').mean()

age_groups
plt.figure(figsize = (8, 8))



# Graph the age bins and the average of the target as a bar plot

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])



# Plot labeling

plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')

plt.title('Failure to Repay by Age Group');
# Extract the EXT_SOURCE variables and show correlations

ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

ext_data_corrs = ext_data.corr()

ext_data_corrs

plt.figure(figsize = (8, 6))



# Heatmap of correlations

sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
plt.figure(figsize = (10, 12))



# iterate through the sources

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):

    

    # create a new subplot for each source

    plt.subplot(3, 1, i + 1)

    # plot repaid loans

    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')

    # plot loans that were not repaid

    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')

    

    # Label the plots

    plt.title('Distribution of %s by Target Value' % source)

    plt.xlabel('%s' % source); plt.ylabel('Density');

    

plt.tight_layout(h_pad = 2.5)
# Copy the data for plotting

plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()



# Add in the age of the client in years

plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']



# Drop na values and limit to first 100000 rows

plot_data = plot_data.dropna().loc[:100000, :]



# Function to calculate correlation coefficient between two columns

def corr_func(x, y, **kwargs):

    r = np.corrcoef(x, y)[0][1]

    ax = plt.gca()

    ax.annotate("r = {:.2f}".format(r),

                xy=(.2, .8), xycoords=ax.transAxes,

                size = 20)



# Create the pairgrid object

grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,

                    hue = 'TARGET', 

                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])



# Upper is a scatter plot

grid.map_upper(plt.scatter, alpha = 0.2)



# Diagonal is a histogram

grid.map_diag(sns.kdeplot)



# Bottom is density plot

grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);



plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05);
# Make a new dataframe for polynomial features

poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]

poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]



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
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
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

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']

app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')



# Merge polnomial features into testing dataframe

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')



# Align the dataframes

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)



# Print out the new shapes

print('Training data with polynomial features shape: ', app_train_poly.shape)

print('Testing data with polynomial features shape:  ', app_test_poly.shape)
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
plt.figure(figsize = (12, 20))

# iterate through the new features

for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):

    

    # create a new subplot for each source

    plt.subplot(4, 1, i + 1)

    # plot repaid loans

    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')

    # plot loans that were not repaid

    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')

    

    # Label the plots

    plt.title('Distribution of %s by Target Value' % feature)

    plt.xlabel('%s' % feature); plt.ylabel('Density');

    

plt.tight_layout(h_pad = 2.5)
from sklearn.preprocessing import MinMaxScaler, Imputer



# Drop the target from the training data

if 'TARGET' in app_train:

    train = app_train.drop(columns = ['TARGET'])

else:

    train = app_train.copy()

    

# Feature names

features = list(train.columns)



# Copy of the testing data

test = app_test.copy()



# Median imputation of missing values

imputer = Imputer(strategy = 'median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range = (0, 1))



# Fit on the training data

imputer.fit(train)



# Transform both training and testing data

train = imputer.transform(train)

test = imputer.transform(app_test)



# Repeat with the scaler

scaler.fit(train)

train = scaler.transform(train)

test = scaler.transform(test)



print('Training data shape: ', train.shape)

print('Testing data shape: ', test.shape)
from sklearn.linear_model import LogisticRegression



# Make the model with the specified regularization parameter

log_reg = LogisticRegression(C = 0.0001)



# Train on the training data

log_reg.fit(train, train_labels)
log_reg_pred = log_reg.predict_proba(test)[:, 1]
submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred



submit.head()
submit.to_csv('log_reg_baseline.csv', index = False)
from sklearn.ensemble import RandomForestClassifier



# Make the random forest classifier

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
# Train on the training data

random_forest.fit(train, train_labels)



# Extract feature importances

feature_importance_values = random_forest.feature_importances_

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})



# Make predictions on the test data

predictions = random_forest.predict_proba(test)[:, 1]
# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest_baseline.csv', index = False)
poly_features_names = list(app_train_poly.columns)



# Impute the polynomial features

imputer = Imputer(strategy = 'median')



poly_features = imputer.fit_transform(app_train_poly)

poly_features_test = imputer.transform(app_test_poly)



# Scale the polynomial features

scaler = MinMaxScaler(feature_range = (0, 1))



poly_features = scaler.fit_transform(poly_features)

poly_features_test = scaler.transform(poly_features_test)



random_forest_poly = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
# Train on the training data

random_forest_poly.fit(poly_features, train_labels)



# Make predictions on the test data

predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]

# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest_baseline_engineered.csv', index = False)
app_train_domain = app_train_domain.drop(columns = 'TARGET')



domain_features_names = list(app_train_domain.columns)



# Impute the domainnomial features

imputer = Imputer(strategy = 'median')



domain_features = imputer.fit_transform(app_train_domain)

domain_features_test = imputer.transform(app_test_domain)



# Scale the domainnomial features

scaler = MinMaxScaler(feature_range = (0, 1))



domain_features = scaler.fit_transform(domain_features)

domain_features_test = scaler.transform(domain_features_test)



random_forest_domain = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)



# Train on the training data

random_forest_domain.fit(domain_features, train_labels)



# Extract feature importances

feature_importance_values_domain = random_forest_domain.feature_importances_

feature_importances_domain = pd.DataFrame({'feature': domain_features_names, 'importance': feature_importance_values_domain})



# Make predictions on the test data

predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]
# Make a submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest_baseline_domain.csv', index = False)
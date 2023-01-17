# numpy and pandas for data manipulation 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder



# File system management

import os



# Suppress warnings

import warnings

warnings.filterwarnings('ignore')



# matplotlib.and seaborn for visualization

import matplotlib.pyplot as plt

import seaborn as sns
# List available files

print(os.listdir('../input/'))
# Read training data

app_train = pd.read_csv('../input/application_train.csv')

print('Training data shape: ', app_train.shape)

app_train.head()
# Read testing data

app_test = pd.read_csv('../input/application_test.csv')

print('Testing data shape:', app_test.shape)

app_test.head()
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();
def missing_values_table(df):

    '''

    Function to calculate missing values by column

    Arguments:

        df: dataframe

    Output:

        miss_val_table_renamed_cols: dataframe with missing values

    '''

    

    # Total missing values:

    miss_value = df.isnull().sum()

    

    # Missing values in percent:

    miss_value_pct = 100 * df.isnull().sum() / len(df)

    

    # Make a table with the results

    miss_val_table = pd.concat([miss_value, miss_value_pct], axis=1)

    

    # Rename the columns

    miss_val_table_renamed_cols = miss_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})

    

    # Sort the table by percentage of missing, descending

    miss_val_table_renamed_cols = miss_val_table_renamed_cols[

                                  miss_val_table_renamed_cols.iloc[:, 1] != 0].sort_values('% of Total Values', ascending = False).round(1)

    

    # Print some summary information

    print('Your selected dataframe has ' + str(df.shape[1]) 

          + ' columns.\n')

    print('There are' + str(miss_val_table_renamed_cols.shape[0]) +

          ' columns that have missing values')

    

    return miss_val_table_renamed_cols
missing_values = missing_values_table(app_train)

missing_values.head(20)
# Number of each type of column

app_train.dtypes.value_counts()
# Number of unique entries in each object column

app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in app_train:

    if app_train[col].dtype =='object':

        

        # If 2 or fewer unique categories

        if len(list(app_train[col].unique())) <= 2:

            

            # Train on the training data

            le.fit(app_train[col])

            

            # Transform both training and testing data

            app_train[col] = le.transform(app_train[col])

            app_test[col] = le.transform(app_test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            print(col + ' column was label encoded.')

            

print('%d columns were label encoded.' % le_count)
app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)



print('Training Features shape: ', app_train.shape)

print('Testing Features shape: ', app_test.shape)
# Align training and test data



train_labels = app_train['TARGET']



app_train, app_test = app_train.align(app_test, join='inner', axis=1)



app_train['TARGET'] = train_labels



print('Training Features shape: ', app_train.shape)

print('Testing Features shape: ', app_test.shape)

app_train['DAYS_BIRTH'].describe()
(app_train['DAYS_BIRTH'] / -365.25).describe()
app_train['DAYS_EMPLOYED'].describe()
(app_train['DAYS_EMPLOYED'] / -365.25).describe()
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]

non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]

print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))

print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))

print('There are %d anomalous days of employment' % len(anom))
# Create an anomalous flag olumn

app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243



# Replace the anomalous values with NaN

app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)



app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243

app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)



print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
# Find correlations with the target and sort

correlations = app_train.corr()['TARGET'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
# Convert DAYS_BIRTH column to absolute value

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
plt.style.use('fivethirtyeight')



plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)

plt.title('Age of Client');

plt.xlabel('Age (years)');

plt.ylabel('Count');
plt.figure(figsize=(10, 8))



# KDE plot on loans that were repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 

                          'DAYS_BIRTH'] / 365, label = 'target =0')



# KDE plot on loans which were not repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1,

                          'DAYS_BIRTH'] / 365, label = 'target = 1')



# Labeling of plot

plt.xlabel('Age (years)')

plt.ylabel = 'Density'

plt.title = 'Distribution of Ages'
# Age information into a separate dataframe

age_data = app_train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH']/365



# Bin the age data

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], 

                                  bins=np.linspace(20, 70, num=11))

age_data.head()

# Group by the bind and calculate averages

age_groups = age_data.groupby('YEARS_BINNED').mean()

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
plt.figure(figsize=(8, 6))



# Heatmap of correlation

sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25,

            annot=True, vmax=0.6)

plt.title('Correlation HeatMap')
# Make a new datafrane for polynomial features

poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',

                           'DAYS_BIRTH', 'TARGET']]

poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',

                           'DAYS_BIRTH']]



# Imputer for handling missing values

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')

poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns=['TARGET'])



poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)



from sklearn.preprocessing import PolynomialFeatures



# Create the polynomial object with specified degreee

poly_transformer = PolynomialFeatures(degree= 3)
# Train the polynomial

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
from sklearn.preprocessing import MinMaxScaler, Imputer



# Drop the target from the training data

if 'TARGET' in app_train:

    train = app_train.drop(columns=['TARGET'])

else:

    train = app_train.copy

    

# Feature names

features = list(train.columns)



# Copy of the testing data

test = app_test.copy()



# Median imputation of missing values

imputer = Imputer(strategy='median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range=(0,1))



# Fit on the training data

imputer.fit(train)



# Transform both training and testing data

train = imputer.transform(train)

test = imputer.transform(test)



# Repeat with the scaler

scaler.fit(train)

train  = scaler.transform(train)

test = scaler.transform(test)



print('Training data shape: ', train.shape)

print('Testing data shape : ', test.shape)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(C=0.0001)

log_reg.fit(train, train_labels)
log_red_pred = log_reg.predict_proba(test)[:, 1]
# Submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_red_pred



submit.head()
submit.to_csv('log_reg_baseline.csv', index=False)
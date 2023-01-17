# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder



import os



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns
print(os.listdir("/kaggle/input/home-credit-default-risk/"))
# Training Data

app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

print('Training data shape: ', app_train.shape)

app_train.head()
# Testing data features

app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

print('Testing data shape: ', app_test.shape)

app_test.head()
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();
# 전체 data는 ? 307,511 

print(len(app_train))
def missing_values_table(df):

    # Total missing values

    mis_val = df.isnull().sum()

    

    # Percentage of missing values

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    

    # Make a table with the results

    # axis=1 ; app_train + mis_val 컬럼 + mis_val_percent 컬럼 식으로 추가

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    

    # Rename the columns

    mis_val_table_ren_columns = mis_val_table.rename( columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    

    # Sort the table by percentage of missing descending

    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0 ].sort_values('% of Total Values', ascending=False).round(1)

    

    #print("##")

    #print(mis_val_table_ren_columns.iloc[:,1] )

    #print("##")

    #print("##")

    #print(mis_val_table_ren_columns.iloc[:,1] != 0)

    #print("##")





#    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0 ].sort_values(by='% of Total Values', ascending=False).round(1)

#    mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values(by='% of Total Values', ascending=False).round(1)

    

    # Print some summary information

    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" 

         "There are " + str(mis_val_table_ren_columns.shape[0]) + 

         " columns that have missing values.")

    

    # Return the datafrmae with missing information

    return mis_val_table_ren_columns



# Missing values statistics

missing_values = missing_values_table(app_train)

missing_values.head(20)
# Number of each type of column

app_train.dtypes.value_counts()
# Number of unique classes in each object column

app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Label Encoding 전, 데이타 체크, 122개 column임. 

app_train.shape
for col in app_train:

    if app_train[col].dtypes == 'object':

        print(app_train[col])
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in app_train:

    if app_train[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(app_train[col].unique())) <= 2:

            print(app_train[col])

            # Train on the training data

            le.fit(app_train[col])

            

            # Transform both training and testing data

            app_train[col] = le.transform(app_train[col])

            app_test[col] = le.transform(app_test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
app_train.tail(10)
# one-hot encoding of categorical variables

app_train = pd.get_dummies(app_train)

app_test = pd.get_dummies(app_test)



print('Training Features shape: ', app_train.shape)

print('Testing Features shape: ', app_test.shape)
train_labels = app_train['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)



# Add the target back in

app_train['TARGET'] = train_labels



print('Training Features shape: ', app_train.shape)

print('Testing Features shape: ', app_test.shape)
# - ( negative ) days : 현재 대출 신청 기록. 과거 대출은 - 로 표시 

app_train['DAYS_BIRTH'].describe()
# 날짜가 크기 때문에, 년 단위로 변경

( app_train['DAYS_BIRTH'] / 365).describe()
# positive로 표시

( app_train['DAYS_BIRTH'] / 365 * -1).describe()
app_train['DAYS_EMPLOYED'].describe()
# DAYS_EMPLOYED : How many days before the application the person started current employment, 일 하기 시작한 날?

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]

non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]

print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))

print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))

print('There are %d anomalous days of employment' % len(anom))
# Create an anomalous flag column

app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243



# Replace the anomalous values with nan

app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)



app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
# fill to app_test DataFrame

# 변칙적인 값이 어느 정도 중요하다고 판단하는 경우, 변친 값을 판단하는 boolean 컬럼을 새로 만들어 둔다. 

# 그리고, 원래 컬럼에 변칙적인 값이 있으면 그 값은 NaN 으로 채워 둔다. 

app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == 365243

app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)



print('There are %d anomalies in the test data out of %d entries' % (app_test['DAYS_EMPLOYED_ANOM'].sum(), len(app_test)))
# Find correlations with the target and sort

correlations = app_train.corr()['TARGET'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(5))
# Find the correlation of the positive days since birth and target



app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
# Make a histogra of age



# Set the style of plots

plt.style.use('fivethirtyeight')



# Plot the distribution of ages in years

plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)

plt.title('Age of Client')

plt.xlabel('Age (years)')

plt.ylabel('Count')
plt.figure(figsize = (10,8))



# KDE plot of loans that were repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')



# KDE plot of loans which were not repaid on time

sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')



# Labeling of plot

plt.xlabel('Age (years)')

plt.ylabel('Density')

plt.title('Distribution of Ages')
# Age information into a separate dataframe

age_data = app_train[['TARGET', 'DAYS_BIRTH']]

age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365



# Bin the age data

# pd.cut : 실수 값 경계선 지정 function

# bins : 카테고리를 나누는 기준 값

# np.linspace : 20부터 70까지 균등하게 배열을 11개 생성

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

age_data.head(10)
# Group by the bin and calculate average

age_groups = age_data.groupby('YEARS_BINNED').mean()

age_groups
plt.figure(figsize = (8, 8))



# Graph the age bins and the average of the target as a bar plot

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])



# Plot labeling

plt.xticks(rotation = 75)

plt.xlabel('Age Group (years)')

plt.ylabel('Failure to Repay (%)')

plt.title('Failure to Repay by Age Group')
# Extract the EXT_SOURCE variables and show correlations

ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

ext_data_corrs = ext_data.corr()

ext_data_corrs
# 숫자보다는 heatmap

plt.figure(figsize = (8,6))



# Heatmap of correlation

sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heapmap')
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

    plt.title('Distribution of %s by Target Value ' % source)

    plt.xlabel('%s' % source)

    plt.ylabel('Density')

    

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



# imputer for handling missing values ( strategy:median 의미가???? )

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy = 'median')  



poly_target = poly_features['TARGET']



poly_features = poly_features.drop(columns = ['TARGET'])



# Need to impute missing values

poly_features = imputer.fit_transform(poly_features)

poly_features_test = imputer.transform(poly_features_test)



from sklearn.preprocessing import PolynomialFeatures



# Create the polynomial object with specified degree ( degree:3, 3차수 x^3 )

poly_transformer = PolynomialFeatures(degree = 3)
# Train the polynomial features

poly_transformer.fit(poly_features)



# Transform the features

poly_features = poly_transformer.transform(poly_features)

poly_features_test = poly_transformer.transform(poly_features_test)

print('Polynomial Features shape: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
# Create a dataframe of the features

poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))



# Add in the target

poly_features['TARGET'] = poly_target



# Find the correlations with the target

poly_corrs = poly_features.corr()['TARGET'].sort_values()



# Display most negative and most positive

print(poly_corrs.head(10))

print(poly_corrs.tail(5))
# Put test features into dataframe

poly_features_test = pd.DataFrame(poly_features_test, columns = poly_transformer.get_feature_names

                                  (['EXT_SOURCE_1', 'EXT_SOURCE_2',

                                   'EXT_SOURCE_3', 'DAYS_BIRTH']))



# Merge polynomial features into training dataframe

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']

app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')



# Merge polynomial features into testing dataframe

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')



# Align the dataframes

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)



# Print out the new shapes

print('Training data with polynomial features shape : ', app_train_poly.shape)

print('Testing data with polynomial features shape : ', app_test_poly.shape)
app_train_domain = app_train.copy()

app_test_domain = app_test.copy()



# AMT_CREDIT : Credit amount of the loan

# AMT_INCOME_TOTAL : Income of the client

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']



# AMT_ANNUITY ; Loan annuity ( 대출 연금 )

# AMT_INCOME_TOTAL : Income of the client 

app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']



# AMT_ANNUITY ; Loan annuity ( 대출 연금 )

# AMT_CREDIT ; Credit amount of the loan

app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']



# DAYS_EMPLOYED ; How many days before the application the person started current employment

# DAYS_BIRTH ; Client's age in days at the time of application

app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']

app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']

app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
plt.figure(figsize = (12, 20))



# iterate through the new features

for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):

    

    # create a new subplot for each source, 4x1, index

    plt.subplot(4, 1, i + 1)

    

    # plot repaid loans

    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')

    

    # plot loans that were not repaid

    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')

    

    # Label the plots

    plt.title('Distribution of %s by Target Value' % feature)

    plt.xlabel('%s' % feature)

    plt.ylabel('Density')

    

# h_pad ; 인접한 pad 높이 간격

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



# Mean(평균), median(중앙값), mode(최빈값)

# Median(중앙값) imputation of missing values

imputer = Imputer(strategy = 'median')



# Scale each feature to 0-1

scaler = MinMaxScaler(feature_range = (0,1))



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

print(train)
from sklearn.linear_model import LogisticRegression



# Make the model with the specified regularization parameter

log_reg = LogisticRegression(C = 0.0001)



# Train on the training data

log_reg.fit(train, train_labels)
# Make prediction

# Make sure to select the second column only

# predict() : output이 0 or 1

# predict_proba() : 1의 probability

log_reg_pred = log_reg.predict_proba(test)[:,1]
# Submission dataframe

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = log_reg_pred



submit.head()
# Save the submission to a csv file

submit.to_csv('log_reg_baseline.csv', index = False)
from sklearn.ensemble import RandomForestClassifier



# Make the random forest classifier

random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
# Train on the training data

random_forest.fit(train, train_labels)



# Extract feature importances

# Random Forest 장점은 각 독립 변수 중요도(feature importance)를 계산할 수 있다.

feature_importance_values = random_forest.feature_importances_



# 이 문서 거의 마지막 모델 "Model Interpretation : Feature Importances" 에서 사용

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

poly_features_test = imputer.fit_transform(app_test_poly)



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

scaler = MinMaxScaler(feature_range = (0,1))



domain_features = scaler.fit_transform(domain_features)

domain_features_test = scaler.transform(domain_features_test)



random_forest_domain = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)



# Train on the training data

random_forest_domain.fit(domain_features, train_labels)



# Extract feature importances

feature_importance_values_domain = random_forest_domain.feature_importances_

feature_importance_domain = pd.DataFrame({'feature': domain_features_names, 'importance': feature_importance_values_domain})



# Make predictions on the test data

predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]
submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = predictions



# Save the submission dataframe

submit.to_csv('random_forest_baseline_domain.csv', index = False)
def plot_feature_importances(df):

    """

    Plot importances returned by a model. This can work with any measure of

    feature importance provided that higher importance is better. 

    

    Args:

        df (dataframe): feature importances. Must have the features in a column

        called `features` and the importances in a column called `importance

        

    Returns:

        shows a plot of the 15 most importance features

        

        df (dataframe): feature importances sorted by importance (highest to lowest) 

        with a column for normalized importance

        """

    

    # Sort features according to importance

    df = df.sort_values('importance', ascending = False).reset_index()

    

    # Normalize the feature importances to add up to one

    df['importance_normalized'] = df['importance'] / df['importance'].sum()



    # Make a horizontal bar chart of feature importances

    plt.figure(figsize = (10, 6))

    ax = plt.subplot()

    

    # Need to reverse the index to plot most important on top

    ax.barh(list(reversed(list(df.index[:15]))), 

            df['importance_normalized'].head(15), 

            align = 'center', edgecolor = 'k')

    

    # Set the yticks and labels

    ax.set_yticks(list(reversed(list(df.index[:15]))))

    ax.set_yticklabels(df['feature'].head(15))

    

    # Plot labeling

    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')

    plt.show()

    

    return df

# Show the feature importances for the default features

# feature_importances : Randome Forest 후 나온 중요도 변수를 train pandas 데이타에 추가함. 

feature_importances_sorted = plot_feature_importances(feature_importances)

feature_importance_domain_sorted = plot_feature_importances(feature_importance_domain)
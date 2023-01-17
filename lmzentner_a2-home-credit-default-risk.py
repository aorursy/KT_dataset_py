## Reference Will Koehrsen's guide - https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

#imports

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

#List of files
print(os.listdir("../input/"))

#Training data
app_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()

#Testing data features
app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()

#Exploratory Data Analysis
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();

#Function to calculate missing data
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : 'Percent of Total Values'})
    
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            'Percent of Total Values', ascending=False).round(1)
    
        print ("Your selected df has " + str(df.shape[1]) + " columns.\n"
               "There are " + str(mis_val_table_ren_columns.shape[0]) + 
                 " columns that have missing values")
        return mis_val_table_ren_columns

missing_values = missing_values_table(app_train)
missing_values.head(20)

#number of each type of column
app_train.dtypes.value_counts()

#number of unique classes
app_train.select_dtypes('object').apply(pd.Series.nunique, axis =0)

#LabelEncoder
le = LabelEncoder()
le_count = 0

#Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        #if <= 2 unique categories, train on the training data
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            #Transform data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            le_count += 1
print ('%d columns were label encoded' % le_count)

#OneHotEncoder
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training features shape: ', app_train.shape)
print('Testing features shape: ', app_test.shape)

#Aligning data
train_labels = app_train['TARGET']
#keep only columns present in both df
app_train, app_test = app_train.align(app_test, join = 'inner', axis =1)
#Add target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

#Fix ages
(app_train['DAYS_BIRTH'] / -365).describe()

#Fix days employed
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

#Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

#Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

#Find correlations w/ target & sort
correlations = app_train.corr()['TARGET'].sort_values()

#Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

#Correlation of birth date and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])

#Plot age distribution 
plt.style.use('fivethirtyeight')
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.figure(figsize = (10, 8))

#KDE plot of loans repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

#KDE plot of loans not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

#Labeling plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');

#Age information df
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

#Bin age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)

#Group and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups

#graph age bins versus average of target 
plt.figure(figsize = (8, 8))
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');

from sklearn.preprocessing import MinMaxScaler, Imputer

#Drop target from training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()
    
#Feature names
features = list(train.columns)

#Copy the testing data
test = app_test.copy()

#Median imputation of missing values
imputer = Imputer(strategy = 'median')

#Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

#Fit on the training data
imputer.fit(train)

#Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

#Repeat with scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

#RandomForestClassifer
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
random_forest.fit(train, train_labels)

feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance' : feature_importance_values})
predictions = random_forest.predict_proba(test)[:,1]

#Feature importance graph
def plot_feature_importances(df):
    #Sort
    df = df.sort_values('importance', ascending = False).reset_index()
    
    #Normalize 
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    #Chart
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    #Reverse index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    #Label & Ticks
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    return df
feature_importances_sorted = plot_feature_importances(feature_importances)

#Bureau data
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()

#Group by client id, count number of previous loans
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()

#Add to training df
train = pd.read_csv('../input/application_train.csv')
train = train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')

#Fill in missing values 
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.head()

#OneHotEncoder
categorical = pd.get_dummies(bureau.select_dtypes('object'))
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
categorical.head()

#Find correlations w/ target & sort
correlations = train.corr()['TARGET'].sort_values()

#Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

#Submission
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions
submit.head()
submit.to_csv('A2_RFC.csv', index = False)

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

plt.hist(app_train['DAYS_BIRTH']/365, edgecolor='k', bins=25);

plt.title('Age of client');plt.xlabel('Age (years)');plt.ylabel('Count');
## TODO: What KDE actually depicts here? Find out.



plt.figure(figsize = (10,8))



# KDE plot of loans that were repaid on time

sns.kdeplot(app_train.loc[app_train.TARGET == 0, 'DAYS_BIRTH']/365, label='target == 0')

#KDE plot of loans that were not repaid om time

sns.kdeplot()
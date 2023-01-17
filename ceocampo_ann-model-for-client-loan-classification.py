# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
df = pd.read_csv('../input/lending-club-loan-data/lending_club_loan_two.csv')
df.info()
sns.countplot(df['loan_status'])

plt.show()
plt.figure(figsize=(12,4))

plt.xlim(0,45000)

sns.distplot(df['loan_amnt'], kde=False)

plt.show()
# Examining the correlation of all numerical features to one another

df.corr()
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.ylim(12,0)

plt.show()
sns.scatterplot(x='installment', y='loan_amnt', data=df, alpha=0.1, edgecolor=None)

plt.show()
sns.boxplot(x='loan_status', y='loan_amnt', data=df)

plt.show()
df.groupby('loan_status')['loan_amnt'].describe()
grades = sorted(df['grade'].unique())

sub_grades = sorted(df['sub_grade'].unique())
sns.countplot(x='grade', data=df, hue=df['loan_status'], order=grades, palette='seismic')

plt.show()
plt.figure(figsize=(14,4))

sns.countplot(x='sub_grade', data=df, alpha=0.8, order=sub_grades, palette='seismic')

plt.show()
plt.figure(figsize=(14,4))

sns.countplot(x='sub_grade', data=df, alpha=0.8, hue='loan_status', order=sub_grades, palette='seismic')

plt.show()
df_f_g = df[(df['grade'] == 'F') | (df['grade'] == 'G')]

FG_grades = sorted(df_f_g['sub_grade'].unique())
plt.figure(figsize=(12,4))

sns.countplot(x='sub_grade', data=df_f_g, hue='loan_status', order=FG_grades, palette='seismic')

plt.show()
# Mapping target to binary

df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
df[['loan_repaid', 'loan_status']]
df.corr()['loan_repaid'].sort_values()
plt.figure(figsize=(10,6))

df.corr()['loan_repaid'].drop('loan_repaid').sort_values().plot(kind='barh')

plt.show()
df.head()
len(df)
df.isnull().sum()
# Percentage of missing values per feature

df.isnull().sum() / len(df) * 100
df['emp_title'].nunique()
df['emp_title'].value_counts()
# Dropping 'emp_title' feature

df = df.drop('emp_title', axis=1)
list(df['emp_length'].unique())
sorted_emp_length = ['< 1 year',

 '1 year',

 '2 years',

 '3 years',

 '4 years',

 '5 years',

 '6 years',

 '7 years',

 '8 years',

 '9 years',

 '10+ years']
plt.figure(figsize=(12,4))

sns.countplot(x=df['emp_length'], order=sorted_emp_length)

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(x='emp_length', data=df, order=sorted_emp_length, hue='loan_status')

plt.show()
default_count = df[df['loan_repaid'] == 0].groupby('emp_length').count()['loan_repaid']

total_count = df.groupby('emp_length').count()['loan_repaid']

percent_default = default_count / total_count
emp_length_series = pd.Series(percent_default)

emp_length_series
emp_length_series.index.values
plt.figure(figsize=(10,4))

sns.barplot(x=emp_length_series.index, y=emp_length_series.values, order=sorted_emp_length)

plt.show()
# Dropping 'emp_length' feature.

df = df.drop('emp_length', axis=1)
# Remaining null values to deal with

df.isnull().sum()
df[['purpose', 'title']]
# Dropping 'title' feature

df = df.drop('title', axis=1)
# Investigating 'mort_acc' feature

df['mort_acc'].value_counts()
# Investigating 'mort_acc' correlation with other features.

df.corr()['mort_acc'].sort_values()
total_acc_avg = df.groupby('total_acc')['mort_acc'].mean()

total_acc_avg
sns.scatterplot(x='total_acc', y='mort_acc', data=df, alpha=0.1, edgecolor=None)

plt.show()
# Missing values in 'mort_acc' will be filled in based its average value based on 'total_acc' grouping 

def fill_mort_acc(total_acc, mort_acc):

    if np.isnan(mort_acc):

        return total_acc_avg[total_acc]

    else: 

        return mort_acc
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
# Recheck missing data

df.isnull().sum()
# Dropping the rest of the missing values.

df = df.dropna()
df.isnull().sum()
# Numberical features

df.select_dtypes(exclude=['object']).columns
# Categorical features

df.select_dtypes(include=['object']).columns
df.info()
# Converting 'term' data from str to int

df['term'] = df['term'].map({' 36 months': 36, ' 60 months': 60})
df['term']
# Dropping 'grade' feature

df = df.drop('grade', axis=1)
# Converting 'sub_grade' feature into dummy variables

df_subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
# Concatenating 'sub_grade' dummies to primary dataframe.

df = pd.concat([df, df_subgrade_dummies], axis=1)
# Dropping original 'sub_grade' feature

df = df.drop('sub_grade', axis=1)
df.columns
# Creating more dummy variables for the following features: 'verification_status', 'application_type', 'initial_list_status', 'purpose'

df_more_dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']], drop_first=True)

df_more_dummies.head()
# Concatenating additional dummy variables to primary dataframe

df = pd.concat([df, df_more_dummies], axis=1)
# Dropping the 'verification_status', 'application_type', 'initial_list_status', and 'purpose' features 

df = df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1)
df.columns
df['home_ownership'].value_counts()
# Grouping 'NONE' and 'ANY' with 'OTHER' in the 'home_ownership' feature

df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
# Rechecking 'home_ownership' feature

df['home_ownership'].value_counts()
# Converting 'home_ownership' to dummy variables and adding them to primary dataframe

df_home_dummies = pd.get_dummies(df['home_ownership'], drop_first=True)

df = pd.concat([df, df_home_dummies], axis=1)

df = df.drop('home_ownership', axis=1)

df.columns
# Extracting zipcode from first row

df['address'][0][-5:]
# Creating a new feature 'zipcode'

df['zipcode'] = df['address'].apply(lambda x: x[-5:])
df['zipcode'].value_counts()
# Creating dummy variables for 'zipcode' feature

df_zipcode_dummies = pd.get_dummies(df['zipcode'], drop_first=True)

df = pd.concat([df, df_zipcode_dummies], axis=1)



# Dropping 'address' feature

df = df.drop(['zipcode', 'address'], axis=1)

df.columns
# Issue date would be listed only if a loan was issued. Feature will be dropped to avoid data leakage.

df = df.drop('issue_d', axis=1)
df['earliest_cr_line']
# Extracting year from 'earliest_cr_line' and creating a new features 'year' as an integer

df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: int(x[-4:]))
# Dropping 'earliest_cr_line' feature

df = df.drop('earliest_cr_line', axis=1)
df.columns
# Dropping 'loan_status' label. Target was earlier converted to 'loan_repaid'.

df = df.drop('loan_status', axis=1)
df.info()
from sklearn.model_selection import train_test_split

X = df.drop('loan_repaid', axis=1)

y = df['loan_repaid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
# Normalizing and scaling features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fitting and transforming training and test sets

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_test_scaled
y_train = y_train.values

y_test = y_test.values
# Importing tensorflow

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout
# Instantiate model

model = Sequential()



# input layer

model.add(Dense(units=78, activation='relu'))

model.add(Dropout(0.25))



# hidden layers

model.add(Dense(units=39, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(units=19, activation='relu'))

model.add(Dropout(0.25))



# output layer

model.add(Dense(units=1, activation='sigmoid'))



# compile model

model.compile(loss='binary_crossentropy', optimizer='adam')
# Fitting model

model.fit(X_train_scaled, y_train, epochs=25, validation_data=(X_test_scaled, y_test), batch_size=256)
# Saving model

from tensorflow.keras.models import load_model

model.save('Client_Loan_ANN_model.h5')
# Creating dataframe including training loss values vs. validation loss values

df_model = pd.DataFrame(model.history.history)

df_model.head()
df_model.plot()

plt.show()
# Creating predictions using model

y_pred = model.predict_classes(X_test_scaled)
from sklearn.metrics import confusion_matrix, classification_report
y_test.shape
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
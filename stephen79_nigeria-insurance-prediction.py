# load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as profiling

# stop warnings

import warnings

warnings.filterwarnings('ignore')
# display column limit

pd.set_option('display.max_columns',100)
# load data

train = pd.read_csv('../input/train_data.csv')

validation = pd.read_csv('../input/test_data.csv')

train.head()
train.describe()
# shuffle

train = train.sample(frac=1)
# missing values

train.isnull().sum()
# Missing values

train['Garden'].fillna(train['Garden'].mode()[0],inplace=True)

train['Geo_Code'] = train['Geo_Code'].fillna(0)

train['Date_of_Occupancy'].fillna(train['Date_of_Occupancy'].mode()[0],inplace=True)

train['Building Dimension'].fillna(train['Building Dimension'].mode()[0],inplace=True)





# validation

validation['Garden'].fillna(validation['Garden'].mode()[0],inplace=True)

validation['Building Dimension'].fillna(validation['Building Dimension'].mode()[0],inplace=True)

validation['Date_of_Occupancy'].fillna(validation['Date_of_Occupancy'].mode()[0],inplace=True)

validation['Geo_Code'] = validation['Geo_Code'].fillna(0)

train['years_of_occupation'] = 2016 - train['Date_of_Occupancy']

# test

validation['years_of_occupation'] = 2016 - validation['Date_of_Occupancy']

train.head()
train.describe()
validation.head()
# find missing values

train.isnull().sum().sum()
validation.isnull().sum().sum()
sns.countplot(x='YearOfObservation', data=train, hue='Claim') 
sns.countplot(x='NumberOfWindows', data=train, hue='Claim')
sns.countplot(x='Residential', data=train, hue='Claim')
sns.countplot(x='Building_Painted', data=train, hue='Claim')
sns.countplot(x='Building_Fenced', data=train, hue='Claim')
sns.countplot(x='Garden',data=train, hue='Claim')
sns.countplot(x='Settlement', data=train, hue='Claim')
sns.countplot(x='Building_Type', data=train, hue='Claim')
sns.distplot(train['Building Dimension'])
train.head()
# checking the balance of the data

print(' Building has Claims during insured period: ' + str(train['Claim'].value_counts()[0]) + ' is', round(train['Claim'].value_counts()[0]/len(train) * 100,2), '% of dataset')

print(' Building does not have Claims: ' + str(train['Claim'].value_counts()[1]) + ' is', round(train['Claim'].value_counts()[1]/len(train) * 100,2), '% of dataset')
# smote

# oversampling

from imblearn.over_sampling import SMOTE



count_class_0, count_class_1 = train.Claim.value_counts()



# divide by class

train_class_0 = train[train['Claim'] == 0]

train_class_1 = train[train['Claim'] == 1]
train_class_1_over = train_class_1.sample(count_class_0, replace=True)

train_test_over = pd.concat([train_class_0, train_class_1_over], axis=0)



print('Random over-sampling:')

print(train_test_over.Claim.value_counts())



train_test_over.Claim.value_counts().plot(kind='bar', title='Count (Claim)');
train = train_test_over
# rename columns

train.rename(columns={'Building Dimension':'Building_Dimension'},inplace=True)

# test

validation.rename(columns={'Building Dimension':'Building_Dimension'},inplace=True)
# train

train.loc[train['NumberOfWindows'] > 0, 'numberofwindows'] = 0

train.loc[train['NumberOfWindows'] < 0, 'numberofwindows'] = 1







#validation

validation.loc[validation['NumberOfWindows'] > 0, 'numberofwindows'] = 0

validation.loc[validation['NumberOfWindows'] < 0, 'numberofwindows'] = 1



train.head()
train["numberofwindows"].fillna("0", inplace = True) 

validation["numberofwindows"].fillna("0", inplace = True) 
# binning 

bins = [0,100,200,300,400,500]

labels = [1,2,3,4,5]

train['years_bin'] = pd.cut(train['years_of_occupation'], bins,labels=labels)

validation['years_bin'] = pd.cut(validation['years_of_occupation'], bins,labels=labels)

train.head()
# binning number of windows

bins = [0,2,4,8,10]

labels = [4,3,2,1]

train['NumberOfWindows'] = pd.cut(train['NumberOfWindows'], bins,labels=labels)

validation['NumberOfWindows'] = pd.cut(validation['NumberOfWindows'], bins,labels=labels)
sns.countplot(x='NumberOfWindows', data=train, hue='Claim')
# binning number of Building_Dimension

bins = [0,5000,10000,15000,20000,25000]

labels = [5,4,3,2,1]

train['Building_Dimension'] = pd.cut(train['Building_Dimension'], bins,labels=labels)

validation['Building_Dimension'] = pd.cut(validation['Building_Dimension'], bins,labels=labels)
sns.countplot(x='Building_Dimension', data=train, hue='Claim')
# normalizing

from sklearn.preprocessing import MinMaxScaler

# Building Dimension

scaler_Building_Dimension = MinMaxScaler()

train['Building_Dimension'] = train['Building_Dimension'].astype('float64')

train['Building_Dimension'] = scaler_Building_Dimension.fit_transform(train.Building_Dimension.values.reshape(-1,1))



# YearOfObservatio

scaler_YearOfObservation = MinMaxScaler()

train['YearOfObservation'] = scaler_YearOfObservation.fit_transform(train.YearOfObservation.values.reshape(-1,1))

# Date_of_Occupancy

scaler_Date_of_Occupancy = MinMaxScaler()

train['Date_of_Occupancy'] = scaler_Date_of_Occupancy.fit_transform(train.Date_of_Occupancy.values.reshape(-1,1))



# years_of_occupation

scaler_years_of_occupation = MinMaxScaler()

train['years_of_occupation'] = scaler_years_of_occupation.fit_transform(train.years_of_occupation.values.reshape(-1,1))





# validation

# Building Dimension

scaler_Building_Dimension = MinMaxScaler()

validation['Building_Dimension'] = validation['Building_Dimension'].astype('float64')

validation['Building_Dimension'] = scaler_Building_Dimension.fit_transform(validation.Building_Dimension.values.reshape(-1,1))



# YearOfObservatio

scaler_YearOfObservation = MinMaxScaler()

validation['YearOfObservation'] = scaler_YearOfObservation.fit_transform(validation.YearOfObservation.values.reshape(-1,1))

# Date_of_Occupancy

scaler_Date_of_Occupancy = MinMaxScaler()

validation['Date_of_Occupancy'] = scaler_Date_of_Occupancy.fit_transform(validation.Date_of_Occupancy.values.reshape(-1,1))



# years_of_occupation

scaler_years_of_occupation = MinMaxScaler()

validation['years_of_occupation'] = scaler_years_of_occupation.fit_transform(validation.years_of_occupation.values.reshape(-1,1))
train = train.round(4)
validation = validation.round(4)
# binary conversion

train['Building_Painted_Binary'] = train['Building_Painted'].map({'N':1,'V':0})

train['Building_Fenced_Binary'] = train['Building_Fenced'].map({'N':1,'V':0})

train['Garden_Binary'] = train['Garden'].map({'O':1,'V':0})

train['Settlement_Binary'] = train['Settlement'].map({'U':1,'R':0})

# test

validation['Building_Painted_Binary'] = validation['Building_Painted'].map({'N':1,'V':0})

validation['Building_Fenced_Binary'] = validation['Building_Fenced'].map({'N':1,'V':0})

validation['Garden_Binary'] = validation['Garden'].map({'O':1,'V':0})

validation['Settlement_Binary'] = validation['Settlement'].map({'U':1,'R':0})
train.head()
# value

train['value'] = train['Building_Type']+ train['Building_Painted_Binary']+train['Building_Fenced_Binary']+train['Garden_Binary']+train['Settlement_Binary']+train['Residential'] 

validation['value'] = validation['Building_Type']+ validation['Building_Painted_Binary']+validation['Building_Fenced_Binary']+validation['Garden_Binary']+validation['Settlement_Binary']+validation['Residential']

# house_grade

train['house_grade'] = train['value'] * train['Building_Dimension'] * train['years_of_occupation'] * train['Insured_Period']

validation['house_grade'] = validation['value'] * validation['Building_Dimension'] * validation['years_of_occupation'] * validation['Insured_Period']

# insurance_factor

#train['insurance_factor'] = train ['value'] * train['residence_before_observation']

#validation['insurance_factor'] = validation ['value'] * validation['residence_before_observation']
train['exponential'] = np.log(train['Building_Type']**2 + train['Building_Painted_Binary']**2 + train['Building_Fenced_Binary']**2 + train['Garden_Binary']**2 + train['Settlement_Binary']**2 + train['Residential']**2)

validation['exponential'] = np.log(validation['Building_Type']**2 + validation['Building_Painted_Binary']**2 + validation['Building_Fenced_Binary']**2 + validation['Garden_Binary']**2 + validation['Settlement_Binary']**2 + validation['Residential']**2)
train.corr()*100
# drop irrelevant colums

train = train.drop(['YearOfObservation','NumberOfWindows','Building_Painted','Building_Fenced','Garden','Settlement','Geo_Code'], axis=1)

# test

validation = validation.drop(['YearOfObservation','NumberOfWindows','Building_Painted','Building_Fenced','Garden','Settlement','Geo_Code'], axis=1)

train.round(4)

train.head()
validation.isnull().sum()
validation["Building_Dimension"].fillna("0", inplace = True) 

validation["house_grade"].fillna("2", inplace = True) 
train.head()
# drop column customer id

train = train.drop(['Customer Id'], axis=1)
# validation

validation1 = validation.copy()

validation = validation.drop(['Customer Id'], axis=1)
train.shape
train['Garden_Binary'].fillna(train['Garden_Binary'].mode()[0],inplace=True)

train['value'].fillna(train['value'].mode()[0],inplace=True)

train['house_grade'].fillna(train['house_grade'].mode()[0],inplace=True)

train['exponential'].fillna(train['exponential'].mode()[0],inplace=True)

train['years_bin'].fillna(train['years_bin'].mode()[0],inplace=True)                            



#validation

validation['Garden_Binary'].fillna(validation['Garden_Binary'].mode()[0],inplace=True)

validation['value'].fillna(validation['value'].mode()[0],inplace=True)

validation['house_grade'].fillna(validation['house_grade'].mode()[0],inplace=True)

validation['exponential'].fillna(validation['exponential'].mode()[0],inplace=True)

validation['years_bin'].fillna(validation['years_bin'].mode()[0],inplace=True)
# feature selection

y = train.Claim



X = train.drop(['Claim'], axis=1)
train = train.drop(['Date_of_Occupancy','years_of_occupation','Building_Painted_Binary','years_of_occupation'], axis=1)

validation = validation.drop(['Date_of_Occupancy','years_of_occupation','Building_Painted_Binary','years_of_occupation'], axis=1)
# feature selection

y = train.Claim



X = train.drop(['Claim'], axis=1)
#split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# bring the test dataset

from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR = LR.fit(X_train, y_train)



submit = LR.predict(validation)
submission = pd.DataFrame({'Customer Id':validation1['Customer Id'],'Claim':submit})
submission.to_csv('submit129.csv', index=False)
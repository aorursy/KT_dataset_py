!pip install feature_engine
#Import the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import probplot

from sklearn.model_selection import train_test_split, cross_val_score

from feature_engine.missing_data_imputers import MeanMedianImputer,CategoricalVariableImputer,RandomSampleImputer

import feature_engine.categorical_encoders as ct

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
#Read the data

data = pd.read_csv('/kaggle/input/titanic/train.csv')
#Checking the head

data.head()
#Join Ticket  observation by '_' 

data['Ticket']=data['Ticket'].apply(lambda x : '_'.join(x.split()))
#Describe the data

data.describe()
#Getting the shape

data.shape
data = data[[col for col in data.columns if col not in ['Name','PassengerId']]]
data.columns
#checking the counts of target per category

data['Survived'].value_counts(ascending=False)
#Plot the target count data

sns.barplot(data.groupby('Survived')['Survived'].count().index, data.groupby('Survived')['Survived'].count().values)
#Checking the missing value if any

data.isnull().mean()*100
#Method to get the numerical and categorical column form the dataset

def dtype_check(data):

    categorical_list = []

    numerical_list = []

    for col in data.columns:

        if(data[col].dtype == 'object'):

            categorical_list.append(col)

        else:

            numerical_list.append(col)

    return categorical_list, numerical_list
categorical , numerical = dtype_check(data)
categorical, numerical
#Seggregate the data to X and y

X = data[[col for col in data.columns if col not in 'Survived']]

y = data['Survived']
#Split the data to training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#Method to create visualise the data ditrbution and outliers

def diagonastic_plot(data, column):

    

    plt.figure(figsize=(10,6))

    plt.subplot(1,3,1)

    sns.distplot(data[column].dropna())

    

    plt.subplot(1,3,2)

    probplot(data[column].dropna(), plot=plt)

    

    plt.subplot(1,3,3)

    sns.boxplot(data[column], orient='v')

    

    plt.show()
#Impute the missing Age column

imputer = MeanMedianImputer(imputation_method='mean')

imputer.fit(X_train)
#Transform the variable

X_train = imputer.transform(X_train)

X_test = imputer.transform(X_test)
#Check unique value in each categorical columns

unique_dict = {}

for col in categorical:

    unique_dict[col] = X_train[col].unique()
unique_dict
#Check number of unique value in each categorical column

nunique_dict = {}

for col in categorical:

    nunique_dict[col] = X_train[col].nunique() 
nunique_dict
X_train.isnull().mean()*100
#Adding Missing Indicator in the Cabin Column

impute_cabin = CategoricalVariableImputer(variables=['Cabin'])

impute_cabin.fit(X_train)
#Transform the variable

X_train = impute_cabin.transform(X_train)

X_test = impute_cabin.transform(X_test)
#Join Ticket  observation by '_' 

data['Cabin']=data['Cabin'].apply(lambda x : '_'.join(str(x).split()))
#Impute the Embarked column

imputer_e = RandomSampleImputer(variables=['Embarked'])

imputer_e.fit(X_train)
#Transform the varibale

X_train = imputer_e.transform(X_train)

X_test = imputer_e.transform(X_test)
#Encode the categorical variable of Ticket

encoder_t = ct.OneHotCategoricalEncoder(variables=['Ticket'])

encoder_t.fit(X_train)
#Transform the variable

X_train = encoder_t.transform(X_train)

X_test = encoder_t.transform(X_test)
#Encode the categorical variable of Cabin

encoder_c = ct.OneHotCategoricalEncoder(variables=['Cabin'])

encoder_c.fit(X_train)
#Transform the variable

X_train = encoder_c.transform(X_train)

X_test = encoder_c.transform(X_test)
#Encode the Sex and Embarked column

encoder_se = ct.OneHotCategoricalEncoder(variables=['Sex', 'Embarked'])

encoder_se.fit(X_train)
#Transform the variable

X_train = encoder_se.transform(X_train)

X_test = encoder_se.transform(X_test)
#Getting the shape of the data

X_train.shape, X_test.shape
#Fitting the random forest model

rf = RandomForestClassifier(n_estimators=1000, max_depth=7)

rf.fit(X_train, y_train)
print(classification_report(y_train, rf.predict(X_train)))
print(classification_report(y_test, rf.predict(X_test)))
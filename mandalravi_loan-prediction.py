# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt 

import seaborn as sns 



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Setting the path

path ="../input"

os.chdir(path)
# reading data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample.csv")
train.shape,  test.shape
# Get the Columns Name from train dataset



train.columns, test.columns
# Understand the each variable metadata or information

train.info()
numeric_features = train.select_dtypes(include = ['int64', 'float64']).columns

categorical_features = train.iloc[:, 0:12].select_dtypes(include = ['object']).columns



print("Numeric features:", numeric_features)

print("Categorical features:", categorical_features)
train.isnull().sum()
train.head(5)
train.Loan_Status.value_counts(normalize = True)
sns.countplot(train['Loan_Status'],label="Count")

plt.show()
train.describe()
ApplicantIncome_cv = train['ApplicantIncome'].std()/train['ApplicantIncome'].mean()

CoapplicantIncome_cv = train['CoapplicantIncome'].std()/train['CoapplicantIncome'].mean()

LoanAmount_cv = train['LoanAmount'].std()/train['LoanAmount'].mean()

Loan_Amount_Term_cv = train['Loan_Amount_Term'].std()/train['Loan_Amount_Term'].mean()

Credit_History_cv = train['Credit_History'].std()/train['Credit_History'].mean()



print("Coefficient of variance for ApplicantIncome is:", ApplicantIncome_cv)

print("Coefficient of variance for CoapplicantIncome is:", CoapplicantIncome_cv)

print("Coefficient of variance for LoanAmount is:", LoanAmount_cv)

print("Coefficient of variance for Loan_Amount_Term is:", Loan_Amount_Term_cv)

print("Coefficient of variance for Credit_History is:", Credit_History_cv)
#Numerical Variable

plt.figure(1)

plt.subplot(121)

sns.distplot(train['ApplicantIncome']);



plt.subplot(122)

train['ApplicantIncome'].plot.box(figsize=(16,5))



plt.show();
#Numerical Variable

plt.figure(1)

plt.subplot(121)

sns.distplot(train['CoapplicantIncome']);



plt.subplot(122)

train['CoapplicantIncome'].plot.box(figsize=(16,5))



plt.show();
#Numerical Variable

plt.figure(1)

plt.subplot(121)

sns.distplot(train['LoanAmount'].dropna());



plt.subplot(122)

train['LoanAmount'].dropna().plot.box(figsize=(16,5))



plt.show();
#Numerical Variable

print(train['Loan_Amount_Term'].value_counts())



plt.figure(1)

plt.subplot(121)

sns.distplot(train['Loan_Amount_Term'].dropna());



plt.subplot(122)

train['Loan_Amount_Term'].dropna().plot.box(figsize=(16,5))



plt.show();
#Numerical Variable

print(train['Credit_History'].value_counts())

plt.figure(1)

plt.subplot(121)

sns.distplot(train['Credit_History'].dropna());



plt.subplot(122)

train['Credit_History'].dropna().plot.box(figsize=(16,5))



plt.show();
#Categorical Features



plt.figure(1)

plt.subplot(221)

train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')



plt.subplot(222)

train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')



plt.subplot(223)

train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')



plt.subplot(224)

train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')



plt.show()
# Ordinal Variables

plt.figure(1)

plt.subplot(131)

train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(22,4),title= 'Dependents')



plt.subplot(132)

train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')



plt.subplot(133)

train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')



plt.show()


sns.countplot(train['Gender'], hue=train['Loan_Status'])

plt.show()



sns.countplot(train['Dependents'], hue=train['Loan_Status'])

plt.show()



sns.countplot(train['Education'], hue=train['Loan_Status'])

plt.show()



sns.countplot(train['Self_Employed'], hue=train['Loan_Status'])

plt.show()



sns.countplot(train['Property_Area'], hue=train['Loan_Status'])

plt.show()
#Looking at a correlation among all numeric variables



corr_matrix = train[numeric_features].corr()

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(corr_matrix, vmax=.8, annot=True, square=True, cmap="BuPu");
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer



numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])
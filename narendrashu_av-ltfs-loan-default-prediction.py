# !pip install fastai==0.7 --no-deps

# !pip install torch==0.4.1 torchvision==0.2.1

import fastai

from fastai.vision import *

from fastai.collab import *

from fastai.tabular import *

fastai.__version__
path = '../working'

!ls {path}
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from fastai.collab import *

# from fastai.tabular import *

from fastai.imports import *

#from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

import itertools

from sklearn.metrics import *

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

from fastai.imports import *

# Any results you write to the current directory are saved as output.
PATH = "../input"

!ls {PATH}
path = '../working'

!ls {path}
# reading the data



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test_bqCt9Pv.csv')



# getting the shapes of the datasets

print("Shape of Train :", train.shape)

print("Shape of Test :", test.shape)
def display_all(df):

    with pd.option_context("display.max_rows", 1000): 

        with pd.option_context("display.max_columns", 1000): 

            display(df)
# lets look at the head of the train



train.head()
# lets look at the head of the test data



test.head()
# let's decsribe the train set



train.describe()
# let's describe the test set



test.describe()
# get the info of train



train.info()
# get the info of test set



test.info()
# getting the data types of train



train.dtypes
# getting the data types of test



test.dtypes
# checking if there exists any NULL values in the train set



train.isnull().sum()
# checking if there exists any NULL values in the test set



test.isnull().sum()
# checking the values present in the Employement.Type attribute in the train and test sets



train['Employment.Type'].value_counts()
# filling the missing values in the Employment.Type attribute of train and test sets



# Employement Type has two types of Employment i.e., self employed and salaried

# but the empty values must be the people who don't  work at all that's why it is empty

# let's fill unemployed in the place of Null values



train['Employment.Type'].fillna('Unemployed', inplace = True)

test['Employment.Type'].fillna('Unemployed', inplace = True)



# let's check if there is any null values still left or not

print("Null values left in the train set:", train.isnull().sum().sum())

print("Null values left in the test set:", test.isnull().sum().sum())
# let's save the unique id of the test set and labels set



unique_id = test['UniqueID']

y = train['loan_default']



# let's delete the last column from the dataset to  concat train and test

train = train.drop(['loan_default'], axis = 1)



# shape of train

train.shape, test.shape,y.shape
# lets concat the train and test sets for preprocessing and visualizations



data = pd.concat([train, test], axis = 0,ignore_index=True)



# let's check the shape

data.shape

# let's check the employement type in the data

data['Employment.Type'].value_counts()

#train['Employment.Type'].value_counts(), test['Employment.Type'].value_counts()
# plotting a donut chart



size = [187429, 147013, 11104]

colors = ['pink', 'lightblue', 'lightgreen']

labels = "Self Employed", "Salried", "Unemployed" 

explode = [0.05, 0.05, 0.05]



circle = plt.Circle((0, 0), 0.7, color = 'white')



plt.rcParams['figure.figsize'] = (10, 10)

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, pctdistance = 1, autopct = '%.2f%%')

plt.title('Types of Employments', fontsize = 30)

plt.axis('off')

p = plt.gcf()

p.gca().add_artist(circle)

plt.legend()

plt.show()
# encodings for type of employments



data['Employment.Type'] = data['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))

train['Employment.Type'] = train['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))

test['Employment.Type'] = test['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))



# checking the values  of employement type

data['Employment.Type'].value_counts()
# checking the columns names of the data



data.columns
#let's check the unique values of ids in different branchs



print("Total no. of Unique Ids :", data['UniqueID'].nunique())

print("Total no. of Unique Branches :", data['branch_id'].nunique())

print("Total no. of Unique Suppliers :", data['supplier_id'].nunique())

print("Total no. of Unique Manufactures :", data['manufacturer_id'].nunique())

print("Total no. of Unique Current pincode Ids :", data['Current_pincode_ID'].nunique())

print("Total no. of Unique State IDs :",data['State_ID'].nunique())

print("Total no. of Unique Employee code IDs :", data['Employee_code_ID'].nunique())
# check the distribution of disbursed amount



plt.rcParams['figure.figsize'] = (18, 5)



plt.subplot(1, 3, 1)

sns.distplot(data['disbursed_amount'],  color = 'orange')

plt.title('Disburesed Amount')



plt.subplot(1, 3, 2)

sns.distplot(data['asset_cost'], color = 'pink')

plt.title('Asset Cost')



plt.subplot(1, 3, 3)

sns.distplot(data['ltv'], color = 'red')

plt.title('Loan to value of the asset')



plt.show()
#performing log transformations on disbursed amount, ltv, and asset cost



data['disbursed_amount'] = np.log1p(data['disbursed_amount'])

data['ltv'] = np.log1p(data['ltv'])

data['asset_cost'] = np.log1p(data['asset_cost'])



train['disbursed_amount'] = np.log1p(train['disbursed_amount'])

train['ltv'] = np.log1p(train['ltv'])

train['asset_cost'] = np.log1p(train['asset_cost'])



test['disbursed_amount'] = np.log1p(test['disbursed_amount'])

test['ltv'] = np.log1p(test['ltv'])

test['asset_cost'] = np.log1p(test['asset_cost'])



plt.rcParams['figure.figsize'] = (18, 5)



plt.subplot(1, 3, 1)

sns.distplot(data['disbursed_amount'],  color = 'orange')

plt.title('Disburesed Amount')



plt.subplot(1, 3, 2)

sns.distplot(data['asset_cost'], color = 'pink')

plt.title('Asset Cost')



plt.subplot(1, 3, 3)

sns.distplot(data['ltv'], color = 'red')

plt.title('Loan to value of the asset')



plt.show()
# date of birth is an useless attribute 

#  the only thing we can extract the is the year of birth

# let's first convert the date into date-time format



data['Date.of.Birth'] = pd.to_datetime(data['Date.of.Birth'], errors = 'coerce')

train['Date.of.Birth'] = pd.to_datetime(train['Date.of.Birth'], errors = 'coerce')

test['Date.of.Birth'] = pd.to_datetime(test['Date.of.Birth'], errors = 'coerce')



# extracting the year of birth of the customers

data['Year_of_birth'] = data['Date.of.Birth'].dt.year

train['Year_of_birth'] = train['Date.of.Birth'].dt.year

test['Year_of_birth'] = test['Date.of.Birth'].dt.year

data['Year_of_birth'].min(), data['Year_of_birth'].max()
##assuming we don't have enough histroy to secure a loan till age of 18 i.e. max year should be 2001.we will cap this year

data.loc[data['Date.of.Birth'].dt.year >2001,'Date.of.Birth'] = pd.to_datetime(20010101,format='%Y%m%d')

train.loc[train['Date.of.Birth'].dt.year >2001,'Date.of.Birth'] = pd.to_datetime(20010101,format='%Y%m%d')

test.loc[train['Date.of.Birth'].dt.year >2001,'Date.of.Birth'] = pd.to_datetime(20010101,format='%Y%m%d')
import datetime as DT

now = pd.Timestamp(DT.datetime.now())

data['age'] = (now - data['Date.of.Birth']).astype('<m8[Y]') 

data['Year_of_birth'] = data['Date.of.Birth'].dt.year



train['age'] = (now - train['Date.of.Birth']).astype('<m8[Y]') 

train['Year_of_birth'] = train['Date.of.Birth'].dt.year



test['age'] = (now - test['Date.of.Birth']).astype('<m8[Y]') 

test['Year_of_birth'] = test['Date.of.Birth'].dt.year

sns.distplot(data['age'],color='red')

plt.title('Distribution by age')
# checking the values inside date of year

sns.distplot(data['Year_of_birth'], color = 'blue')

plt.title('Distribution of Year of birth')
# plotting a countplot



sns.countplot(data['NO.OF_INQUIRIES'], palette = 'muted')

plt.title('No. of Inquiries',  fontsize = 30)
# # plotting countplot for credit history of users

plt.rcParams['figure.figsize'] = (18, 5)

sns.countplot(data['CREDIT.HISTORY.LENGTH'].head(50))

plt.title('Credit History')

plt.xticks(rotation = 45)
##convert string formatted year.month columns to number of months

def convert__int(column):

    months=[i for i in range(len(column))]

    for j in range(len(column)):

        months[j] = int(re.findall(r'\d+', column[j])[0]) *12 + int(re.findall(r'\d+', column[j])[1])

        #column[i] = months[i]

    return months
# # plotting countplot for credit history of users

plt.rcParams['figure.figsize'] = (18, 5)

sns.countplot(data['CREDIT.HISTORY.LENGTH'].tail(50))

plt.title('Credit History')

plt.xticks(rotation = 45)
# changing the credit history format from ayrsbmonths to number of months 

data['AVERAGE.ACCT.AGE'] = convert__int(data['AVERAGE.ACCT.AGE'])

data['CREDIT.HISTORY.LENGTH'] = convert__int(data['CREDIT.HISTORY.LENGTH'])



train['AVERAGE.ACCT.AGE'] = convert__int(train['AVERAGE.ACCT.AGE'])

train['CREDIT.HISTORY.LENGTH'] = convert__int(train['CREDIT.HISTORY.LENGTH'])



test['AVERAGE.ACCT.AGE'] = convert__int(test['AVERAGE.ACCT.AGE'])

test['CREDIT.HISTORY.LENGTH'] = convert__int(test['CREDIT.HISTORY.LENGTH'])

# data['CREDIT.HISTORY.LENGTH'] = data['CREDIT.HISTORY.LENGTH'].apply(lambda x: x.split(' ')[0])

# data['CREDIT.HISTORY.LENGTH'] = data['CREDIT.HISTORY.LENGTH'].apply(lambda x: x.split('yrs')[0])

#data['CREDIT.HISTORY.LENGTH'].value_counts()
# distribution of credit history years



plt.rcParams['figure.figsize'] = (18, 5)

sns.distplot(data['CREDIT.HISTORY.LENGTH'])

plt.title('Credit History in Years', fontsize = 25)

plt.show()
# average.acct.age i.e., average loan tenure



sns.distplot(data['AVERAGE.ACCT.AGE'].head(50))

plt.title('Average Loan Tenure')

plt.xticks(rotation = 45)
# average.acct.age i.e., average loan tenure



sns.countplot(data['AVERAGE.ACCT.AGE'].tail(50), palette = 'colorblind')

plt.title('Average Loan Tenure')

plt.xticks(rotation = 45)
# changing the average account age format from ayrsbmonths to years 

# as no. of years are more important



# data['AVERAGE.ACCT.AGE'] = data['AVERAGE.ACCT.AGE'].apply(lambda x: x.split(' ')[0])

# data['AVERAGE.ACCT.AGE'] = data['AVERAGE.ACCT.AGE'].apply(lambda x: x.split('yrs')[0])

#data['AVERAGE.ACCT.AGE'].value_counts()
# loans defaulted in last six months



data['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].value_counts()
# new loans taken by the customer before disbursement



data['NEW.ACCTS.IN.LAST.SIX.MONTHS'].value_counts()
# EMI Amount of the Secondary Plan



plt.subplot(1, 2, 1)

sns.distplot(data['SEC.INSTAL.AMT'])

plt.title('EMI Amount Secondary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.subplot(1, 2, 2)

sns.distplot(data['PRIMARY.INSTAL.AMT'])

plt.title('EMI Amount Primary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.show()
# let's apply log transformations on EMI Amount of the Primary Loan and Secondary loan



data['PRIMARY.INSTAL.AMT'] = np.log1p(data['PRIMARY.INSTAL.AMT'])

data['SEC.INSTAL.AMT'] = np.log1p(data['SEC.INSTAL.AMT'])



train['PRIMARY.INSTAL.AMT'] = np.log1p(train['PRIMARY.INSTAL.AMT'])

train['SEC.INSTAL.AMT'] = np.log1p(train['SEC.INSTAL.AMT'])



test['PRIMARY.INSTAL.AMT'] = np.log1p(test['PRIMARY.INSTAL.AMT'])

test['SEC.INSTAL.AMT'] = np.log1p(test['SEC.INSTAL.AMT'])



plt.subplot(1, 2, 1)

sns.distplot(data['SEC.INSTAL.AMT'], color = 'yellow')

plt.title('EMI Amount Secondary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.subplot(1, 2, 2)

sns.distplot(data['PRIMARY.INSTAL.AMT'],color = 'yellow')

plt.title('EMI Amount Primary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.show()
# distribution for different attributesof secondary accounts





plt.rcParams['figure.figsize'] = (18, 12)    

plt.subplot(2, 3, 1)

sns.distplot(data['SEC.NO.OF.ACCTS'], color = 'green')

plt.title('Total loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.distplot(data['SEC.ACTIVE.ACCTS'], color = 'green')

plt.title('Active loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.distplot(data['SEC.OVERDUE.ACCTS'], color = 'green')

plt.title('Default Accounts at the time of disbursement')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.distplot(data['SEC.CURRENT.BALANCE'], color = 'green')

plt.title('Principal Outstanding amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.distplot(data['SEC.SANCTIONED.AMOUNT'], color = 'green')

plt.title('Total Sanctioned Amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.distplot(data['SEC.DISBURSED.AMOUNT'], color = 'green')

plt.title('Total Disbured Amount')

plt.xticks(rotation = 45)
# applying log transformation to all these attributes



data['SEC.NO.OF.ACCTS'] = np.log1p(data['SEC.NO.OF.ACCTS'])

data['SEC.ACTIVE.ACCTS'] = np.log1p(data['SEC.ACTIVE.ACCTS'])

data['SEC.OVERDUE.ACCTS'] = np.log1p(data['SEC.OVERDUE.ACCTS'])

#data['SEC.CURRENT.BALANCE'] = np.log1p(data['SEC.CURRENT.BALANCE'])

data['SEC.SANCTIONED.AMOUNT'] = np.log1p(data['SEC.SANCTIONED.AMOUNT'])

data['SEC.DISBURSED.AMOUNT'] = np.log1p(data['SEC.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

data['SEC.CURRENT.BALANCE'].fillna(data['SEC.CURRENT.BALANCE'].mean(), inplace = True)

train['SEC.NO.OF.ACCTS'] = np.log1p(train['SEC.NO.OF.ACCTS'])

train['SEC.ACTIVE.ACCTS'] = np.log1p(train['SEC.ACTIVE.ACCTS'])

train['SEC.OVERDUE.ACCTS'] = np.log1p(train['SEC.OVERDUE.ACCTS'])

#train['SEC.CURRENT.BALANCE'] = np.log1p(train['SEC.CURRENT.BALANCE'])

train['SEC.SANCTIONED.AMOUNT'] = np.log1p(train['SEC.SANCTIONED.AMOUNT'])

train['SEC.DISBURSED.AMOUNT'] = np.log1p(train['SEC.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

train['SEC.CURRENT.BALANCE'].fillna(train['SEC.CURRENT.BALANCE'].mean(), inplace = True)
test['SEC.NO.OF.ACCTS'] = np.log1p(test['SEC.NO.OF.ACCTS'])

test['SEC.ACTIVE.ACCTS'] = np.log1p(test['SEC.ACTIVE.ACCTS'])

test['SEC.OVERDUE.ACCTS'] = np.log1p(test['SEC.OVERDUE.ACCTS'])

#test['SEC.CURRENT.BALANCE'] = np.log1p(test['SEC.CURRENT.BALANCE'])

test['SEC.SANCTIONED.AMOUNT'] = np.log1p(test['SEC.SANCTIONED.AMOUNT'])

test['SEC.DISBURSED.AMOUNT'] = np.log1p(test['SEC.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

test['SEC.CURRENT.BALANCE'].fillna(test['SEC.CURRENT.BALANCE'].mean(), inplace = True)


plt.rcParams['figure.figsize'] = (18, 12)    

plt.subplot(2, 3, 1)

sns.distplot(data['SEC.NO.OF.ACCTS'], color = 'red')

plt.title('Total loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.distplot(data['SEC.ACTIVE.ACCTS'], color = 'red')

plt.title('Active loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.distplot(data['SEC.OVERDUE.ACCTS'], color = 'red')

plt.title('Default Accounts')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.distplot(data['SEC.CURRENT.BALANCE'], color = 'red')

plt.title('Principal Outstanding amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.distplot(data['SEC.SANCTIONED.AMOUNT'], color = 'red')

plt.title('Total Sanctioned Amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.distplot(data['SEC.DISBURSED.AMOUNT'], color = 'red')

plt.title('Total Disbured Amount')

plt.xticks(rotation = 45)



plt.show()
#  applying log transformations to the primary account attributes



data['PRI.NO.OF.ACCTS'] = np.log1p(data['PRI.NO.OF.ACCTS'])

data['PRI.ACTIVE.ACCTS'] = np.log1p(data['PRI.ACTIVE.ACCTS'])

#data['PRI.OVERDUE.ACCTS'] = np.log1p(data['PRI.OVERDUE.ACCTS'])

#data['PRI.CURRENT.BALANCE'] = np.log1p(data['PRI.CURRENT.BALANCE'])

data['PRI.SANCTIONED.AMOUNT'] = np.log1p(data['PRI.SANCTIONED.AMOUNT'])

data['PRI.DISBURSED.AMOUNT'] = np.log1p(data['PRI.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

data['PRI.CURRENT.BALANCE'].fillna(data['PRI.CURRENT.BALANCE'].mean(), inplace = True)

data['PRI.SANCTIONED.AMOUNT'].fillna(data['PRI.SANCTIONED.AMOUNT'].mean(), inplace = True)

data['PRI.OVERDUE.ACCTS'].fillna(data['PRI.OVERDUE.ACCTS'].mean(), inplace = True)

data['PRI.DISBURSED.AMOUNT'].fillna(data['PRI.DISBURSED.AMOUNT'].mean(), inplace = True)

#data['PRI.OVERDUE.ACCTS'].fillna(data['PRI.OVERDUE.ACCTS'].mean(), inplace = True)
#  applying log transformations to the primary account attributes



train['PRI.NO.OF.ACCTS'] = np.log1p(train['PRI.NO.OF.ACCTS'])

train['PRI.ACTIVE.ACCTS'] = np.log1p(train['PRI.ACTIVE.ACCTS'])

#data['PRI.OVERDUE.ACCTS'] = np.log1p(train['PRI.OVERDUE.ACCTS'])

#train['PRI.CURRENT.BALANCE'] = np.log1p(train['PRI.CURRENT.BALANCE'])

train['PRI.SANCTIONED.AMOUNT'] = np.log1p(train['PRI.SANCTIONED.AMOUNT'])

train['PRI.DISBURSED.AMOUNT'] = np.log1p(train['PRI.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

train['PRI.CURRENT.BALANCE'].fillna(train['PRI.CURRENT.BALANCE'].mean(), inplace = True)

train['PRI.SANCTIONED.AMOUNT'].fillna(train['PRI.SANCTIONED.AMOUNT'].mean(), inplace = True)

train['PRI.OVERDUE.ACCTS'].fillna(train['PRI.OVERDUE.ACCTS'].mean(), inplace = True)

train['PRI.DISBURSED.AMOUNT'].fillna(train['PRI.DISBURSED.AMOUNT'].mean(), inplace = True)
#  applying log transformations to the primary account attributes



test['PRI.NO.OF.ACCTS'] = np.log1p(test['PRI.NO.OF.ACCTS'])

test['PRI.ACTIVE.ACCTS'] = np.log1p(test['PRI.ACTIVE.ACCTS'])

#test['PRI.OVERDUE.ACCTS'] = np.log1p(test['PRI.OVERDUE.ACCTS'])

#test['PRI.CURRENT.BALANCE'] = np.log1p(test['PRI.CURRENT.BALANCE'])

test['PRI.SANCTIONED.AMOUNT'] = np.log1p(test['PRI.SANCTIONED.AMOUNT'])

test['PRI.DISBURSED.AMOUNT'] = np.log1p(test['PRI.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

test['PRI.CURRENT.BALANCE'].fillna(test['PRI.CURRENT.BALANCE'].mean(), inplace = True)

test['PRI.SANCTIONED.AMOUNT'].fillna(test['PRI.SANCTIONED.AMOUNT'].mean(), inplace = True)

test['PRI.OVERDUE.ACCTS'].fillna(test['PRI.OVERDUE.ACCTS'].mean(), inplace = True)

test['PRI.DISBURSED.AMOUNT'].fillna(test['PRI.DISBURSED.AMOUNT'].mean(), inplace = True)
# plotting distribution plots for these attributes



plt.rcParams['figure.figsize'] = (18, 12)    

plt.subplot(2, 3, 1)

sns.distplot(data['PRI.NO.OF.ACCTS'], color = 'violet')

plt.title('Total loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.distplot(data['PRI.ACTIVE.ACCTS'], color = 'violet')

plt.title('Active loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.distplot(data['PRI.OVERDUE.ACCTS'], color = 'violet')

plt.title('Default Accounts')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.distplot(data['PRI.CURRENT.BALANCE'], color = 'violet')

plt.title('Principal Outstanding amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.distplot(data['PRI.SANCTIONED.AMOUNT'], color = 'violet')

plt.title('Total Sanctioned Amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.distplot(data['PRI.DISBURSED.AMOUNT'], color = 'violet')

plt.title('Total Disbured Amount')

plt.xticks(rotation = 45)



plt.show()
# checking the bureau score description



plt.rcParams['figure.figsize'] = (19, 6)

sns.countplot(data['PERFORM_CNS.SCORE.DESCRIPTION'], palette = 'pastel')

plt.title('Bureau Score Description', fontsize = 30)

plt.xticks(rotation = 90)

plt.show()
# checking the perform cns score description



data['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
# encodings for bureau score(perform cns score distribution)



data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('No Bureau History Available', 0)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Sufficient History Not Available', 0)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Not Enough Info available on the customer', 0)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Activity seen on the customer (Inactive)',0)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Updates available in last 36 months', 0)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Only a Guarantor', 0)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: More than 50 active Accounts found',0)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('M-Very High Risk', 1)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('L-Very High Risk', 1)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('K-High Risk', 2)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('J-High Risk', 2)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('I-Medium Risk', 3)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('H-Medium Risk', 3)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('G-Low Risk', 4)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('F-Low Risk', 4)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('E-Low Risk', 4)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('D-Very Low Risk', 5)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('C-Very Low Risk', 5)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('B-Very Low Risk', 5)

data['PERFORM_CNS.SCORE.DESCRIPTION'] = data['PERFORM_CNS.SCORE.DESCRIPTION'].replace('A-Very Low Risk', 5)



# checing the values in bureau score

data['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('No Bureau History Available', 0)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Sufficient History Not Available', 0)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Not Enough Info available on the customer', 0)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Activity seen on the customer (Inactive)',0)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Updates available in last 36 months', 0)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Only a Guarantor', 0)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: More than 50 active Accounts found',0)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('M-Very High Risk', 1)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('L-Very High Risk', 1)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('K-High Risk', 2)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('J-High Risk', 2)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('I-Medium Risk', 3)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('H-Medium Risk', 3)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('G-Low Risk', 4)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('F-Low Risk', 4)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('E-Low Risk', 4)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('D-Very Low Risk', 5)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('C-Very Low Risk', 5)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('B-Very Low Risk', 5)

train['PERFORM_CNS.SCORE.DESCRIPTION'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].replace('A-Very Low Risk', 5)
test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('No Bureau History Available', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Sufficient History Not Available', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Not Enough Info available on the customer', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Activity seen on the customer (Inactive)',0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Updates available in last 36 months', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Only a Guarantor', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: More than 50 active Accounts found',0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('M-Very High Risk', 1)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('L-Very High Risk', 1)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('K-High Risk', 2)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('J-High Risk', 2)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('I-Medium Risk', 3)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('H-Medium Risk', 3)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('G-Low Risk', 4)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('F-Low Risk', 4)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('E-Low Risk', 4)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('D-Very Low Risk', 5)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('C-Very Low Risk', 5)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('B-Very Low Risk', 5)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('A-Very Low Risk', 5)
# checking the bureau score



plt.rcParams['figure.figsize'] = (15, 5)

plt.subplot(1, 2, 1)

sns.distplot(data['PERFORM_CNS.SCORE'], color = 'purple')

plt.title('Before Log transformations')



plt.subplot(1, 2, 2)

data['PERFORM_CNS.SCORE'] = np.log1p(data['PERFORM_CNS.SCORE'])

train['PERFORM_CNS.SCORE'] = np.log1p(train['PERFORM_CNS.SCORE'])

test['PERFORM_CNS.SCORE'] = np.log1p(test['PERFORM_CNS.SCORE'])

sns.distplot(data['PERFORM_CNS.SCORE'], color = 'maroon')

plt.title('After Log transformations')



plt.show()
# customer has passport or not



data['Passport_flag'].value_counts()
# customer has driving license or not



data['Driving_flag'].value_counts()
# customer has voter-id card or not



data['VoterID_flag'].value_counts()
# customer has pan card or not



data['PAN_flag'].value_counts()
# customer has aadhar card or not



data['Aadhar_flag'].value_counts()
# customer has shared the mobile no. or not



data['MobileNo_Avl_Flag'].value_counts()
# lets extract features from disbursal dates

# as all  the disbursement dates are of year 2018 so we can extract the months



data['DisbursalDate'] = pd.to_datetime(data['DisbursalDate'], errors = 'coerce')

train['DisbursalDate'] = pd.to_datetime(train['DisbursalDate'], errors = 'coerce')

test['DisbursalDate'] = pd.to_datetime(test['DisbursalDate'], errors = 'coerce')



# extracting the month of the disbursement

data['DisbursalMonth'] = data['DisbursalDate'].dt.month

train['DisbursalMonth'] = train['DisbursalDate'].dt.month

test['DisbursalMonth'] = test['DisbursalDate'].dt.month



add_datepart(data, 'DisbursalDate')

add_datepart(train, 'DisbursalDate')

add_datepart(test, 'DisbursalDate')

data['DisbursalMonth'].value_counts()
# plotting the Disbursal date



plt.rcParams['figure.figsize'] = (18, 5)

sns.countplot(data['DisbursalMonth'], palette = 'colorblind')

plt.title('Months', fontsize = 30)

plt.show()
# some attributes are categorical but they are in integer so let's convert them into category



# data['branch_id'] = data['branch_id'].astype('category')

# data['manufacturer_id'] = data['manufacturer_id'].astype('category')

# data['State_ID'] = data['State_ID'].astype('category')



# train['branch_id'] = train['branch_id'].astype('category')

# train['manufacturer_id'] = train['manufacturer_id'].astype('category')

# train['State_ID'] = train['State_ID'].astype('category')



# test['branch_id'] = test['branch_id'].astype('category')

# test['manufacturer_id'] = test['manufacturer_id'].astype('category')

# test['State_ID'] = test['State_ID'].astype('category')



# from sklearn.preprocessing import LabelEncoder



# le = LabelEncoder()

# data['branch_id'] = le.fit_transform(data['branch_id'])

# data['manufacturer_id'] = le.fit_transform(data['manufacturer_id'])

# data['State_ID'] = le.fit_transform(data['State_ID'])



# train['branch_id'] = le.fit_transform(train['branch_id'])

# train['manufacturer_id'] = le.fit_transform(train['manufacturer_id'])

# train['State_ID'] = le.fit_transform(train['State_ID'])



# test['branch_id'] = le.fit_transform(test['branch_id'])

# test['manufacturer_id'] = le.fit_transform(test['manufacturer_id'])

# test['State_ID'] = le.fit_transform(test['State_ID'])

# checking the values in these attributes

#data['branch_id'].value_counts()

#data['manufacturer_id'].value_counts()

#data['State_ID'].value_counts()
# removing unnecassary columns



data = data.drop(['Date.of.Birth','UniqueID'], axis = 1)

train = train.drop(['Date.of.Birth', 'UniqueID'], axis = 1)

test = test.drop(['Date.of.Birth', 'UniqueID'], axis = 1)



# checking the new columns of data

data.columns
# looking at the sample of the pre-processed data



data.sample(5)
# checking the target variable



y.value_counts(normalize=True)
# there is a big difference in the no. of values for 1 and 0

# so we can apply SMOTE or over-sampling

# that means replicating the samples of 1 to lessen the parity between 0 and 1 values



# lets install imblearn

#!pip install -U imbalanced-learn
from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')

%time X_smt, y_smt = smt.fit_sample(train, y)
X_smt.shape, y_smt.shape, train.shape,y.shape
from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=200,verbose=1, n_jobs=-1,random_state=42)

%time cv =StratifiedKFold(n_splits=20,shuffle=True,random_state=45)

%time pre_rf = cross_val_predict(rf, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')

roc_auc_score(y_smt, pre_rf[:,1])
rf.fit(X_smt,y_smt)

rf_smote = rf.predict_proba(test)[:,1]

submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': rf_smote})

submission.to_csv('submission_smote_rf.csv',index=False)



submission.head()
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(learning_rate=1.2,n_estimators=150)

%time pre_ada = cross_val_predict(ada, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')

roc_auc_score(y_smt, pre_ada[:,1])
ada.fit(X_smt,y_smt)

ada_smote = ada.predict_proba(test)[:,1]

submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': ada_smote})

submission.to_csv('submission_smote_ada.csv',index=False)



submission.head()
from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(n_estimators = 200, n_jobs = -1,learning_rate = 0.25,eval_metric='merror',reg_alpha=0.1)

%time pre_xgb = cross_val_predict(xgb, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')

roc_auc_score(y_smt, pre_xgb[:,1])
xgb.fit(X_smt,y_smt)

xgb_smote = xgb.predict_proba(test)[:,1]

submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': xgb_smote})

submission.to_csv('submission_smote_xgb.csv',index=False)



submission.head()
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=1,learning_rate=0.15,n_estimators=400,min_child_samples=25,n_jobs=-1,reg_alpha=0.1,reg_lambda=0.1)

%time pre_lgb = cross_val_predict(lgb, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')

roc_auc_score(y_smt, pre_lgb[:,1])
lgb.fit(X_smt,y_smt)

lgb_smote = lgb.predict_proba(test)[:,1]

submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': lgb_smote})

submission.to_csv('submission_smote_lgb.csv',index=False)



submission.head()
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(random_state=1,loss='exponential',learning_rate=0.26,n_estimators=150,max_depth =5,min_samples_leaf=1)

%time pre_gbm = cross_val_predict(gbm, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')

roc_auc_score(y_smt, pre_gbm[:,1])
gbm.fit(X_smt,y_smt)

gbm_smote = gbm.predict_proba(test)[:,1]

submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': gbm_smote})

submission.to_csv('submission_smote_gbm.csv',index=False)



submission.head()
avg_smote = rf_smote*0.17 + ada_smote*0.20 + lgb_smote*0.21 + xgb_smote*0.21 + gbm_smote*0.21

submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': avg_smote})

submission.to_csv('submission_smote_avg.csv',index=False)
avg_smote2 = ada_smote*0.25 + lgb_smote*0.25 + xgb_smote*0.25 + gbm_smote*0.25

submission = pd.DataFrame({'UniqueID': unique_id,'loan_default': avg_smote2})

submission.to_csv('submission_smote_avg2.csv',index=False)
# separating train and test datasets from data



# x_train = train

# x_test = test

# y.shape

# # checking the shape of train and test

# print("Shape of train :", x_train.shape)

# print("Shape of test :", x_test.shape)
# applying SMOTE



# from imblearn.over_sampling import SMOTE



# x_resample, y_resample = SMOTE().fit_sample(x_train, y_train.values.ravel()) 



# # checking the shape of x_resample and y_resample

# print("Shape of x:", x_resample.shape)

# print("Shape of y:", y_resample.shape)


# # train and valid sets from train

# from sklearn.model_selection import train_test_split



# x_train, x_valid, y_train, y_valid = train_test_split(train, y, test_size = 0.2, random_state = 0)



# # checking the shapes

# print(x_train.shape)

# print(y_train.shape)

# print(x_valid.shape)

# print(y_valid.shape)
# applying standardization



# standardization



from sklearn.preprocessing import StandardScaler



# sc = StandardScaler()

# x_train = sc.fit_transform(x_train)

# x_valid = sc.transform(x_valid)

# x_test = sc.transform(x_test)
# RANDOM FOREST CLASSIFIER



from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import classification_report



# model_rf = RandomForestClassifier(n_estimators =150,n_jobs=-1,random_state=1)

# model_rf.fit(x_train, y_train)



# y_pred = model_rf.predict(x_valid)



# print("Training Accuracy: ", model_rf.score(x_train, y_train))

# print('Testing Accuarcy: ', model_rf.score(x_valid, y_valid))

# print('Traing AUC:',roc_auc_score(y_train,model_rf.predict_proba(x_train)[:,1]))

# print('Testing AUC:',roc_auc_score(y_valid,model_rf.predict_proba(x_valid)[:,1]))



# # making a classification report

# cr = classification_report(y_valid,  y_pred)

# print(cr)



# # making a confusion matrix

# cm = confusion_matrix(y_valid, y_pred)

# sns.heatmap(cm, annot = True)
# # getting the avg precision score

# from sklearn.metrics import average_precision_score



# apc = average_precision_score(y_valid, y_pred)

# print('Average Precision Score :', apc)
##### plotting an AUC ROC Curve



# from sklearn.metrics import precision_recall_curve

# from sklearn.utils.fixes import signature



# precision, recall, _ = precision_recall_curve(y_valid, y_pred)



# step_kwargs = ({'step':'post'} if 'step' in signature(plt.fill_between).parameters else{})



# plt.step(recall, precision, color = 'red', alpha = 0.6, where = 'post')

# plt.fill_between(recall, precision, color = 'red', alpha = 0.6, **step_kwargs)



# plt.title('Precision Recall Curve')

# plt.xlabel('Recall', fontsize = 15)

# plt.ylabel('Precision', fontsize =15)

# plt.ylim([0.0, 1.05])

# plt.xlim([0.0, 1.0])
# ADA BOOST CLASSIFIER



from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



# model_ada = AdaBoostClassifier(learning_rate=1.2,n_estimators=100)

# model_ada.fit(x_train, y_train)



# y_pred = model_ada.predict(x_valid)



# print("Training Accuracy: ", model_ada.score(x_train, y_train))

# print('Testing Accuarcy: ', model_ada.score(x_valid, y_valid))

# print('Traing AUC:',roc_auc_score(y_train,model_ada.predict_proba(x_train)[:,1]))

# print('Testing AUC:',roc_auc_score(y_valid,model_ada.predict_proba(x_valid)[:,1]))



# # making a classification report

# cr = classification_report(y_valid,  y_pred)

# print(cr)



# # making a confusion matrix

# cm = confusion_matrix(y_valid, y_pred)

# sns.heatmap(cm, annot = True)
# # getting the avg precision score

from sklearn.metrics import average_precision_score



# apc = average_precision_score(y_valid, y_pred)

# print('Average Precision Score :', apc)
# plotting an AUC ROC Curve



from sklearn.metrics import precision_recall_curve

from sklearn.utils.fixes import signature



# precision, recall, _ = precision_recall_curve(y_valid, y_pred)



# step_kwargs = ({'step':'post'} if 'step' in signature(plt.fill_between).parameters else{})



# plt.step(recall, precision, color = 'pink', alpha = 0.6, where = 'post')

# plt.fill_between(recall, precision, color = 'pink', alpha = 0.6, **step_kwargs)



# plt.title('Precision Recall Curve')

# plt.xlabel('Recall', fontsize = 15)

# plt.ylabel('Precision', fontsize =15)

# plt.ylim([0.0, 1.05])

# plt.xlim([0.0, 1.0])
# Xg-Boost Classifier



from xgboost.sklearn import XGBClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



# model_xgb = XGBClassifier(n_estimators = 200, n_jobs = -1,learning_rate = 0.25,eval_metric='merror',reg_alpha=0.1)

# model_xgb.fit(x_train, y_train)



# y_pred = model_xgb.predict(x_valid)



# print("Training Accuracy: ", model_xgb.score(x_train, y_train))

# print('Testing Accuarcy: ', model_xgb.score(x_valid, y_valid))

# print('Traing AUC:',roc_auc_score(y_train,model_xgb.predict_proba(x_train)[:,1]))

# print('Testing AUC:',roc_auc_score(y_valid,model_xgb.predict_proba(x_valid)[:,1]))

# # making a classification report

# cr = classification_report(y_valid,  y_pred)

# print(cr)



# # making a confusion matrix

# cm = confusion_matrix(y_valid, y_pred)

# sns.heatmap(cm, annot = True)
# # getting the avg precision score

# from sklearn.metrics import average_precision_score



# apc = average_precision_score(y_valid, y_pred)

# print('Average Precision Score :', apc)
# # plotting an AUC ROC Curve



# from sklearn.metrics import precision_recall_curve

# from sklearn.utils.fixes import signature



# precision, recall, _ = precision_recall_curve(y_valid, y_pred)



# step_kwargs = ({'step':'post'} if 'step' in signature(plt.fill_between).parameters else{})



# plt.step(recall, precision, color = 'pink', alpha = 0.6, where = 'post')

# plt.fill_between(recall, precision, color = 'pink', alpha = 0.6, **step_kwargs)



# plt.title('Precision Recall Curve')

# plt.xlabel('Recall', fontsize = 15)

# plt.ylabel('Precision', fontsize =15)

# plt.ylim([0.0, 1.05])

# plt.xlim([0.0, 1.0])
# light boost classifier



from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



# model_lgb = LGBMClassifier(random_state=1,learning_rate=0.15,n_estimators=400,min_child_samples=25,n_jobs=-1,reg_alpha=0.1,reg_lambda=0.1)

# model_lgb.fit(x_train, y_train)



# y_pred = model_lgb.predict(x_valid)



# print("Training Accuracy: ", model_lgb.score(x_train, y_train))

# print('Testing Accuarcy: ', model_lgb.score(x_valid, y_valid))

# print('Traing AUC:',roc_auc_score(y_train,model_lgb.predict_proba(x_train)[:,1]))

# print('Testing AUC:',roc_auc_score(y_valid,model_lgb.predict_proba(x_valid)[:,1]))



# # making a classification report

# cr = classification_report(y_valid,  y_pred)

# print(cr)



# # making a confusion matrix

# cm = confusion_matrix(y_valid, y_pred)

# sns.heatmap(cm, annot = True)
# # getting the avg precision score

from sklearn.metrics import average_precision_score



# apc = average_precision_score(y_valid, y_pred)

# print('Average Precision Score :', apc)
# # plotting an AUC ROC Curve



from sklearn.metrics import precision_recall_curve

from sklearn.utils.fixes import signature



# precision, recall, _ = precision_recall_curve(y_valid, y_pred)



# step_kwargs = ({'step':'post'} if 'step' in signature(plt.fill_between).parameters else{})



# plt.step(recall, precision, color = 'pink', alpha = 0.6, where = 'post')

# plt.fill_between(recall, precision, color = 'pink', alpha = 0.6, **step_kwargs)



# plt.title('Precision Recall Curve')

# plt.xlabel('Recall', fontsize = 15)

# plt.ylabel('Precision', fontsize =15)

# plt.ylim([0.0, 1.05])

# plt.xlim([0.0, 1.0])
# GBM  classifier



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



# model_gbm = GradientBoostingClassifier(random_state=1,loss='exponential',learning_rate=0.26,n_estimators=150,max_depth =5,min_samples_leaf=1)

# model_gbm.fit(x_train, y_train)



# y_pred = model_gbm.predict(x_valid)



# print("Training Accuracy: ", model_gbm.score(x_train, y_train))

# print('Testing Accuarcy: ', model_gbm.score(x_valid, y_valid))

# print('Traing AUC:',roc_auc_score(y_train,model_gbm.predict_proba(x_train)[:,1]))

# print('Testing AUC:',roc_auc_score(y_valid,model_gbm.predict_proba(x_valid)[:,1]))



# # making a classification report

# cr = classification_report(y_valid,  y_pred)

# print(cr)



# # making a confusion matrix

# cm = confusion_matrix(y_valid, y_pred)

# sns.heatmap(cm, annot = True)
# getting the avg precision score

from sklearn.metrics import average_precision_score



# apc = average_precision_score(y_valid, y_pred)

# print('Average Precision Score :', apc)
# plotting an AUC ROC Curve



from sklearn.metrics import precision_recall_curve

from sklearn.utils.fixes import signature



# precision, recall, _ = precision_recall_curve(y_valid, y_pred)



# step_kwargs = ({'step':'post'} if 'step' in signature(plt.fill_between).parameters else{})



# plt.step(recall, precision, color = 'pink', alpha = 0.6, where = 'post')

# plt.fill_between(recall, precision, color = 'pink', alpha = 0.6, **step_kwargs)



# plt.title('Precision Recall Curve')

# plt.xlabel('Recall', fontsize = 15)

# plt.ylabel('Precision', fontsize =15)

# plt.ylim([0.0, 1.05])

# plt.xlim([0.0, 1.0])
# let's plot the feature importance plot for the lg boost model

# feature = pd.DataFrame()

# x_train = pd.DataFrame(x_train)

# feature['name'] = x_train.columns

# feature['importance'] = model_lgb.feature_importances_



# feature.sort_values(by = ['importance'], ascending = True, inplace = True)

# feature.set_index('name', inplace = True)



# feature.plot(kind = 'barh', color = 'purple', figsize = (5, 15), fontsize = 10)
# ## fit all the models on whole training set (No validation set needed) before predicting on testing set



# model_rf.fit(train, y)

# model_ada.fit(train, y)

# model_xgb.fit(train, y)

# model_lgb.fit(train, y)

# model_gbm.fit(train, y)
# # let's predict for the tests set

# y_pred_rf = model_rf.predict_proba(x_test)[:,1]

# y_pred_ada = model_ada.predict_proba(x_test)[:,1]

# y_pred_xgb = model_xgb.predict_proba(x_test)[:,1]

# y_pred_lgb = model_lgb.predict_proba(x_test)[:,1]

# y_pred_gbm = model_lgb.predict_proba(x_test)[:,1]
# ##Boosting.

## XGB, LGB, GBM performed better than RF and ADA on validation set. let's give more weiht to them

#avg_pred = y_pred_ada

# avg_pred = y_pred_rf*0.17 + y_pred_ada*0.20 + y_pred_lgb*0.21 + y_pred_xgb*0.21 + y_pred_gbm*0.21
# lets look at the submission file

submission = pd.read_csv('../input/sample_submission_24jSKY6.csv')

submission.head()
#  let's create a submission file

#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

# submission_avg = pd.DataFrame({'UniqueID': unique_id,'loan_default': avg_pred})

# submission_avg.to_csv('submission_avg.csv',index=False)

# #Visualize the first 5 rows

# submission_avg.head()
# submission_rf = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_rf})

# submission_rf.to_csv('submission_rf.csv',index=False)

# #Visualize the first 5 rows

# submission_rf.head()
# submission_ada = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_ada})

# submission_ada.to_csv('submission_ada.csv',index=False)

# #Visualize the first 5 rows

# submission_ada.head()
# submission_xgb = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_xgb})

# submission_xgb.to_csv('submission_xgb.csv',index=False)

# #Visualize the first 5 rows

# submission_xgb.head()
# submission_lgb = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_lgb})

# submission_lgb.to_csv('submission_lgb.csv',index=False)

# #Visualize the first 5 rows

# submission_lgb.head()
# submission_gbm = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_gbm})

# submission_gbm.to_csv('submission_gbm.csv',index=False)

# #Visualize the first 5 rows

# submission_gbm.head()
from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

#filename = 'submission.csv'

# submission.to_csv(filename,index=False)



# print('Saved file: ' + filename)

# create_download_link(filename='submission1.csv')
# display_all(train.head())
# def k_folds_xgb(X, y, X_test, k,n_est):

#     folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)

#     y_test = np.zeros((X_test.shape[0], 2))

#     y_oof = np.zeros((X.shape[0]))

#     score = 0

#     for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

#         clf =  XGBClassifier(n_estimators = n_est, n_jobs = -1,learning_rate = 0.25,eval_metric='merror',reg_alpha=0.1)

#         clf.fit(X.iloc[train_idx], y[train_idx])

#         y_oof[val_idx] = clf.predict(X.iloc[val_idx])

#         y_test += clf.predict_proba(X_test) / folds.n_splits

#         score += roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])

#         print('Fold: {} score: {}'.format(i,roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])))

#     print('Avg ROC', score / folds.n_splits) 

        

#     return y_oof, y_test 
# y_oof, y_pred_kxgb = k_folds_xgb(X_smt, y_smt, test, k= 2,n_est=5)
# submission_kxgb = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_kxgb[:,1]})

# submission_kxgb.to_csv('submission_kxgb.csv',index=False)

# #Visualize the first 5 rows

# submission_kxgb.head()
# def k_folds_gbm(X, y, X_test, k,n_est):

#     folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)

#     y_test = np.zeros((X_test.shape[0], 2))

#     y_oof = np.zeros((X.shape[0]))

#     score = 0

#     for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

#         clf =  GradientBoostingClassifier(random_state=1,loss='exponential',learning_rate=0.26,n_estimators=n_est,max_depth =5,min_samples_leaf=1)

#         clf.fit(X.iloc[train_idx], y[train_idx])

#         y_oof[val_idx] = clf.predict(X.iloc[val_idx])

#         y_test += clf.predict_proba(X_test) / folds.n_splits

#         score += roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])

#         print('Fold: {} score: {}'.format(i,roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])))

#     print('Avg ROC', score / folds.n_splits) 

        

#     return y_oof, y_test 
# y_oof, y_pred_kgbm = k_folds_gbm(train, y, x_test, k= 20,n_est=200)
# submission_kgbm = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_kgbm[:,1]})

# submission_kgbm.to_csv('submission_kgbm.csv',index=False)

# #Visualize the first 5 rows

# submission_kgbm.head()
# def k_folds_ada(X, y, X_test, k,n_est):

#     folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)

#     y_test = np.zeros((X_test.shape[0], 2))

#     y_oof = np.zeros((X.shape[0]))

#     score = 0

#     for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

#         clf =  AdaBoostClassifier(learning_rate=1.2,n_estimators=n_est)

#         clf.fit(X.iloc[train_idx], y[train_idx])

#         y_oof[val_idx] = clf.predict(X.iloc[val_idx])

#         y_test += clf.predict_proba(X_test) / folds.n_splits

#         score += roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])

#         print('Fold: {} score: {}'.format(i,roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])))

#     print('Avg ROC', score / folds.n_splits) 

        

#     return y_oof, y_test 
# y_oof, y_pred_kada = k_folds_ada(train, y, x_test, k= 20,n_est=200)
# submission_kada = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_kada[:,1]})

# submission_kada.to_csv('submission_kada.csv',index=False)

# #Visualize the first 5 rows

# submission_kada.head()
# def k_folds_rf(X, y, X_test, k,n_est):

#     folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)

#     y_test = np.zeros((X_test.shape[0], 2))

#     y_oof = np.zeros((X.shape[0]))

#     score = 0

#     for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

#         clf =  RandomForestClassifier(n_estimators =n_est,n_jobs=-1,random_state=1)

#         clf.fit(X.iloc[train_idx], y[train_idx])

#         y_oof[val_idx] = clf.predict(X.iloc[val_idx])

#         y_test += clf.predict_proba(X_test) / folds.n_splits

#         score += roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])

#         print('Fold: {} score: {}'.format(i,roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])))

#     print('Avg ROC', score / folds.n_splits) 

        

#     return y_oof, y_test 
# y_oof, y_pred_krf = k_folds_rf(train, y, x_test, k= 20,n_est=250)
# submission_krf = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_krf[:,1]})

# submission_krf.to_csv('submission_krf.csv',index=False)

# #Visualize the first 5 rows

# submission_krf.head()
# def k_folds_lgb(X, y, X_test, k,n_est):

#     folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)

#     y_test = np.zeros((X_test.shape[0], 2))

#     y_oof = np.zeros((X.shape[0]))

#     score = 0

#     for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

#         clf =  LGBMClassifier(random_state=1,learning_rate=0.15,n_estimators=n_est,min_child_samples=25,n_jobs=-1,reg_alpha=0.1,reg_lambda=0.1)

#         clf.fit(X.iloc[train_idx], y[train_idx])

#         y_oof[val_idx] = clf.predict(X.iloc[val_idx])

#         y_test += clf.predict_proba(X_test) / folds.n_splits

#         score += roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])

#         print('Fold: {} score: {}'.format(i,roc_auc_score(y[val_idx],clf.predict_proba(X.iloc[val_idx])[:,1])))

#     print('Avg ROC', score / folds.n_splits) 

        

#     return y_oof, y_test 
# y_oof, y_pred_klgb = k_folds_lgb(train, y, x_test, k= 20,n_est=500)
# submission_klgb = pd.DataFrame({'UniqueID': unique_id,'loan_default': y_pred_klgb[:,1]})

# submission_klgb.to_csv('submission_klgb.csv',index=False)

# #Visualize the first 5 rows

# submission_klgb.head()
# # avg_pred2 = y_pred_krf*0.17 + y_pred_kada*0.20 + y_pred_kxgb*0.21 + y_pred_klgb*0.21 + y_pred_kgbm*0.21

# avg_pred =  (y_pred_kxgb*0.28 + y_pred_klgb*0.28 + y_pred_kgbm*0.28 + y_pred_kada*0.16)

# submission_avg = pd.DataFrame({'UniqueID': unique_id,'loan_default': avg_pred[:,1]})

# submission_avg.to_csv('submission_avg.csv',index=False)

# #Visualize the first 5 rows

# submission_avg.head()
# display_all(train.head())
# display_all(test.head())
# train['loan_default'] = y

# train.shape, test.shape, y.shape
# dep_var = 'loan_default'

# cat_vars = ['branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Employment.Type', 'State_ID', 'Employee_code_ID',

#            'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag', 'NEW.ACCTS.IN.LAST.SIX.MONTHS',

#            'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','NO.OF_INQUIRIES','Year_of_birth','DisbursalMonth','DisbursalYear','DisbursalWeek',

#            'DisbursalDay','DisbursalDayofweek','DisbursalDayofyear','DisbursalIs_month_end','DisbursalIs_month_start',

#            'DisbursalIs_quarter_end','DisbursalIs_quarter_start','DisbursalIs_year_end','DisbursalIs_year_start']

# cont_vars = ['disbursed_amount','asset_cost','ltv','PERFORM_CNS.SCORE','PERFORM_CNS.SCORE.DESCRIPTION','PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS',

#              'PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT','SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS',

#              'SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT',

#              'AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','DisbursalElapsed']

# procs = [Categorify]
# data = (TabularList.from_df(train, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)

#                 .split_by_idx(list(range(55000,100000)))

#                 .label_from_df(cols=dep_var)

#                 .add_test(TabularList.from_df(test, path=path, cat_names=cat_vars, cont_names=cont_vars))

#                 .databunch())
# data.show_batch(rows=5)
# learn = tabular_learner(data, layers=[500,100], ps=[0.001,0.01], emb_drop=0.04, 

#                          metrics=accuracy)
# learn.model
# learn.lr_find()

# learn.recorder.plot()
# learn.fit_one_cycle(3, 1e-2, wd=0.15)
# learn.recorder.plot_losses()
# learn.fit_one_cycle(1, 3e-4)
#learn.TTA(DatasetType.Test)
# test_preds = learn.get_preds(DatasetType.Test)[0][:,1]

# submit = pd.DataFrame({'UniqueID':unique_id, 'loan_default':test_preds}, columns=['UniqueID', 'loan_default'])

# submit.to_csv('submit_1.csv',index=False)

# submit.head()
#create_download_link(filename='submit.csv')
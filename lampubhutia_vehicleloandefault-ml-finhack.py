# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

%matplotlib inline

from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (10, 8)

plt.rcParams['font.size'] = 14
# Importing the train and test datasets



train = pd.read_csv("../input/train_LTFS.csv")

test = pd.read_csv("../input/test_LTFS.csv")
train_original=train.copy() 

test_original=test.copy()
train.columns, train.shape
test.columns, test.shape
# Print data types for each variable 

print(train.dtypes)
train.isnull().sum(),train.shape
test.isnull().sum(),test.shape
# Filling Missing values



train['Employment.Type'].fillna('Unemployed', inplace = True)

test['Employment.Type'].fillna('Unemployed', inplace = True)



# let's check if there is any null values still left or not

print("Null values left in the train set:", train.isnull().sum().sum())

print("Null values left in the test set:", test.isnull().sum().sum())
train['loan_default'].value_counts()



# Normalize can be set to True to print proportions instead of number 

train['loan_default'].value_counts(normalize=True)



train['loan_default'].value_counts().plot.bar()
matrix = train.corr() 

f, ax = plt.subplots(figsize=(30, 18)) 

sns.heatmap(matrix, vmax=1.5, square=True,annot=True, fmt=".1f",cmap="BuPu")

plt.show()
#performing log transformations on disbursed amount, ltv, and asset cost



# training dataset

train['disbursed_amount'] = np.log1p(train['disbursed_amount'])

train['ltv'] = np.log1p(train['ltv'])

train['asset_cost'] = np.log1p(train['asset_cost'])



# test data set

test['disbursed_amount'] = np.log1p(test['disbursed_amount'])

test['ltv'] = np.log1p(test['ltv'])

test['asset_cost'] = np.log1p(test['asset_cost'])



# plotting training dataset

plt.rcParams['figure.figsize'] = (18, 5)



plt.subplot(1, 3, 1)

sns.distplot(train['disbursed_amount'],  color = 'orange')

plt.title('Disburesed Amount')



plt.subplot(1, 3, 2)

sns.distplot(train['asset_cost'], color = 'pink')

plt.title('Asset Cost')



plt.subplot(1, 3, 3)

sns.distplot(train['ltv'], color = 'red')

plt.title('Loan to value of the asset')



plt.show()
print("Total no. of Unique Ids :", train['UniqueID'].nunique())

print("Total no. of Unique Branches :", train['branch_id'].nunique())

print("Total no. of Unique Suppliers :", train['supplier_id'].nunique())

print("Total no. of Unique Manufactures :", train['manufacturer_id'].nunique())

print("Total no. of Unique Current pincode Ids :", train['Current_pincode_ID'].nunique())

print("Total no. of Unique State IDs :", train['State_ID'].nunique())

print("Total no. of Unique Employee code IDs :", train['Employee_code_ID'].nunique())

print("Total no. of Unique Ids :", test['UniqueID'].nunique())

print("Total no. of Unique Branches :", test['branch_id'].nunique())

print("Total no. of Unique Suppliers :", test['supplier_id'].nunique())

print("Total no. of Unique Manufactures :", test['manufacturer_id'].nunique())

print("Total no. of Unique Current pincode Ids :", test['Current_pincode_ID'].nunique())

print("Total no. of Unique State IDs :", test['State_ID'].nunique())

print("Total no. of Unique Employee code IDs :", test['Employee_code_ID'].nunique())
# normalizing the value 

plt.figure(1) 

plt.subplot(311) 

train['manufacturer_id'].value_counts(normalize=True).plot.bar(figsize=(24,10), title= 'manufacturer_id', fontsize=14) 

plt.subplot(312) 

train['State_ID'].value_counts(normalize=True).plot.bar(title= 'State_ID',fontsize=14) 

plt.subplot(313) 

train['branch_id'].value_counts(normalize=True).plot.bar(title= 'branch_id', fontsize=14) 



plt.show()
# converting the DOB in date-time-format to extract the year of birth 



train['Date.of.Birth'] = pd.to_datetime(train['Date.of.Birth'],errors = 'coerce')



# extracting the year of birth of the customers

train['Year_of_birth'] = train['Date.of.Birth'].dt.year



# checking the values inside date of year

sns.distplot(train['Year_of_birth'], color = 'green')

plt.title('Distribution of Year of birth')
# Changing Employment.Type dtype object to int64



train['Employment.Type'] = train['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))



# checking the values  of employement type

train['Employment.Type'].value_counts()
test['Employment.Type'] = test['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))



# checking the values  of employement type

test['Employment.Type'].value_counts()






#Visualizing the Employment Type



sns.countplot(x='Employment.Type',data=train)



plt.show()



train['Employment.Type'].value_counts()
# features extraction from disbursal dates

# Extracting months as all disbursement done in year 2018.



train['DisbursalDate'] = pd.to_datetime(train['DisbursalDate'], errors = 'coerce')



# extracting the month of the disbursement

train['DisbursalMonth'] = train['DisbursalDate'].dt.month



train['DisbursalMonth'].value_counts()



plt.rcParams['figure.figsize'] = (18, 5)

sns.countplot(train['DisbursalMonth'], palette = 'colorblind')

plt.title('Months', fontsize = 30)

# customer has aadhar card or not

sns.countplot(x="Aadhar_flag", data=train)



train['Aadhar_flag'].value_counts()

# customer has shared the mobile no. or not



sns.countplot(x="MobileNo_Avl_Flag", data=train)



train['MobileNo_Avl_Flag'].value_counts()
# customer has pan card or not

sns.countplot(x="PAN_flag", data=train)

train['PAN_flag'].value_counts()
# customer shared voter-id card or not

sns.countplot(x="VoterID_flag", data=train)

train['VoterID_flag'].value_counts()
# customer shared driving license or not

sns.countplot(x="Driving_flag", data=train)

train['Driving_flag'].value_counts()
# customer shared passport or not

sns.countplot(x="Passport_flag", data=train)

train['Passport_flag'].value_counts()
# checking the perform cns score description



sns.countplot(x='PERFORM_CNS.SCORE.DESCRIPTION',data=train)

plt.xticks(rotation = 90)

plt.show()



train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
# encodings for bureau score(perform cns score distribution)



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



# checing the values in bureau score

train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()
# encodings for bureau score(perform cns score distribution)



test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('No Bureau History Available', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Sufficient History Not Available', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Not Enough Info available on the customer', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Activity seen on the customer (Inactive)',0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: No Updates available in last 36 months', 0)

test['PERFORM_CNS.SCORE.DESCRIPTION'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].replace('Not Scored: Only a Guarantor', 0)

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

sns.distplot(train['PERFORM_CNS.SCORE'], color = 'green')

plt.title('Before Log transformations')



# tranforming to log 



plt.subplot(1, 2, 2)

train['PERFORM_CNS.SCORE'] = np.log1p(train['PERFORM_CNS.SCORE'])

sns.distplot(train['PERFORM_CNS.SCORE'], color = 'blue')

plt.title('After Log transformations')

plt.show()



# for test

test['PERFORM_CNS.SCORE'] = np.log1p(test['PERFORM_CNS.SCORE'])

#  applying log transformations to the primary account attributes



train['PRI.NO.OF.ACCTS'] = np.log1p(train['PRI.NO.OF.ACCTS'])

train['PRI.ACTIVE.ACCTS'] = np.log1p(train['PRI.ACTIVE.ACCTS'])

train['PRI.OVERDUE.ACCTS'] = np.log1p(train['PRI.OVERDUE.ACCTS'])

#train['PRI.CURRENT.BALANCE'] = np.log1p(train['PRI.CURRENT.BALANCE'])

#train['PRI.SANCTIONED.AMOUNT'] = np.log1p(train['PRI.SANCTIONED.AMOUNT'])

train['PRI.DISBURSED.AMOUNT'] = np.log1p(train['PRI.DISBURSED.AMOUNT'])





#  filling  missing values in sec.current.balance

train['PRI.CURRENT.BALANCE'].fillna(train['PRI.CURRENT.BALANCE'].mean(), inplace = True)

train['PRI.SANCTIONED.AMOUNT'].fillna(train['PRI.SANCTIONED.AMOUNT'].mean(), inplace = True)



#  for test

test['PRI.NO.OF.ACCTS'] = np.log1p(test['PRI.NO.OF.ACCTS'])

test['PRI.ACTIVE.ACCTS'] = np.log1p(test['PRI.ACTIVE.ACCTS'])

test['PRI.OVERDUE.ACCTS'] = np.log1p(test['PRI.OVERDUE.ACCTS'])

#test['PRI.CURRENT.BALANCE'] = np.log1p(test['PRI.CURRENT.BALANCE'])

#test['PRI.SANCTIONED.AMOUNT'] = np.log1p(test['PRI.SANCTIONED.AMOUNT'])

test['PRI.DISBURSED.AMOUNT'] = np.log1p(test['PRI.DISBURSED.AMOUNT'])





#  filling  missing values in sec.current.balance

test['PRI.CURRENT.BALANCE'].fillna(test['PRI.CURRENT.BALANCE'].mean(), inplace = True)

test['PRI.SANCTIONED.AMOUNT'].fillna(test['PRI.SANCTIONED.AMOUNT'].mean(), inplace = True)







# plotting distribution plots for these attributes



plt.rcParams['figure.figsize'] = (20, 16)    

plt.subplot(2, 3, 1)

sns.distplot(train['PRI.NO.OF.ACCTS'], color = 'violet')

plt.title('Total loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.distplot(train['PRI.ACTIVE.ACCTS'], color = 'violet')

plt.title('Active loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.distplot(train['PRI.OVERDUE.ACCTS'], color = 'violet')

plt.title('Default Accounts')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.distplot(train['PRI.CURRENT.BALANCE'], color = 'violet')

plt.title('Principal Outstanding amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.distplot(train['PRI.SANCTIONED.AMOUNT'], color = 'violet')

plt.title('Total Sanctioned Amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.distplot(train['PRI.DISBURSED.AMOUNT'], color = 'violet')

plt.title('Total Disbured Amount')

plt.xticks(rotation = 45)



plt.show()
# distribution for different attributesof secondary accounts





plt.rcParams['figure.figsize'] = (20, 14)    

plt.subplot(2, 3, 1)

sns.distplot(train['SEC.NO.OF.ACCTS'], color = 'red')

plt.title('Total loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.distplot(train['SEC.ACTIVE.ACCTS'], color = 'red')

plt.title('Active loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.distplot(train['SEC.OVERDUE.ACCTS'], color = 'red')

plt.title('Default Accounts at the time of disbursement')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.distplot(train['SEC.CURRENT.BALANCE'], color = 'red')

plt.title('Principal Outstanding amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.distplot(train['SEC.SANCTIONED.AMOUNT'], color = 'red')

plt.title('Total Sanctioned Amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.distplot(train['SEC.DISBURSED.AMOUNT'], color = 'red')

plt.title('Total Disbured Amount')

plt.xticks(rotation = 45)



plt.show()
train['SEC.NO.OF.ACCTS'] = np.log1p(train['SEC.NO.OF.ACCTS'])

train['SEC.ACTIVE.ACCTS'] = np.log1p(train['SEC.ACTIVE.ACCTS'])

train['SEC.OVERDUE.ACCTS'] = np.log1p(train['SEC.OVERDUE.ACCTS'])



train['SEC.SANCTIONED.AMOUNT'] = np.log1p(train['SEC.SANCTIONED.AMOUNT'])

train['SEC.DISBURSED.AMOUNT'] = np.log1p(train['SEC.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

train['SEC.CURRENT.BALANCE'].fillna(train['SEC.CURRENT.BALANCE'].mean(), inplace = True)





# for test 





test['SEC.NO.OF.ACCTS'] = np.log1p(test['SEC.NO.OF.ACCTS'])

test['SEC.ACTIVE.ACCTS'] = np.log1p(test['SEC.ACTIVE.ACCTS'])

test['SEC.OVERDUE.ACCTS'] = np.log1p(test['SEC.OVERDUE.ACCTS'])



test['SEC.SANCTIONED.AMOUNT'] = np.log1p(test['SEC.SANCTIONED.AMOUNT'])

test['SEC.DISBURSED.AMOUNT'] = np.log1p(test['SEC.DISBURSED.AMOUNT'])



#  filling  missing values in sec.current.balance

test['SEC.CURRENT.BALANCE'].fillna(test['SEC.CURRENT.BALANCE'].mean(), inplace = True)
plt.rcParams['figure.figsize'] = (20, 16)    

plt.subplot(2, 3, 1)

sns.distplot(train['SEC.NO.OF.ACCTS'], color = 'blue')

plt.title('Total loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 2)

sns.distplot(train['SEC.ACTIVE.ACCTS'], color = 'blue')

plt.title('Active loan taken by customer')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 3)

sns.distplot(train['SEC.OVERDUE.ACCTS'], color = 'blue')

plt.title('Default Accounts')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 4)

sns.distplot(train['SEC.CURRENT.BALANCE'], color = 'blue')

plt.title('Principal Outstanding amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 5)

sns.distplot(train['SEC.SANCTIONED.AMOUNT'], color = 'blue')

plt.title('Total Sanctioned Amount')

plt.xticks(rotation = 45)



plt.subplot(2, 3, 6)

sns.distplot(train['SEC.DISBURSED.AMOUNT'], color = 'blue')

plt.title('Total Disbured Amount')

plt.xticks(rotation = 45)



plt.show()
# EMI Amount of the Secondary Plan



plt.subplot(1, 2, 1)

sns.distplot(train['SEC.INSTAL.AMT'])

plt.title('EMI Amount Secondary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.subplot(1, 2, 2)

sns.distplot(train['PRIMARY.INSTAL.AMT'])

plt.title('EMI Amount Primary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.show()
#performing log transformations



train['PRIMARY.INSTAL.AMT'] = np.log1p(train['PRIMARY.INSTAL.AMT'])

train['SEC.INSTAL.AMT'] = np.log1p(train['SEC.INSTAL.AMT'])





plt.subplot(1, 2, 1)

sns.distplot(train['SEC.INSTAL.AMT'], color = 'yellow')

plt.title('EMI Amount Secondary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.subplot(1, 2, 2)

sns.distplot(train['PRIMARY.INSTAL.AMT'],color = 'yellow')

plt.title('EMI Amount Primary Plan', fontsize = 20)

plt.xticks(rotation = 45)



plt.show()



# test



test['PRIMARY.INSTAL.AMT'] = np.log1p(test['PRIMARY.INSTAL.AMT'])

test['SEC.INSTAL.AMT'] = np.log1p(test['SEC.INSTAL.AMT'])
# New Accts in last six months

train['NEW.ACCTS.IN.LAST.SIX.MONTHS'].value_counts()
# loans defaulted accounts in last six months

train['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].value_counts()
plt.subplot(1, 2, 1)

sns.distplot(train['NEW.ACCTS.IN.LAST.SIX.MONTHS'])

plt.title('NEW.ACCTS.IN.LAST.SIX.MONTHS', fontsize = 20)

plt.xticks(rotation = 45)



plt.subplot(1, 2, 2)

sns.distplot(train['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'])

plt.title('DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', fontsize = 20)

plt.xticks(rotation = 45)



plt.show()
# average.acct.age i.e., average loan tenure



sns.countplot(train['AVERAGE.ACCT.AGE'].head(50), palette = 'colorblind')

plt.title('Average Loan Tenure')

plt.xticks(rotation = 45)

plt.show()
# Converting the given 'CREDIT.HISTORY.LENGTH' in months



import re



train['CREDIT.HISTORY.LENGTH']= train['CREDIT.HISTORY.LENGTH'].apply(lambda x: (re.sub('[a-z]','',x)).split())

train['CREDIT.HISTORY.LENGTH']= train['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0])*12+int(x[1]))



# Converting the given 'AVERAGE.ACCT.AGE' in months

train['AVERAGE.ACCT.AGE']= train['AVERAGE.ACCT.AGE'].apply(lambda x: (re.sub('[a-z]','',x)).split())

train['AVERAGE.ACCT.AGE']= train['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0])*12+int(x[1]))



# Converting the given 'CREDIT.HISTORY.LENGTH' in months

test['CREDIT.HISTORY.LENGTH']= test['CREDIT.HISTORY.LENGTH'].apply(lambda x: (re.sub('[a-z]','',x)).split())

test['CREDIT.HISTORY.LENGTH']= test['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0])*12+int(x[1]))



# Converting the given 'AVERAGE.ACCT.AGE' in months

test['AVERAGE.ACCT.AGE']= test['AVERAGE.ACCT.AGE'].apply(lambda x: (re.sub('[a-z]','',x)).split())

test['AVERAGE.ACCT.AGE']= test['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0])*12+int(x[1]))
# distribution of AVERAGE LOAN TENURE in years

plt.title('AVERAGE LOAN TENURE', fontsize = 25)

plt.rcParams['figure.figsize'] = (18, 5)

sns.countplot(train['AVERAGE.ACCT.AGE'].head(50))

#(x='AVERAGE.ACCT.AGE',data=train,palette = 'dark')

plt.show()

#train['AVERAGE.ACCT.AGE'].value_counts()
# plotting credit history of users



plt.rcParams['figure.figsize'] = (18, 5)

sns.countplot(train['CREDIT.HISTORY.LENGTH'].head(50))

plt.title('Credit History')

plt.xticks(rotation = 90)

plt.show()

#train['CREDIT.HISTORY.LENGTH'].value_counts()
sns.countplot(train['NO.OF_INQUIRIES'], palette = 'muted')

plt.title('No. of Inquiries',  fontsize = 30)

plt.show()

train['NO.OF_INQUIRIES'].value_counts()
train['Downpayment']=train['asset_cost']-train['disbursed_amount'] 

test['Downpayment']=test['asset_cost']-test['disbursed_amount']


plt.figure(1) 

plt.subplot(121) 

sns.distplot(train['Downpayment']);

plt.subplot(122) 

train['Downpayment'].plot.box(figsize=(16,5))
train['Downpayment_log'] = np.log(train['Downpayment'])

plt.figure(1) 

plt.subplot(121) 

sns.distplot(train['Downpayment_log']);

plt.subplot(122) 

train['Downpayment_log'].plot.box(figsize=(16,5))

test['Downpayment_log'] = np.log(test['Downpayment'])



# some attributes are categorical but they are in integer so let's convert them into category



train['branch_id'] = train['branch_id'].astype('category')

train['manufacturer_id'] = train['manufacturer_id'].astype('category')

train['State_ID'] = train['State_ID'].astype('category')





test['branch_id'] = test['branch_id'].astype('category')

test['manufacturer_id'] = test['manufacturer_id'].astype('category')

test['State_ID'] = test['State_ID'].astype('category')





from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train['branch_id'] = le.fit_transform(train['branch_id'])

train['manufacturer_id'] = le.fit_transform(train['manufacturer_id'])

train['State_ID'] = le.fit_transform(train['State_ID'])







train.shape, test.shape
train=train.drop(['supplier_id','Current_pincode_ID', 'Date.of.Birth', 'DisbursalDate', 'Employee_code_ID','PRI.DISBURSED.AMOUNT', 'disbursed_amount','PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT','VoterID_flag','PRI.ACTIVE.ACCTS','Year_of_birth','DisbursalMonth','branch_id', 'manufacturer_id', 'State_ID'], axis=1) 

test=test.drop(['supplier_id','Current_pincode_ID', 'Date.of.Birth', 'DisbursalDate', 'Employee_code_ID','PRI.DISBURSED.AMOUNT', 'disbursed_amount','PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT','VoterID_flag','PRI.ACTIVE.ACCTS','branch_id', 'manufacturer_id', 'State_ID'], axis=1)
train.shape, test.shape
X = train.drop('loan_default',1) 

y = train.loan_default
X.head()
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state = 0)
print(x_train.shape)

print(y_train.shape)

print(x_cv.shape)

print(y_cv.shape)
#calling logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression(class_weight='balanced')

logreg.fit(X, y)

print(logreg.coef_)

print(logreg.intercept_)
#fitting the model with x and y attributes of train data

#in this it is goin to learn the pattern

logreg.fit(x_train, y_train)
#now applying our learnt model on test and also on train data

y_log_pred_test = logreg.predict(x_cv)

y_log_pred_train = logreg.predict(x_train)
#creating a confusion matrix to understand the classification

conf = metrics.confusion_matrix(y_cv, y_log_pred_test)

conf
# save confusion matrix and slice into four pieces

confusion = metrics.confusion_matrix(y_cv, y_log_pred_test)

print(confusion)

#[row, column]

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

print ("TP",TP)

print ("TN",TN)

print("FN",FN)

print ("FP",FP)
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(conf,cmap = cmap,xticklabels=['predicted_default_NO=0','predicted_default_yes=1'],yticklabels=['actual_default_NO=0','actual_default_yes=1'],annot=True, fmt="d")
# print the first 25 true and predicted responses

print('True', y_cv.values[0:15])

print('Pred', y_log_pred_test[0:15])
#comparing the metrics of predicted lebel and real label of test data

print('Accuracy_Score:', metrics.accuracy_score(y_cv, y_log_pred_test))
# Method to calculate Classification Error

    



print('Classification Error:',1 - metrics.accuracy_score(y_cv, y_log_pred_test))
# Method to calculate Sensitivity



print('Sensitivity or Recall:', metrics.recall_score(y_cv, y_log_pred_test))
specificity = TN / (TN + FP)



print(specificity)


from sklearn.metrics import classification_report

print(classification_report(y_cv, y_log_pred_test))
# print the first 10 predicted responses

# 1D array (vector) of binary values (0, 1)

logreg.predict(x_cv)[0:10]
# print the first 10 predicted probabilities of class membership

logreg.predict_proba(x_cv)[0:10]
# print the first 10 predicted probabilities for class 1   ( predicting Loan_default =1)

logreg.predict_proba(x_cv)[0:10, 1]
# store the predicted probabilities for class 1

y_pred_prob = logreg.predict_proba(x_cv)[:, 1]
y_pred_prob[0:10]
# Plotting predicion through histogram of predicted probabilities

%matplotlib inline

import matplotlib.pyplot as plt



# 8 bins

plt.hist(y_pred_prob, bins=8)



# x-axis limit from 0 to 1

plt.xlim(0,1)

plt.title('Histogram of predicted probabilities')

plt.xlabel('Predicted probability of default')

plt.ylabel('Frequency')
# IMPORTANT: first argument is true values, second argument is predicted probabilities



# we pass y_cv and y_pred_prob

# we do not use y_pred, because it will give incorrect results without generating an error

# roc_curve returns 3 objects fpr, tpr, thresholds

# fpr: false positive rate

# tpr: true positive rate

fpr, tpr, thresholds = metrics.roc_curve(y_cv, y_pred_prob)



plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for default classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
# IMPORTANT: first argument is true values, second argument is predicted probabilities



print(metrics.roc_auc_score(y_cv, y_pred_prob))
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3,random_state =4)
# Handing Class im balance

from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(x_train, y_train)
from sklearn.model_selection import StratifiedKFold

cv =StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(max_depth = 10, n_estimators=300,verbose=1, n_jobs=1,random_state=42)
forest=rf.fit(X_smt,y_smt)
print(forest.score(X_smt,y_smt))
pre = forest.predict(X_smt)
pre_ = forest.predict_proba(X_smt)
test.head()
pred = forest.predict(test)
pred_=forest.predict_proba(test)
df_confusion_rf = metrics.confusion_matrix(y_smt, pre)

df_confusion_rf
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)

sns.heatmap(df_confusion_rf, cmap = cmap,xticklabels=['Prediction No','Prediction Yes'],yticklabels=['Actual No','Actual Yes'], annot=True,

            fmt='d')
y_smt.shape, pre.shape
# print the first 15 true and predicted responses

print('True', y_smt[0:15])

print('Pred', pre[0:15])
#comparing the metrics of predicted lebel and real label of test data

print('Accuracy_Score:', metrics.accuracy_score(y_smt, pre))
# Method to calculate Sensitivity



print('Sensitivity or Recall:', metrics.recall_score(y_smt, pre))
from sklearn.metrics import classification_report

print(classification_report(y_smt, pre))


from sklearn import metrics 

fpr, tpr, thresholds = metrics.roc_curve(y_smt, pre) 

auc = metrics.roc_auc_score(y_smt, pre) 

plt.figure(figsize=(12,8)) 

plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 

plt.xlabel('False Positive Rate')  

plt.ylabel('True Positive Rate') 

plt.legend(loc=4) 

plt.show()
from sklearn.model_selection import cross_val_predict

pre_rf = cross_val_predict(rf, cv=cv, X=X_smt,y=y_smt, verbose=1)
from sklearn.metrics import roc_auc_score

print("auc score =\t" ,roc_auc_score(y_smt, pre_rf))
pre_rf


rf.fit(X_smt,y_smt)
pred_out = rf.predict(test)
pred_out
prob_out=rf.predict_proba(test)[:,1]
prob_out[0:10], pred_out.shape, prob_out.shape
submission = pd.read_csv("../input/Submission_LTFS.csv")
submission['loan_default']=pred_out            # filling Loan_Status with predictions

submission['UniqueID']=test['UniqueID'] # filling Unique_ID with test Unique_ID
pd.DataFrame(submission, columns=['UniqueID','loan_default']).to_csv('submission_rf.csv')
from sklearn import preprocessing 

for f in train.columns: 

    if train[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder() 

        lbl.fit(list(train[f].values)) 

        train[f] = lbl.transform(list(train[f].values))



for f in test.columns: 

    if test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder() 

        lbl.fit(list(test[f].values)) 

        test[f] = lbl.transform(list(test[f].values))



train.fillna((-999), inplace=True) 

test.fillna((-999), inplace=True)



train=np.array(train) 

test=np.array(test) 

train = train.astype(float) 

test = test.astype(float)



 
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3,random_state =4)
import xgboost as xgb

from xgboost import XGBClassifier

xgb= XGBClassifier(n_estimators=120, learning_rate=1, n_jobs=-1,random_state=42)

predict = cross_val_predict(xgb, cv=cv, X=X_smt,y=y_smt, verbose=1,method='predict_proba')
boost=xgb.fit(X_smt,y_smt)

pred_xgb = boost.predict(X_smt)

pred_xg=boost.predict(test)


print("auc score =\t" ,roc_auc_score(y_smt, pred_xgb))
pred_xgb, pred_xgb.shape
from sklearn.model_selection import cross_val_score

xgb= XGBClassifier(n_estimators=120, learning_rate=1, n_jobs=-1,random_state=42)

scores = cross_val_score(xgb, cv=cv, X=X_smt,y=y_smt, verbose=1,scoring='roc_auc')

print("auc\t=\t",scores.mean())


submission["loan_default"] = pred_xg

submission.to_csv("submission_xgb.csv", index=False)

submission.head()
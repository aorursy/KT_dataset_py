# import libraries

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.impute import SimpleImputer

plt.style.use('fivethirtyeight')

import os
df = pd.read_csv('/kaggle/input/should-this-loan-be-approved-or-denied/SBAnational.csv',low_memory=False)

df.head()
df.isnull().sum()
df = df.drop(columns = 'ChgOffDate')

df = df.dropna()
df.dtypes
# makes column for Paid in Full (1=Paid in full, 0 = no)

df['Paid'] = df['MIS_Status'].replace({'P I F':'1','CHGOFF':'0'}, regex=True)

df['Paid'] = df['Paid'].astype(float)





#fixes object values in some of our columns

def fix_num(number):

    num = number.replace("$", "")

    num = num.replace(",","")

    num = num.replace(" ","")

    return float(num)



df['BalanceGross'] = df['BalanceGross'].apply(lambda x: fix_num(x))

df['DisbursementGross'] = df['DisbursementGross'].apply(lambda x: fix_num(x))

df['ChgOffPrinGr'] = df['ChgOffPrinGr'].apply(lambda x: fix_num(x))

df['GrAppv'] = df['GrAppv'].apply(lambda x: fix_num(x))

df['SBA_Appv'] = df['SBA_Appv'].apply(lambda x: fix_num(x))
# changes fiscal year of commitment from object to int

df['ApprovalFY'] = df['ApprovalFY'].replace({'A':'','B':''}, regex = True).astype(int)



# changes new vs existing business from 1 and 2 to 1(new) and 0(existing) for interpretability

df['NewExist'] = df['NewExist'].replace(1,0)

df['NewExist'] = df['NewExist'].replace(2,1)



# changes RevLineCR to binary variable

df['RevLineCr'] = df['RevLineCr'].replace({'Y':'1','N':'0'}, regex=True)

valid = ['1', '0']

df = df.loc[df['RevLineCr'].isin(valid)]

df['RevLineCr'] = df['RevLineCr'].astype(int)



# changes LowDoc to binary variable

df['LowDoc'] = df['LowDoc'].replace({'Y':'1', 'N':'0'}, regex=True)

valid1 = ['1', '0']

df = df.loc[df['LowDoc'].isin(valid)]

df['LowDoc'] = df['LowDoc'].astype(int)



# makes franchise a binary variables

df['FranchiseCode'] = df['FranchiseCode'].replace(1,0)

df['FranchiseCode'] = np.where((df.FranchiseCode != 0),1,df.FranchiseCode)

df.rename(columns={"FranchiseCode":"Franchise"},inplace=True)
# Real Estate

df['RealEstate'] = df['Term'] > 240 

df['RealEstate'] = df['RealEstate'].astype(str)

df['RealEstate'] = df['RealEstate'].replace({'False':'0','True':'1'},regex=True).astype(int)



# Recession

rec_years = [1969,1970,1973,1974,1975,1980,1981,1982,1990,1991,2001,2007,2008,2009]

df['Recession'] = df['ApprovalFY'].isin(rec_years)

df['Recession'] = df['Recession'].astype(str)

df['Recession'] = df['Recession'].replace({'False':'0','True':'1'},regex=True).astype(int)
# makes dataframe with only numeric variables

numeric = ['Paid','ApprovalFY', 'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',

           'UrbanRural', 'RevLineCr', 'LowDoc', 'DisbursementGross', 'BalanceGross', 

            'GrAppv', 'SBA_Appv', 'RealEstate', 'Franchise','Recession']

num_df = df[numeric]

num_df.describe().T
corr = num_df.corr()

fig, ax = plt.subplots(figsize=(15,15))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot = True, ax=ax, mask=mask, cmap = "Blues").set(title='Feature Correlations')

plt.show()
fig = plt.figure(figsize=(12,7))

sns.distplot(a=df['Term'], bins = 40,kde=False)

plt.title('Distribution of Loan Terms')

plt.ylabel('Count')

plt.show()
fig = plt.figure(figsize=(12,7))

sns.distplot(a=df['DisbursementGross'], kde=False, bins=40)

plt.title('Distibution of Gross Disbursement')

plt.ylabel('Count')

plt.show()
fig = plt.figure(figsize=(10,10))

sns.lineplot(x="ApprovalFY", y="Paid", data=df)

plt.title('SBA Loan PIF(Paid in Full) Rate Over Time')

plt.show()
fig = plt.figure(figsize=(10,10))

sns.distplot(a=df['ApprovalFY'],bins=40, kde=False)

plt.title('Amount of SBA Loans Taken Over Time')

plt.ylabel('Count')

plt.xlabel('Year')

plt.show()
post_1985 = df[df['ApprovalFY'] >= 1986]

fig = plt.figure(figsize=(15,10))

sns.countplot(x="ApprovalFY",data=post_1985,hue="Paid")

plt.title('Amount of Defulted vs Paid SBA Loans over Time')

plt.xticks(rotation=75)

plt.show()
# map the NAICS codes to their actual industries

df['NAICS'] = df['NAICS'].astype(str)

def first_two(string):

    s = string[:2]

    return s

df['NAICS'] = df['NAICS'].apply(lambda x: first_two(x))



dic = {'11': 'Agriculture, Fishing, Forestry, and Hunting',

      '21': 'Mining, Quarrying, Oil and Gas Extraction',

      '22': 'Utilities', '23': 'Construction', '31':'Manufacturing',

      '32': 'Manufacturing', '33': 'Manufacturing', '42': 'Wholesale Trade',

      '44': 'Retail Trade', '45':'Retail Trade', '48':'Transport and Warehouse',

      '49': 'Transport and Warehouse', '51':'Information',

      '52': 'Finance and Insurance', '53':'Real Estate and Rental Leasing',

      '54':'Profesisonal, Scientific, and Technical Services',

      '55':'Management', '56':'Administrative and Support and Waste Management',

      '61':'Educational Services', '62':'Health Care and Social Assistance',

      '71': 'Arts, Entertainment and Recreation', '72':'Accomodation and Food Services',

      '81': 'Other Services', '92':'Public Administration'}



df['NAICS'] = df['NAICS'].map(dic)
fig = plt.figure(figsize=(15,10))

sns.barplot(x=df['NAICS'],y = df['Paid'],palette = "deep")

plt.xticks(rotation=80)

plt.title('Paid In Full Rates for Each Industry')

plt.xlabel('Industry')

plt.ylabel('Loan PIF Rate')

plt.show()
fig = plt.figure(figsize=(15,10))

sns.barplot(x=df['NAICS'],y = df['Term'],palette = "deep")

plt.xticks(rotation=80)

plt.title('Loan Terms for Each Industry')

plt.xlabel('Industry')

plt.ylabel('Loan Term (Months)')

plt.show()
fig = plt.figure(figsize=(15,10))

sns.barplot(x=df['NAICS'],y = df['DisbursementGross'],palette = "deep")

plt.xticks(rotation=80)

plt.title('Loan Size in Each Industry')

plt.xlabel('Industry')

plt.ylabel('Gross Disbursement(USD)')

plt.show()
state_default = df.groupby(['State','Paid'])['State'].count().unstack('Paid')

state_default['Total SBA Loans Taken'] = state_default[1] + state_default[0]

state_default['PIF Percentage'] = state_default[1]/(state_default[1] + state_default[0])

state_default['Default Percentage'] = (1 - state_default['PIF Percentage'])

state_default = state_default.sort_values(by = 'Default Percentage')

state_default
fig = plt.figure(figsize=(7,7))

sns.barplot(x="RealEstate", y="Paid", data=df)

plt.title('PIF Rate for Loans backed by Real Estate')

plt.xlabel('Real Estate')

plt.show()
fig = plt.figure(figsize=(12,7))

sns.lineplot(x="ApprovalFY", y ="RealEstate", data=df)

plt.xlabel('Year')

plt.title('Real Estate Use over Time')



fig = plt.figure(figsize=(12,7))

sns.barplot(x="NAICS",y="RealEstate",data=df)

plt.title('Real Estate Use by Industry')

plt.xticks(rotation = 80)



plt.show()
df = df.drop(['LoanNr_ChkDgt','Bank','GrAppv','DisbursementGross','Name', 'City', 'MIS_Status', 'ApprovalDate', 

              'Zip','BankState', 'DisbursementDate','ChgOffPrinGr','BalanceGross'], axis = 1)



# take log log to fix skew

df['SBA_Appv'] = np.log(df['SBA_Appv'])



# mapping state and industry default rates



state_def = {'MT':.068, 'WY':.069,'VT':.073,'ND':.076,'SD':.078,'ME':.096,

            'NH':.105,'NM':.107,'NE':.112,'AK':.114,'IA':.115,'MN':.116,

            'RI':.118,'WI':.121,'MA':.127,'KS':.129,'WA':.133,'CT':.136,

            'ID':.141,'PA':.145,'OR':.149,'MO':.151,'HI':.153,'OK':.154,

            'MS':.157,'WV':.162,'OH':.163,'AL':.165,'AR':.167,'IN':.175,

            'UT':.175,'DE':.175,'CA':.177,'CO':.178,'VA':.180,'LA':.181,

            'NC':.184,'TX':.186,'MD':.191,'KY':.192,'SC':.192,'NY':.195,

            'NJ':.195,'AZ':.203,'TN':.206,'MI':.225,'NV':.225,'IL':.225,

            'GA':.227,'DC':.235,'FL':.257}



df['State'] = df['State'].map(state_def)



ind_def = {'Accomodation and Food Services':.217,'Administrative and Support and Waste Management':.225,

          'Agriculture, Fishing, Forestry, and Hunting':.089,'Arts, Entertainment and Recreatiom':.202,

          'Construction':.227,'Educational Services':.236,'Finance and Insurance':.276,

          'Health Care and Social Assistance':.101,'Information':.242,'Management':.098,

          'Manufacturing':.149,'Mining, Quarrying, Oil and Gas Extraction':.083,

          'Other Services':.191,'Profesisonal, Scientific, and Technical Services':.184,

          'Public Administration':.155,'Real Estate and Rental Leasing':.279,

          'Retail Trade':.222,'Transport and Warehouse':.258,'Utilities':.137,

          'Wholesale Trade':.187}



df['NAICS'] = df['NAICS'].map(ind_def)

df = df.rename(columns={'State':'State Default Rate','NAICS':'Industry Default Rate'})
# splitting up our data and getting everything ready

y = df['Paid']

x = df.drop(['Paid'], axis = 1)

train_X, test_X, train_y, test_y = train_test_split(x, y, random_state = 0)

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)
forest = RandomForestClassifier(n_estimators=100)

forest.fit(train_X, train_y)

forest_pred = forest.predict(test_X)

score = (accuracy_score(test_y,forest_pred) * 100)

print('Random Forest Accuracy: %r' % round(score,2), '%')
gboost = GradientBoostingClassifier()

gboost.fit(train_X, train_y)

gboost_pred = gboost.predict(test_X)

score = (accuracy_score(test_y,gboost_pred) * 100)

print('Gradient Boost Accuracy: %r' % round(score,2), '%')
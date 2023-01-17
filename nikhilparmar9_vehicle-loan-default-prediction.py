import os

import re

import gc 

from tqdm import tqdm

from datetime import date     #calculating age

from datetime import datetime #converting string to date

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV , train_test_split

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score , f1_score , make_scorer

from sklearn.preprocessing import StandardScaler,OneHotEncoder , LabelEncoder ,normalize

from sklearn.feature_selection import SelectKBest,f_classif,chi2

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from matplotlib.pyplot import figure

print("DONE")
dd1=pd.read_csv("../input/lt-vehicle-loan-default-prediction/train.csv")
print("Total Size of the Provided Data: ",dd1.shape)
print("Provided Features: ",dd1.shape[1])
#Lets looks at data description

info = pd.read_csv("../input/lt-vehicle-loan-default-prediction/data_dictionary.csv")

info
# CHANGING TO UPPER CASE

dd1.columns=dd1.columns.str.upper()

# REPLACING "." with "_" in column names

dd1.columns=dd1.columns.str.replace(".","_")
dd1.head()
dd1.head()
dd1.columns
dd1['LOAN_DEFAULT'].value_counts()
#Graph

my_pal = {0: 'lightblue', 1: 'red'}



plt.figure(figsize = (12, 6))

ax = sns.countplot(x = 'LOAN_DEFAULT', data = dd1, palette = my_pal)

plt.title('Class Distribution')

plt.show()
dd1.describe()
dd1.info()
dd1.isnull().sum()
dd1['EMPLOYMENT_TYPE'].unique()
## TOTAL NULL VALUES

dd1['EMPLOYMENT_TYPE'].isna().sum()
dd1['EMPLOYMENT_TYPE'].value_counts()
dd1['EMPLOYMENT_TYPE'] = dd1['EMPLOYMENT_TYPE'].fillna(method = 'bfill')
dd1['EMPLOYMENT_TYPE'].unique()
dd1['EMPLOYMENT_TYPE'].isna().sum()
dd1['EMPLOYMENT_TYPE'].value_counts()
dd1.nunique()
dd1['PERFORM_CNS_SCORE_DESCRIPTION'].nunique()
dd1['PERFORM_CNS_SCORE_DESCRIPTION'].unique()
dd1['PERFORM_CNS_SCORE_DESCRIPTION'].value_counts()
dd1 = dd1.replace({'PERFORM_CNS_SCORE_DESCRIPTION':{'C-Very Low Risk':'Low', 'A-Very Low Risk':'Low',

                                                       'B-Very Low Risk':'Low', 'D-Very Low Risk':'Low',

                                                       'F-Low Risk':'Low', 'E-Low Risk':'Low', 'G-Low Risk':'Low',

                                                       'H-Medium Risk': 'Medium', 'I-Medium Risk': 'Medium',

                                                       'J-High Risk':'High', 'K-High Risk':'High','L-Very High Risk':'Very_High',

                                                       'M-Very High Risk':'Very_High','Not Scored: More than 50 active Accounts found':'Not_Scored',

                                                       'Not Scored: Only a Guarantor':'Not_Scored','Not Scored: Not Enough Info available on the customer':'Not_Scored',

                                                        'Not Scored: No Activity seen on the customer (Inactive)':'Not_Scored','Not Scored: No Updates available in last 36 months':'Not_Scored',

                                                       'Not Scored: Sufficient History Not Available':'Not_Scored', 'No Bureau History Available':'Not_Scored'

                                                       }})
dd1["PERFORM_CNS_SCORE_DESCRIPTION"].unique()
dd1['PERFORM_CNS_SCORE_DESCRIPTION'].value_counts()
for i in dd1.columns:

    print('Distinct_values for the column:',i)

    print('No.of unique items:',dd1[i].nunique())

    print(dd1[i].unique())

    print('-'*30)

    print('')
dd1.info()
# Before Convertion

dd1['AVERAGE_ACCT_AGE']
dd1[['AVERAGE_ACCT_Yr','AVERAGE_ACCT_Month']] = dd1['AVERAGE_ACCT_AGE'].str.split("yrs",expand=True)

dd1[['AVERAGE_ACCT_Month','AVERAGE_ACCT_Month1']] = dd1['AVERAGE_ACCT_Month'].str.split("mon",expand=True)

dd1["AVERAGE_ACCT_AGE"]= dd1["AVERAGE_ACCT_Yr"].astype(str).astype(int)+((dd1["AVERAGE_ACCT_Month"].astype(str).astype(int))/12)

dd1= dd1.drop(columns= ["AVERAGE_ACCT_Yr","AVERAGE_ACCT_Month",'AVERAGE_ACCT_Month1'])
dd1[['CREDIT_HISTORY_LENGTH_Yr','CREDIT_HISTORY_LENGTH_Month']] = dd1['CREDIT_HISTORY_LENGTH'].str.split("yrs",expand=True)

dd1[['CREDIT_HISTORY_LENGTH_Month','CREDIT_HISTORY_LENGTH_Month1']] = dd1['CREDIT_HISTORY_LENGTH_Month'].str.split("mon",expand=True)

dd1["CREDIT_HISTORY_LENGTH"]= dd1["CREDIT_HISTORY_LENGTH_Yr"].astype(str).astype(int)+((dd1["CREDIT_HISTORY_LENGTH_Month"].astype(str).astype(int))/12)

dd1= dd1.drop(columns= ["CREDIT_HISTORY_LENGTH_Yr","CREDIT_HISTORY_LENGTH_Month",'CREDIT_HISTORY_LENGTH_Month1'])

## Aftering Converting

dd1['AVERAGE_ACCT_AGE']
now = pd.Timestamp('now')

dd1['DATE_OF_BIRTH'] = pd.to_datetime(dd1['DATE_OF_BIRTH'], format='%d-%m-%y')

dd1['DATE_OF_BIRTH'] = dd1['DATE_OF_BIRTH'].where(dd1['DATE_OF_BIRTH'] < now, dd1['DATE_OF_BIRTH'] -  np.timedelta64(100, 'Y'))

dd1['AGE'] = (now - dd1['DATE_OF_BIRTH']).astype('<m8[Y]')
now = pd.Timestamp('now')

dd1['DISBURSALDATE'] = pd.to_datetime(dd1['DISBURSALDATE'], format='%d-%m-%y')

dd1['DISBURSALDATE'] = dd1['DISBURSALDATE'].where(dd1['DISBURSALDATE'] < now, dd1['DISBURSALDATE'] -  np.timedelta64(100, 'Y'))

dd1['LOAN_AGE'] = (now - dd1['DISBURSALDATE']).astype('<m8[Y]')
sns.distplot(dd1['AGE'], color = 'blue')

plt.title('Distribution of Age')
dd1['DATE_OF_BIRTH'].dtypes
dd1.info()
df=dd1.copy()
y=df['LOAN_DEFAULT']

X=df.drop("LOAN_DEFAULT",axis=1)
# from sklearn.model_selection import train_test_split,KFold,cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print("Size of X",X.shape)

print("Size of y",y.shape)

print("Size of X_train",X_train.shape)

print("Size of y_train",y_train.shape)
y_train_plot=pd.DataFrame(y_train,columns=['LOAN_DEFAULT'])

y_test_plot=pd.DataFrame(y_test,columns=['LOAN_DEFAULT'])



defaulters_train=y_train_plot['LOAN_DEFAULT'].sum()

non_defaulters_train=len(y_train_plot)-y_train_plot['LOAN_DEFAULT'].sum()

total_train=len(y_train_plot)



defaulters_test=y_test_plot['LOAN_DEFAULT'].sum()

non_defaulters_test=len(y_test_plot)-y_test_plot['LOAN_DEFAULT'].sum()

total_test=len(y_test_plot)
print("\n")

print("X_TRAIN INFO: Total:",total_train)

print("DEFAULTERS:",defaulters_train,"->",round(defaulters_train/total_train,2),"Percent")

print("Non-DEFAULTERS:",non_defaulters_train,"->",round(non_defaulters_train/total_train,2),"%")

print("\n")

print("X_TEST INFO: Total:",total_test)

print("DEFAULTERS:",defaulters_test,"->",round(defaulters_test/total_test,2),"Percent")

print("Non-DEFAULTERS:",non_defaulters_test,"->",round(non_defaulters_test/total_test,2),"%")
#Graph

my_pal = {0: 'lightblue', 1: 'red'}



plt.figure(figsize = (6, 3))

ax = sns.countplot(x = 'LOAN_DEFAULT', data = y_train_plot, palette = my_pal)

plt.title('X_Train Class Distribution')

plt.show()
#Graph

my_pal = {0: 'lightblue', 1: 'red'}



plt.figure(figsize = (6, 3))

ax = sns.countplot(x = 'LOAN_DEFAULT', data = y_test_plot, palette = my_pal)

plt.title('X_Test Class Distribution')

plt.show()
columnsToDelete = ['UNIQUEID','MOBILENO_AVL_FLAG','CURRENT_PINCODE_ID',

                   'EMPLOYEE_CODE_ID','STATE_ID','BRANCH_ID','MANUFACTURER_ID',

                   'SUPPLIER_ID','DATE_OF_BIRTH','DISBURSALDATE','NO_OF_INQUIRIES']

## BEFORE DELETING THE COLUMNS

print("Size AFTER Deleting the Features",len(X_train.columns))



## DROPING THE COLUMNS FROM THE DATA FRAME

X_train=X_train.drop(X_train[columnsToDelete],axis=1)



## AFTER DROPPING THE COLUMNS

print("Size AFTER Deleting the Features",len(X_train.columns))
X_train.columns
numericalTypes=['DISBURSED_AMOUNT', 'ASSET_COST', 'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS', 'LTV',

           'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT', 

           'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS', 'SEC_OVERDUE_ACCTS', 

           'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT', 'SEC_DISBURSED_AMOUNT', 

           'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT', 'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'PERFORM_CNS_SCORE',

           'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'LOAN_AGE','AGE','AVERAGE_ACCT_AGE','CREDIT_HISTORY_LENGTH']



categoricalTypes=[ 'AADHAR_FLAG', 'PAN_FLAG', 'VOTERID_FLAG',

             'DRIVING_FLAG', 'PASSPORT_FLAG','EMPLOYMENT_TYPE','PERFORM_CNS_SCORE_DESCRIPTION']
len(numericalTypes)
len(categoricalTypes)
## Creating New dataframe for Numerical and Categorical 

X_train_numerical=X_train[numericalTypes].copy()

X_test_numerical=X_test[numericalTypes].copy()
n = SelectKBest(score_func=f_classif,k=16)

# numcols=n.fit(dd11[numericalTypes],dd11['LOAN_DEFAULT'])

numcols=n.fit(X_train_numerical,y_train)

plt.figure(figsize=(7,7))

sns.barplot(x=numcols.scores_,y=numericalTypes)

plt.title('Best Numerical Features')

plt.show()
## Creating dictionaries to store the feature names and its importance value

topNumFeatures={}



## https://machinelearningmastery.com/feature-selection-with-categorical-data/ 

for i in range(len(n.scores_)):

    topNumFeatures[numericalTypes[i]]=n.scores_[i]



# SORT THE DICTIONARY AS PER THE IMPORTANT SCORES

topNumFeatures = sorted(topNumFeatures.items(), key=lambda x: x[1],reverse=True) 

print("-----------------------TOP FEATURES SORTED AS PER THE HIGH IMPORTANCE----------------")

topNumFeatures
X_train_numerical.loc[:,'No_of_Accounts'] = X_train_numerical['PRI_NO_OF_ACCTS'] + X_train_numerical['SEC_NO_OF_ACCTS']

X_train_numerical.loc[:,'PRI_Inactive_accounts'] = X_train_numerical['PRI_NO_OF_ACCTS'] - X_train_numerical['PRI_ACTIVE_ACCTS']

X_train_numerical.loc[:,'SEC_Inactive_accounts'] = X_train_numerical['SEC_NO_OF_ACCTS'] - X_train_numerical['SEC_ACTIVE_ACCTS']

X_train_numerical.loc[:,'Total_Inactive_accounts'] = X_train_numerical['PRI_Inactive_accounts'] + X_train_numerical['SEC_Inactive_accounts']

X_train_numerical.loc[:,'Total_Overdue_Accounts'] = X_train_numerical['PRI_OVERDUE_ACCTS'] + X_train_numerical['SEC_OVERDUE_ACCTS']

X_train_numerical.loc[:,'Total_Current_Balance'] = X_train_numerical['PRI_CURRENT_BALANCE'] + X_train_numerical['SEC_CURRENT_BALANCE']

X_train_numerical.loc[:,'Total_Sanctioned_Amount'] = X_train_numerical['PRI_SANCTIONED_AMOUNT'] + X_train_numerical['SEC_SANCTIONED_AMOUNT']

X_train_numerical.loc[:,'Total_Disbursed_Amount'] = X_train_numerical['PRI_DISBURSED_AMOUNT'] + X_train_numerical['SEC_DISBURSED_AMOUNT']

X_train_numerical.loc[:,'Total_Installment'] = X_train_numerical['PRIMARY_INSTAL_AMT'] + X_train_numerical['SEC_INSTAL_AMT']







X_test_numerical.loc[:,'No_of_Accounts'] = X_test_numerical['PRI_NO_OF_ACCTS'] + X_test_numerical['SEC_NO_OF_ACCTS']

X_test_numerical.loc[:,'PRI_Inactive_accounts'] = X_test_numerical['PRI_NO_OF_ACCTS'] - X_test_numerical['PRI_ACTIVE_ACCTS']

X_test_numerical.loc[:,'SEC_Inactive_accounts'] = X_test_numerical['SEC_NO_OF_ACCTS'] - X_test_numerical['SEC_ACTIVE_ACCTS']

X_test_numerical.loc[:,'Total_Inactive_accounts'] = X_test_numerical['PRI_Inactive_accounts'] + X_test_numerical['SEC_Inactive_accounts']

X_test_numerical.loc[:,'Total_Overdue_Accounts'] = X_test_numerical['PRI_OVERDUE_ACCTS'] + X_test_numerical['SEC_OVERDUE_ACCTS']

X_test_numerical.loc[:,'Total_Current_Balance'] = X_test_numerical['PRI_CURRENT_BALANCE'] + X_test_numerical['SEC_CURRENT_BALANCE']

X_test_numerical.loc[:,'Total_Sanctioned_Amount'] = X_test_numerical['PRI_SANCTIONED_AMOUNT'] + X_test_numerical['SEC_SANCTIONED_AMOUNT']

X_test_numerical.loc[:,'Total_Disbursed_Amount'] = X_test_numerical['PRI_DISBURSED_AMOUNT'] + X_test_numerical['SEC_DISBURSED_AMOUNT']

X_test_numerical.loc[:,'Total_Installment'] = X_test_numerical['PRIMARY_INSTAL_AMT'] + X_test_numerical['SEC_INSTAL_AMT']
X_test_numerical.columns
X_train_numerical.columns
print("Shape of X_train_numerical: ",X_train_numerical.shape)

print("Shape of X_test_numerical: ",X_test_numerical.shape)
X_train_numerical=X_train_numerical.drop(['PRI_NO_OF_ACCTS','SEC_NO_OF_ACCTS',

			'PRI_ACTIVE_ACCTS','SEC_ACTIVE_ACCTS',

			'PRI_CURRENT_BALANCE','SEC_CURRENT_BALANCE',

			'PRI_Inactive_accounts','SEC_Inactive_accounts',

            'PRI_SANCTIONED_AMOUNT','SEC_SANCTIONED_AMOUNT',

            'PRI_DISBURSED_AMOUNT','SEC_DISBURSED_AMOUNT',

            'PRI_OVERDUE_ACCTS','SEC_OVERDUE_ACCTS',

            'PRIMARY_INSTAL_AMT','SEC_INSTAL_AMT'],axis=1)



X_test_numerical=X_test_numerical.drop(['PRI_NO_OF_ACCTS','SEC_NO_OF_ACCTS',

			'PRI_ACTIVE_ACCTS','SEC_ACTIVE_ACCTS',

			'PRI_CURRENT_BALANCE','SEC_CURRENT_BALANCE',

			'PRI_Inactive_accounts','SEC_Inactive_accounts',

            'PRI_SANCTIONED_AMOUNT','SEC_SANCTIONED_AMOUNT',

            'PRI_DISBURSED_AMOUNT','SEC_DISBURSED_AMOUNT',

            'PRI_OVERDUE_ACCTS','SEC_OVERDUE_ACCTS',

            'PRIMARY_INSTAL_AMT','SEC_INSTAL_AMT'],axis=1)     
X_test_numerical.columns
X_train_numerical.columns
print("After Droping: Shape of X_train_numerical: ",X_train_numerical.shape)

print("After Droping: Shape of X_test_numerical: ",X_test_numerical.shape)
numericalTypesUpdated=['DISBURSED_AMOUNT', 'ASSET_COST', 'LTV', 'NEW_ACCTS_IN_LAST_SIX_MONTHS', 

           'PERFORM_CNS_SCORE','DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'LOAN_AGE','AGE',

           'AVERAGE_ACCT_AGE','CREDIT_HISTORY_LENGTH',

           'No_of_Accounts', 'Total_Inactive_accounts','Total_Overdue_Accounts', 

           'Total_Current_Balance','Total_Sanctioned_Amount', 'Total_Disbursed_Amount','Total_Installment']
## Before Standardization

X_train.head()
scaler = StandardScaler()

scaler.fit(X_train_numerical)

X_train_numerical_std = scaler.transform(X_train_numerical)

X_test_numerical_std = scaler.transform(X_test_numerical)
## Type of Returned Data

type(X_train_numerical_std)
X_train_numerical_std
X_train_numerical_std=pd.DataFrame(X_train_numerical_std,columns=['DISBURSED_AMOUNT', 'ASSET_COST', 'LTV', 'NEW_ACCTS_IN_LAST_SIX_MONTHS',

       'PERFORM_CNS_SCORE', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'LOAN_AGE',

       'AGE', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'No_of_Accounts',

       'Total_Inactive_accounts', 'Total_Overdue_Accounts',

       'Total_Current_Balance', 'Total_Sanctioned_Amount',

       'Total_Disbursed_Amount', 'Total_Installment'])



X_test_numerical_std=pd.DataFrame(X_test_numerical_std,columns=['DISBURSED_AMOUNT', 'ASSET_COST', 'LTV', 'NEW_ACCTS_IN_LAST_SIX_MONTHS',

       'PERFORM_CNS_SCORE', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'LOAN_AGE',

       'AGE', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'No_of_Accounts',

       'Total_Inactive_accounts', 'Total_Overdue_Accounts',

       'Total_Current_Balance', 'Total_Sanctioned_Amount',

       'Total_Disbursed_Amount', 'Total_Installment'])
# Checking the values after converting

X_train_numerical_std
print("Shape of Standardized X_train: ",X_train_numerical_std.shape)

print("Shape of Standardized X_test: ",X_test_numerical_std.shape)
nn = SelectKBest(score_func=f_classif,k='all')

numcols=nn.fit(X_train_numerical_std,y_train)

plt.figure(figsize=(7,7))

sns.barplot(x=numcols.scores_,y=X_train_numerical_std.columns)

plt.title('Best Numerical Features')

plt.show()
## Creating dictionaries to store the feature names and its importance value

topNumFeatures={}



## https://machinelearningmastery.com/feature-selection-with-categorical-data/ 

for i in range(len(nn.scores_)):

#     print('Feature %s: %f' % (numerical[i], n.scores_[i]))

    topNumFeatures[numericalTypesUpdated[i]]=nn.scores_[i]



# SORT THE DICTIONARY AS PER THE IMPORTANT SCORES

topNumFeatures = sorted(topNumFeatures.items(), key=lambda x: x[1],reverse=True) 

print("-----------------------TOP FEATURES SORTED AS PER THE HIGH IMPORTANCE----------------")

topNumFeatures
X_train_categorical=X_train[categoricalTypes].copy()

X_test_categorical=X_test[categoricalTypes].copy()
X_train_categorical.columns
onehot_encoder = OneHotEncoder(sparse=False)

X_train_categorical_encoded = onehot_encoder.fit(X_train_categorical)

X_train_categorical_encoded = onehot_encoder.transform(X_train_categorical) ## NOT EXECUTED

X_test_categorical_encoded = onehot_encoder.transform(X_test_categorical)
# Checking the Encoded Data

X_train_categorical_encoded
print("Shape of X_train after One Hot Encoding: ",X_train_categorical_encoded.shape)
type(X_train_categorical_encoded)
## Obtaining Feature Names from the Classifier

onehot_encoder.get_feature_names(['AADHAR_FLAG', 'PAN_FLAG', 'VOTERID_FLAG', 'DRIVING_FLAG',

       'PASSPORT_FLAG', 'EMPLOYMENT_TYPE', 'PERFORM_CNS_SCORE_DESCRIPTION'])
## Adding the Obtained feature names into LIST

encodedCatColumnNames=['AADHAR_FLAG_0', 'AADHAR_FLAG_1', 'PAN_FLAG_0', 'PAN_FLAG_1',

       'VOTERID_FLAG_0', 'VOTERID_FLAG_1', 'DRIVING_FLAG_0',

       'DRIVING_FLAG_1', 'PASSPORT_FLAG_0', 'PASSPORT_FLAG_1',

       'EMPLOYMENT_TYPE_Salaried', 'EMPLOYMENT_TYPE_Self employed',

       'PERFORM_CNS_SCORE_DESCRIPTION_High',

       'PERFORM_CNS_SCORE_DESCRIPTION_Low',

       'PERFORM_CNS_SCORE_DESCRIPTION_Medium',

       'PERFORM_CNS_SCORE_DESCRIPTION_Not_Scored',

       'PERFORM_CNS_SCORE_DESCRIPTION_Very_High']
X_train_categorical_encoded=pd.DataFrame(X_train_categorical_encoded,columns=['AADHAR_FLAG_0', 'AADHAR_FLAG_1', 'PAN_FLAG_0', 'PAN_FLAG_1',

       'VOTERID_FLAG_0', 'VOTERID_FLAG_1', 'DRIVING_FLAG_0',

       'DRIVING_FLAG_1', 'PASSPORT_FLAG_0', 'PASSPORT_FLAG_1',

       'EMPLOYMENT_TYPE_Salaried', 'EMPLOYMENT_TYPE_Self employed',

       'PERFORM_CNS_SCORE_DESCRIPTION_High',

       'PERFORM_CNS_SCORE_DESCRIPTION_Low',

       'PERFORM_CNS_SCORE_DESCRIPTION_Medium',

       'PERFORM_CNS_SCORE_DESCRIPTION_Not_Scored',

       'PERFORM_CNS_SCORE_DESCRIPTION_Very_High'])



X_test_categorical_encoded=pd.DataFrame(X_test_categorical_encoded,columns=['AADHAR_FLAG_0', 'AADHAR_FLAG_1', 'PAN_FLAG_0', 'PAN_FLAG_1',

       'VOTERID_FLAG_0', 'VOTERID_FLAG_1', 'DRIVING_FLAG_0',

       'DRIVING_FLAG_1', 'PASSPORT_FLAG_0', 'PASSPORT_FLAG_1',

       'EMPLOYMENT_TYPE_Salaried', 'EMPLOYMENT_TYPE_Self employed',

       'PERFORM_CNS_SCORE_DESCRIPTION_High',

       'PERFORM_CNS_SCORE_DESCRIPTION_Low',

       'PERFORM_CNS_SCORE_DESCRIPTION_Medium',

       'PERFORM_CNS_SCORE_DESCRIPTION_Not_Scored',

       'PERFORM_CNS_SCORE_DESCRIPTION_Very_High'])
print("Shape of Encoded X_train Categorical: ",X_train_categorical_encoded.shape)

print("Shape of Encoded X_test Categorical: ",X_test_categorical_encoded.shape)
X_test_categorical_encoded
c = SelectKBest(score_func=chi2)

numcols=c.fit(X_train_categorical_encoded,y_train)

plt.figure(figsize=(7,7))

sns.barplot(x=numcols.scores_,y=encodedCatColumnNames)

plt.title('Best Categorical Features')

plt.show()
## Creating dictionaries to store the feature names and its importance value

topCatFeatures={}



## https://machinelearningmastery.com/feature-selection-with-categorical-data/ 

for i in range(len(c.scores_)):

#     print('Feature %s: %f' % (numerical[i], n.scores_[i]))

    topCatFeatures[encodedCatColumnNames[i]]=c.scores_[i]



# SORT THE DICTIONARY AS PER THE IMPORTANT SCORES

topCatFeatures = sorted(topCatFeatures.items(), key=lambda x: x[1],reverse=True) 

print("-----------------------TOP FEATURES SORTED AS PER THE HIGH IMPORTANCE----------------")

topCatFeatures
X_train_merged = pd.concat([X_train_numerical_std,X_train_categorical_encoded], axis=1)

X_test_merged = pd.concat([X_test_numerical_std,X_test_categorical_encoded], axis=1)
X_train_numerical_std
print("Shape of X_train Merged: ",X_train_merged.shape)

print("Shape of X_test Merged: ",X_test_merged.shape)
print("Shape of y_train : ",y_train.shape)

print("Shape of y_test : ",y_test.shape)
X_train_merged.columns
y_train_corr=pd.DataFrame(y_train)

y_test_corr=pd.DataFrame(y_test)
y_train_corr.shape
y_train_corr=pd.DataFrame(y_train_corr,columns=['LOAN_DEFAULT'])

y_test_corr=pd.DataFrame(y_test_corr,columns=['LOAN_DEFAULT'])
X_train_corr=X_train_merged.copy()
X_train_corr['LOAN_DEFAULT']=y_train.values
corr_mat = X_train_corr.corr()



fig2=plt.figure()

sns.set(rc={'figure.figsize':(30,15)})

k = 34

cols = corr_mat.nlargest(k, 'LOAN_DEFAULT')['LOAN_DEFAULT'].index

cm = np.corrcoef(X_train_corr[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.title('Correlation Matrix')

plt.show()
# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    A =(((C.T)/(C.sum(axis=1))).T)

    B =(C/C.sum(axis=0))

    plt.figure(figsize=(20,4))

    

    labels = [1,2]

    # representing A in heatmap format

    cmap=sns.light_palette("blue")

    plt.subplot(1, 3, 1)

    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Confusion matrix")

    

    plt.subplot(1, 3, 2)

    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Precision matrix")

    

    plt.subplot(1, 3, 3)

    # representing B in heatmap format

    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Recall matrix")

    

    plt.show()
test_len=len(y_test)
predicted_y = np.zeros((test_len,2))

for i in range(test_len):

    rand_probs = np.random.rand(1,2)

    predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Test Data using Random Model",log_loss(y_test, predicted_y, eps=1e-15))



predicted_y =np.argmax(predicted_y, axis=1)

plot_confusion_matrix(y_test, predicted_y)
print("Accuracy On Random Model: ", 50.14)
alpha = [10 ** x for x in range(-5, 2)] # hyperparam for SGD classifier.



log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42,class_weight="balanced")

    clf.fit(X_train_merged, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(X_train_merged, y_train)

    predict_y = sig_clf.predict_proba(X_test_merged)

    log_error_array.append(log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    

    
fig, ax = plt.subplots()

ax.plot(alpha, log_error_array,c='g')

for i, txt in enumerate(np.round(log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.rcParams["figure.figsize"] = [10,7]

plt.show()
best_alpha = np.argmin(log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(X_train_merged, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(X_train_merged, y_train)



predict_y = sig_clf.predict_proba(X_train_merged)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(X_test_merged)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

predicted_y =np.argmax(predict_y,axis=1)

print("Total number of data points :", len(predicted_y))

plot_confusion_matrix(y_test, predicted_y)
print("Accuracy On Random Model: ", 78.27)
#Importing Machine Learning Model

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

    

#Bagging Algo

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier



#statistical Tools

from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import confusion_matrix, roc_curve, auc
lr = LogisticRegression(C=5.0,class_weight="balanced")

knn = KNeighborsClassifier(weights='distance', algorithm='auto', n_neighbors=15)

rfc = RandomForestClassifier(n_estimators=300,criterion='gini',class_weight="balanced")

dtc = DecisionTreeClassifier(class_weight="balanced")
accuracy = {}

roc_r = {}



def train_model(model):

    # Checking accuracy

    model = model.fit(X_train_merged, y_train)

    pred = model.predict(X_test_merged)

    acc = accuracy_score(y_test, pred)*100

    accuracy[model] = acc

    print('accuracy_score',acc)

    print('precision_score',precision_score(y_test, pred)*100)

    print('recall_score',recall_score(y_test, pred)*100)

    print('f1_score',f1_score(y_test, pred)*100)

    roc_score = roc_auc_score(y_test, pred)*100

    roc_r[model] = roc_score

    print('roc_auc_score',roc_score)

    # confusion matrix

    print('confusion_matrix')

    plot_confusion_matrix(y_test,pred)

#     print(pd.DataFrame(confusion_matrix(y_test, pred)))

    fpr, tpr, threshold = roc_curve(y_test, pred)

    roc_auc = auc(fpr, tpr)*100



    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.rcParams["figure.figsize"] = [7,7]

#     plt.figure(figsize=(2,3))

#     plt.figure(figsize=(3,4))

    plt.show()
train_model(lr)
train_model(dtc)
a=15

a
train_model(knn)
train_model(rfc)
xgb = XGBClassifier(scale_pos_weight=3)
train_model(xgb)
xgb = XGBClassifier(scale_pos_weight=2)
train_model(xgb)
from prettytable import PrettyTable 



# Specify the Column Names while initializing the Table 

myTable = PrettyTable(["Classifier", "Accuracy", "AUC"]) 



# Add rows 

myTable.add_row(["Random Model", "50.14", "50"]) 

myTable.add_row(["Logistic Regression->SGD", "78", "-"]) 

myTable.add_row(["Logistic Regression->SKLearn", "57", "59"]) 

myTable.add_row(["Decision Tree", "67", "52"]) 

myTable.add_row(["KNN", "77", "51"]) 

myTable.add_row(["Random Forest", "77", "51"]) 

myTable.add_row(["XGBoost v1", "65", "59"]) 

myTable.add_row(["XGBoost v2", "77", "53"]) 



print(myTable)

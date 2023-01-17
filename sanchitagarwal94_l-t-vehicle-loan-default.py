import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math

import seaborn as sns

import re

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,f1_score,recall_score, roc_auc_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import  RobustScaler
train = pd.read_csv("../input/lt-vehicle-loan-default-prediction/train.csv")

train.head()
train.shape
tr= train
tr.columns
tr.info()
tr.set_index('UniqueID', inplace =True)
tr['loan_default'].value_counts()
test = pd.read_csv('../input/lt-vehicle-loan-default-prediction/test.csv')

test.head()
ts = test.set_index('UniqueID')
ts.head()
ts.columns
tr.isnull().sum()
ts.isnull().sum()
ts.info()
## Train data showing the default proportions where 0 denotes as non-default and 1 denotes as default

tr.loan_default.value_counts().plot.bar()

plt.xlabel('Default Proportion')

plt.ylabel('customers')

plt.title('number of clients')

plt.show()
##Test data showing the employment info of the customers



ts['Employment.Type'].value_counts().plot.bar()

plt.xlabel('Default Proportion')

plt.ylabel('customers')

plt.title('number of clients')

plt.show()
ts['MobileNo_Avl_Flag'].count()
train['Employment.Type'].value_counts()
ts['Employment.Type'].value_counts()
tr.fillna('NAN',inplace=True)

ts.fillna('NAN',inplace=True)
tr['Employment.Type'].value_counts(normalize=True)
ts['Employment.Type'].value_counts(normalize=True)
#Creating function for checking the correlation between variables

def correlationplot(data,width):

    corr = data.corr()

    plt.figure(num=None,figsize=(width, width), dpi=80, facecolor='w', edgecolor='black')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title('Correlation Matrix')

    plt.show()
train.corr()
#Creating function for checking the relation between variables using histogram



def histogramplot(data, no_of_rows):

    nrow,ncol = data.shape

    for i in range (ncol,no_of_rows):

        plt.subplot(ncol,no_of_rows)

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.show()

        

histogramplot(tr,8)
tr.reset_index(inplace=True)
tr.head()
def print_all_values():

    df1=tr.drop('disbursed_amount',axis=1)

    cols=tr.columns

    for col in cols:

        if (tr[col].dtypes !='object'):



            fig1=plt.figure()

            ax1=plt.axes()

            plt.scatter(tr.disbursed_amount,tr[[col]],alpha=1)

            plt.title('Comparison of features with disbursed amount')

            ax1 = ax1.set(xlabel='disbursed_amount', ylabel=col)

            plt.show()

            

            

print_all_values()
def hist_all_values():

    df1=tr.drop('UniqueID',axis=1)

    cols=tr.columns

    for col in cols:

        if (tr[col].dtypes !='object'):



            fig1=plt.figure()

            tr.hist(column=col,grid=True, figsize=(12,8),bins=40)

            plt.title(col)

            plt.ylabel('counts')

            plt.xticks(rotation = 90)

            plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

            plt.show()

            

hist_all_values()
ts.reset_index(inplace=True)
ts.head()
def print_test_values():

    df1=ts.drop('disbursed_amount',axis=1)

    cols=ts.columns

    for col in cols:

        if (ts[col].dtypes !='object'):



            fig1=plt.figure()

            ax1=plt.axes()

            plt.scatter(ts.disbursed_amount,ts[[col]],alpha=1)

            plt.title('comparision of disbusred amount vs other features')

            ax1 = ax1.set(xlabel='disbursed_amount', ylabel=col)

            plt.show()

            

            

print_test_values()
def hist_test_values():

    df1=ts.drop('UniqueID',axis=1)

    cols=ts.columns

    for col in cols:

        if (ts[col].dtypes !='object'):



            fig1=plt.figure()

            ts.hist(column=col,grid=True, figsize=(12,8),bins=40)

            plt.title(col)

            plt.ylabel('counts')

            plt.xticks(rotation = 90)

            plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

            plt.show()

            

hist_test_values()
correlationplot(tr,8)
correlationplot(ts,8)
tr.head()
ts.head()
print(tr.shape)

print(ts.shape)
tr.boxplot(column='disbursed_amount', by='loan_default')
tr.boxplot(column='disbursed_amount', by='NO.OF_INQUIRIES')

ts.boxplot(column='disbursed_amount', by='NO.OF_INQUIRIES')
tr.head()
# creating a function to split the credit risk into risk grade and risk type

def credit_risk(tr):

    d1=[]

    d2=[]

    for i in tr:

        a = i.split("-")

        if len(a) == 1:

            d1.append(a[0])

            d2.append('unknown')

        else:

            d1.append(a[1])

            d2.append(a[0])



    return d1,d2
def calc_number_of_ids(row):

#     print(type(row), row.size)

    return sum(row[['Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag',

       'Passport_flag']])

def check_pri_installment(row):

    if row['PRIMARY.INSTAL.AMT']<=1:

        return 0

    else:

        return row['PRIMARY.INSTAL.AMT']
# Now converting the Score description into number rating from 0 to -5



risk_map = {'No Bureau History Available':-1, 

              'Not Scored: No Activity seen on the customer (Inactive)':-1,

              'Not Scored: Sufficient History Not Available':-1,

              'Not Scored: No Updates available in last 36 months':-1,

              'Not Scored: Only a Guarantor':-1,

              'Not Scored: More than 50 active Accounts found':-1,

              'Not Scored: Not Enough Info available on the customer':-1,

              'Very Low Risk':4,

              'Low Risk':3,

              'Medium Risk':2, 

              'High Risk':1,

              'Very High Risk':0}



#Have used the grading system in descending order because A is least risky and going forward risk increases

sub_risk = {'unknown':-1, 'I':5, 'L':2, 'A':13, 'D':10, 'M':1, 'B':12, 'C':11, 'E':9, 'H':6, 'F':8, 'K':3,

       'G':7, 'J':4}



#Firstly converting the employment type to numbers:



employment_map = {'Self employed':0, 'Salaried':1, 'NAN':-1}

def features_engineering(df):

    



# Now converting the Date of birth of customers into the age and creating a new feature age:



    df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'], format = "%d-%m-%y")

    now = pd.Timestamp('now')

    df['Age'] = (now - df['Date.of.Birth']).astype('<m8[Y]').astype(int)

    age_mean = int(df[df['Age']>0]['Age'].mean())

    df.loc[:,'age'] = df['Age'].apply(lambda x: x if x>0 else age_mean)



# Now converting the Disbursal date of loan into no. of month passed from disbural month.



    df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format = "%d-%m-%y")

    df['disbursal_months_passed'] = ((now - df['DisbursalDate'])/np.timedelta64(1,'M')).astype(int)



#Now converting AVERAGE.ACCT.AGE into number of months :

    df['average_act_age_in_months'] = df['AVERAGE.ACCT.AGE'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))



# Now Converting CREDIT.HISTORY.LENGTH into number of months:



    df['credit_history_length_in_months'] = df['CREDIT.HISTORY.LENGTH'].apply(lambda x : int(re.findall(r'\d+',x)[0])*12 + int(re.findall(r'\d+',x)[1]))



#adding a feature of number of zeroes present in a row so that we can count how many zeroes on row has



    df['number_of_0'] = (df == 0).astype(int).sum(axis=1)

    

#creating additional column to split the PERFORM_CNS.SCORE.DESCRIPTION using credit risk function defined above



    df.loc[:,'credit_risk'],df.loc[:,'credit_risk_grade']  = credit_risk(df["PERFORM_CNS.SCORE.DESCRIPTION"])



#adding loan to asset ratio to check which if the clients with default had suufficient assets to repay loan at time of disbursement



    df.loc[:, 'loan_to_asset_ratio'] = df['disbursed_amount'] /df['asset_cost']



#adding total number of accounts feature:



    df.loc[:,'no_of_accts'] = df['PRI.NO.OF.ACCTS'] + df['SEC.NO.OF.ACCTS']



#Now adding columns carrying total number of  various accounts including the primary and secondary and combing them in one



    df.loc[:,'pri_inactive_accts'] = df['PRI.NO.OF.ACCTS'] - df['PRI.ACTIVE.ACCTS']

    df.loc[:,'sec_inactive_accts'] = df['SEC.NO.OF.ACCTS'] - df['SEC.ACTIVE.ACCTS']

    df.loc[:,'tot_inactive_accts'] = df['pri_inactive_accts'] + df['sec_inactive_accts']

    df.loc[:,'tot_overdue_accts'] = df['PRI.OVERDUE.ACCTS'] + df['SEC.OVERDUE.ACCTS']

    df.loc[:,'tot_current_balance'] = df['PRI.CURRENT.BALANCE'] + df['SEC.CURRENT.BALANCE']

    df.loc[:,'tot_sanctioned_amount'] = df['PRI.SANCTIONED.AMOUNT'] + df['SEC.SANCTIONED.AMOUNT']

    df.loc[:,'tot_disbursed_amount'] = df['PRI.DISBURSED.AMOUNT'] + df['SEC.DISBURSED.AMOUNT']

    df.loc[:,'tot_installment'] = df['PRIMARY.INSTAL.AMT'] + df['SEC.INSTAL.AMT']

    df.loc[:,'bal_disburse_ratio'] = np.round((1+df['tot_disbursed_amount'])/(1+df['tot_current_balance']),2)

    df.loc[:,'pri_tenure'] = (df['PRI.DISBURSED.AMOUNT']/( df['PRIMARY.INSTAL.AMT']+1)).astype(int)

    df.loc[:,'sec_tenure'] = (df['SEC.DISBURSED.AMOUNT']/(df['SEC.INSTAL.AMT']+1)).astype(int)

    df.loc[:,'disburse_to_sactioned_ratio'] =  np.round((df['tot_disbursed_amount']+1)/(1+df['tot_sanctioned_amount']),2)

    df.loc[:,'active_to_inactive_act_ratio'] =  np.round((df['no_of_accts']+1)/(1+df['tot_inactive_accts']),2)

    return df

# adding features for the credit risk and sub risk for which we have described numbers and grades above  

def label_data(df):

    df.loc[:,'credit_risk_label'] = df['credit_risk'].apply(lambda x: risk_map[x])

    df.loc[:,'sub_risk_label'] = df['credit_risk_grade'].apply(lambda x: sub_risk[x])

    return df
def data_correction(df):

    #Many customers have invalid date of birth, so immute invalid data with mean age

    df.loc[:,'PRI.CURRENT.BALANCE'] = df['PRI.CURRENT.BALANCE'].apply(lambda x: 0 if x<0 else x)

    df.loc[:,'SEC.CURRENT.BALANCE'] = df['SEC.CURRENT.BALANCE'].apply(lambda x: 0 if x<0 else x)

    df.loc[:,'employment_label'] = df['Employment.Type'].apply(lambda x: employment_map[x])



    #loan that do not have current pricipal outstanding should have 0 primary installment

    df.loc[:,'new_pri_installment']= df.apply(lambda x : check_pri_installment(x),axis=1)

    return df
def new_data(df):

    df = data_correction(df)

    df = features_engineering(df)

    df = label_data(df)



    return df
train_data = new_data(tr)

train_data = train_data[train_data['number_of_0']<=25]

test_data = new_data(ts)

train_data[train_data['number_of_0']>=20]['number_of_0'].value_counts()
train_data.columns

features = ['disbursed_amount', 'asset_cost',

            'Aadhar_flag', 'PAN_flag',

       'PERFORM_CNS.SCORE',

             'PRI.ACTIVE.ACCTS',

       'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',

       'PRI.DISBURSED.AMOUNT',  'SEC.ACTIVE.ACCTS',

       'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',

       'SEC.DISBURSED.AMOUNT',  'SEC.INSTAL.AMT',

       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',

            'NO.OF_INQUIRIES','disbursal_months_passed',

       'average_act_age_in_months', 'credit_history_length_in_months',

       'number_of_0','loan_to_asset_ratio', 'no_of_accts', 'pri_inactive_accts',

       'sec_inactive_accts', 'tot_inactive_accts', 'tot_overdue_accts',

       'tot_current_balance', 'tot_sanctioned_amount', 'tot_disbursed_amount',

       'tot_installment', 'bal_disburse_ratio', 'pri_tenure', 'sec_tenure',

       'credit_risk_label',

       'employment_label', 'age', 'new_pri_installment'

           ]
print(train_data.shape)

print(test_data.shape)

# std_scaler = StandardScaler()

# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()



scaled_training = train_data.copy()

scaled_testing = test_data.copy()





scaled_training[features] = rob_scaler.fit_transform(scaled_training[features])

scaled_testing[features] = rob_scaler.fit_transform(scaled_testing[features])



y = scaled_training.loan_default

X = scaled_training[features]



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27,stratify=y)

print(X_train.shape, y_train.shape)

print(X_test.shape,y_test.shape)

#Random Forest Testing

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
rfc_pre= rfc.predict(X_test)
accuracy_score(y_test, rfc_pre)
print(confusion_matrix(y_test, rfc_pre))

print(classification_report(y_test, rfc_pre))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=21, stratify=y)

print(X_train.shape, y_train.shape)
# Testing Logistic Regression

logreg= LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve

rfc_pre_prob = rfc.predict_proba(X_test)[:,1]

fpr, tpr , thresholds = roc_curve(y_test, rfc_pre_prob)



plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label= 'Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('RFC ROC Curve')

plt.show()
logreg.predict_proba(X_test)[:,1]
# Trying to use K fold
# Verifying the result of RFC using GridCVsearch 

from sklearn.model_selection import cross_val_score

cv_results = cross_val_score(rfc,X,y,cv=5)
cv_results
np.mean(cv_results)
#Knn method

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(2)

knn.fit(X_train,y_train)

y_pred_knn= knn.predict(X_test)

print(confusion_matrix(y_test, y_pred_knn))

print(classification_report(y_test,y_pred_knn))
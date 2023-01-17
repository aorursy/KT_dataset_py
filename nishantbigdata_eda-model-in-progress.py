import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
app_train = pd.read_csv('../input/application_train.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
#Lets check the application_train file
app_train.head()
#Lets check how many loans are cash loans and how many of them are Revolving loans
app_train['NAME_CONTRACT_TYPE'].value_counts()
#Percentage of cash loan vs revolving loans
app_train['NAME_CONTRACT_TYPE'].value_counts(True)
#Now lets observe how many are cash loans and revolving loan out of all the lons where customer with payment difficulties
# for that first we need to convert the values in NAME_CONTRACT_TYPE columns with numerical values, and lets store the values in a new column 
app_train['NUM_NAME_CONTRACT_TYPE'] = app_train['NAME_CONTRACT_TYPE'].map({'Cash loans':1,'Revolving loans':0 })
app_train[app_train['TARGET'] == 1]['NUM_NAME_CONTRACT_TYPE'].value_counts(True)
#lets see how many of the loans are given to customers with payment diificulty and other cases
app_train['TARGET'].value_counts()
app_train['TARGET'].value_counts(True)
#lets see how many customers having payment difficulties are male
app_train['NUM_CODE_GENDER'] = app_train['CODE_GENDER'].map({'M':1,'F':0 })
app_train[app_train['TARGET'] == 1]['NUM_CODE_GENDER'].value_counts(True)
#FLAG_OWN_CAR	FLAG_OWN_REALTY	CNT_CHILDREN
#Similarly lets observe how many customers having payment difficulties are having there own car
app_train['NUM_FLAG_OWN_CAR'] = app_train['FLAG_OWN_CAR'].map({'Y':1,'N':0 })
app_train[app_train['TARGET'] == 1]['NUM_FLAG_OWN_CAR'].value_counts(True)
#Similarly lets observe how many customers having payment difficulties are having there own house or flat
app_train['NUM_FLAG_OWN_REALTY'] = app_train['FLAG_OWN_REALTY'].map({'Y':1,'N':0 })
app_train[app_train['TARGET'] == 1]['NUM_FLAG_OWN_REALTY'].value_counts(True)
#Similarly, lets check for the number of children of customers having payment difficulty
app_train[app_train['TARGET'] == 1]['CNT_CHILDREN'].value_counts(True)
#Now lets just filter the cases where Target = 1
app_train_t1 =  app_train[app_train['TARGET'] == 1]
#18 July

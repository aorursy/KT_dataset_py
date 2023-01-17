import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
warnings.filterwarnings('ignore')
d1 = pd.read_csv("../input/ecommerce-fraud-data/Customer_DF (1).csv")
d1.columns
d1.head()
d1.info()
d1.describe()
d2 = pd.read_csv('../input/ecommerce-fraud-data/cust_transaction_details (1).csv')
d2.columns
d2.head()
d2.info()
d2.describe()
d1['customerEmail'].nunique()
d2['customerEmail'].nunique()
mail_list=[]
repeat =0
result={}
for i in range(0,168):
    repeat = 0
    for j in range(0,168):
        if d1['customerEmail'][i] == d1['customerEmail'][j]:
            repeat+=1
    result.update({d1['customerEmail'][i]:repeat})
result    
d1[d1['customerEmail']=='johnlowery@gmail.com']
d2['paymentMethodType'].unique()
sns.countplot(d2['paymentMethodType'],data = d2)
sns.countplot(d2['orderState'])
plt.figure(figsize=(16,5))
sns.countplot(d2['paymentMethodProvider'])
plt.tight_layout()
plt.figure(figsize=(15,5))
sns.countplot(d2['paymentMethodProvider'],hue = d2['paymentMethodRegistrationFailure'])
plt.tight_layout()
sns.countplot(d1['No_Payments'],hue = d1['Fraud'])
l = []
for i in range(0,168):
    uncommon=0
    for j in range(0,623):
        if d1['customerEmail'][i]==d2['customerEmail'][j]:
            uncommon+=1
    if uncommon==0:
        l.append(d1['customerEmail'][i])
print(len(l))
l
common =0
for i in d1['customerEmail']:
    for email in d2['customerEmail']:
        if i==email:
            common+=1
            break
common
final = d1[d1['customerEmail'].isin(d2['customerEmail'])== True]
final.shape
final.drop('Unnamed: 0',axis = 1, inplace = True)
final.reset_index(inplace = True)
Total_transaction_amt = []
for i in range(0,143):
    s=0
    for j in range(0,623):
        if(final['customerEmail'][i]==d2['customerEmail'][j]):
            s += d2['transactionAmount'][j]
    Total_transaction_amt.append(s)        

final['Total_transaction_amt'] = Total_transaction_amt
No_transactionsFail = []
for i in range(0,143):
    s=0
    for j in range(0,623):
        if(final['customerEmail'][i]==d2['customerEmail'][j]):
            s += d2['transactionFailed'][j]
    No_transactionsFail.append(s)        
final['No_transactionsFail'] = No_transactionsFail
PaymentRegFail = []
for i in range(0,143):
    s=0
    for j in range(0,623):
        if(final['customerEmail'][i]==d2['customerEmail'][j]):
            s += d2['paymentMethodRegistrationFailure'][j]
    PaymentRegFail.append(s)  
final['PaymentRegFail'] = PaymentRegFail
def col_make(column_name,category):
    array = []
    for i in range(0,143):
        s=0
        for j in range(0,623):
            if(final['customerEmail'][i]==d2['customerEmail'][j]):
                if d2[column_name][j]==category:
                    s+=1
        array.append(s)
    return array 
PaypalPayments = col_make('paymentMethodType','paypal')
ApplePayments = col_make('paymentMethodType','apple pay')
BitcoinPayments = col_make('paymentMethodType','bitcoin')
CardPayments = col_make('paymentMethodType','card')
final['PaypalPayments']= PaypalPayments
final['ApplePayments']= ApplePayments
final['CardPayments']= CardPayments
final['BitcoinPayments']= BitcoinPayments
OrdersFulfilled = col_make('orderState','fulfilled')
OrdersFailed =  col_make('orderState','failed')
OrdersPending = col_make('orderState','pending')
final['OrdersFulfilled'] = OrdersFulfilled
final['OrdersPending'] = OrdersPending
final['OrdersFailed'] = OrdersFailed
JCB_16 = col_make('paymentMethodProvider','JCB 16 digit')
AmericanExp = col_make('paymentMethodProvider','American Express')
VISA_16 =  col_make('paymentMethodProvider','VISA 16 digit')
Discover =  col_make('paymentMethodProvider','Discover')
Voyager = col_make('paymentMethodProvider','Voyager')
VISA_13 = col_make('paymentMethodProvider','VISA 13 digit')
Maestro = col_make('paymentMethodProvider','Maestro')
Mastercard = col_make('paymentMethodProvider','Mastercard')
DC_CB =col_make('paymentMethodProvider','Diners Club / Carte Blanche')
JCB_15= col_make('paymentMethodProvider','JCB 15 digit')
final['JCB_16'] = JCB_16
final['AmericanExp'] = AmericanExp 
final['VISA_16'] = VISA_16 
final['Discover'] = Discover
final['Voyager'] = Voyager 
final['VISA_13'] = VISA_13
final['Maestro'] = Maestro 
final['Mastercard'] = Mastercard
final['DC_CB'] = DC_CB 
final['JCB_15'] = JCB_15
final.shape
Trns_fail_order_fulfilled = []
for i in range(0,143):
    s=0
    for j in range(0,623):
        if(final['customerEmail'][i]==d2['customerEmail'][j]):
            if (d2['orderState'][j]=='fulfilled') & (d2['transactionFailed'][j]==1):
                s+=1
    Trns_fail_order_fulfilled.append(s)
final['Trns_fail_order_fulfilled'] = Trns_fail_order_fulfilled
Duplicate_IP = []
for i in range(0,143):
    s=0
    for j in range(0,143):
        if(final['customerIPAddress'][i]==final['customerIPAddress'][j]):
            s+=1
    s-=1        
    Duplicate_IP.append(s)
final['Duplicate_IP'] = Duplicate_IP
Fraud_Decoded = []
for i in range(0,143):
    s=0
    if(final['Fraud'][i]==True):
        s+=1        
    Fraud_Decoded.append(s)
final['Fraud_Decoded'] = Fraud_Decoded
Duplicate_Address = []
for i in range(0,143):
    s=0
    for j in range(0,143):
        if(final['customerBillingAddress'][i]==final['customerBillingAddress'][j]):
            s+=1
    s-=1        
    Duplicate_Address.append(s)
final['Duplicate_Address']=Duplicate_Address
final[final['Fraud']==True].count()
final.head()
sns.barplot(x = final['No_Transactions'],y = final['No_transactionsFail'],hue = final['Fraud'])
final[(final['No_transactionsFail'] == 6) & (final['No_Transactions']==0)==True]
print(final['customerPhone'].nunique())
print(final['customerDevice'].nunique())
print(final['customerIPAddress'].nunique())
print(final['customerBillingAddress'].nunique())
final[final['Duplicate_IP']>0]
final[final['Duplicate_Address']>0]
sns.countplot(x = final['OrdersFulfilled'], hue = final['Fraud'])
final.columns
X = final[['No_Transactions',
       'No_Orders', 'No_Payments', 'Total_transaction_amt',
       'No_transactionsFail', 'PaymentRegFail', 'PaypalPayments',
       'ApplePayments', 'CardPayments', 'BitcoinPayments', 'OrdersFulfilled',
       'OrdersPending', 'OrdersFailed','Trns_fail_order_fulfilled','Duplicate_IP','Duplicate_Address','JCB_16', 'AmericanExp', 'VISA_16',
       'Discover', 'Voyager', 'VISA_13', 'Maestro', 'Mastercard', 'DC_CB',
       'JCB_15']]
y = final['Fraud_Decoded']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
rfc = RandomForestClassifier(n_estimators=150)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)
print(accuracy_score(y_test,pred))
sns.heatmap(data = confusion_matrix(y_test,pred),annot = True)
print(classification_report(y_test,pred))
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train,y_train)
log_pred =logr.predict(X_test)
print(accuracy_score(y_test,log_pred))
sns.heatmap(data=confusion_matrix(y_test,log_pred),annot = True)
print(classification_report(y_test,log_pred))
from sklearn.svm import SVC
svc = SVC(gamma = 'auto')
svc.fit(X_train,y_train)
svc_pred=svc.predict(X_test)
print(accuracy_score(y_test,pred))
sns.heatmap(data = confusion_matrix(y_test,pred),annot = True)
print(classification_report(y_test,pred))
from sklearn.model_selection import GridSearchCV
svc_param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
gridsvc = GridSearchCV(SVC(),svc_param_grid,refit=True,verbose=3)
gridsvc.fit(X_train,y_train)
gridsvc.best_params_
gridsvc.best_estimator_
grid_svc_predictions = gridsvc.predict(X_test)
print(accuracy_score(y_test,grid_svc_predictions))
sns.heatmap(data = confusion_matrix(y_test,grid_svc_predictions),annot= True)
print(classification_report(y_test,grid_svc_predictions))
logr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
grid_logr = GridSearchCV(LogisticRegression(),logr_param_grid,refit=True,verbose=3)
grid_logr.fit(X_train,y_train)
grid_logr.best_params_
grid_logr.best_estimator_
grid_logr_predictions = grid_logr.predict(X_test)
print(accuracy_score(y_test,grid_logr_predictions))
sns.heatmap(data = confusion_matrix(y_test,grid_logr_predictions),annot = True)
print(classification_report(y_test,grid_logr_predictions))
rfc_param_grid = { 
    'n_estimators': [100,150,200,350,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
grid_rfc = GridSearchCV(RandomForestClassifier(),rfc_param_grid,refit=True,verbose=3)
grid_rfc.fit(X_train,y_train)
grid_rfc.best_params_
grid_rfc.best_estimator_
grid_rfc_predictions = grid_rfc.predict(X_test)
print(accuracy_score(y_test,grid_rfc_predictions))
sns.heatmap(data = confusion_matrix(y_test,grid_rfc_predictions),annot = True)
print(classification_report(y_test,grid_rfc_predictions))
rfc.feature_importances_
from sklearn.model_selection import cross_val_score
cv_scores_rfc = cross_val_score(grid_rfc.best_estimator_, X, y, cv=5)
print(cv_scores_rfc)
print("Mean 5-Fold R Squared: {}".format(np.mean(cv_scores_rfc)))
cv_scores_logr = cross_val_score(grid_logr.best_estimator_, X, y, cv=5)
print(cv_scores_logr)
print("Mean 5-Fold R Squared: {}".format(np.mean(cv_scores_logr)))
cv_scores_svc = cross_val_score(gridsvc.best_estimator_, X, y, cv=5)
print(cv_scores_svc)
print("Mean 5-Fold R Squared: {}".format(np.mean(cv_scores_svc)))

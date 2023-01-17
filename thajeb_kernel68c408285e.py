import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/seleksidukungaib/train.csv')
test = pd.read_csv('../input/seleksidukungaib/test.csv')
train.shape, test.shape
train.head(10)
train.isnull().sum()
train.drop(['idx','userId','date_collected'],axis=1).corr()
train['date_collected'].equals(train['date'])
sns.distplot(train['max_recharge_trx'],kde=False,rug=True)
sns.distplot(train['average_topup_trx'],kde=False,rug=True)
sns.distplot(train['average_transfer_trx'],kde=False,rug=True)
sns.distplot(train['num_transfer_trx'],kde=False,rug=True)
sns.countplot(x='isVerifiedPhone',hue='isChurned',data=train,palette='RdBu_r')
noPhoneVerified = len(train[train['isVerifiedPhone']==0].index)
print(noPhoneVerified)
print(noPhoneVerified/len(train))
sns.countplot(x='isVerifiedEmail',hue='isChurned',data=train,palette='RdBu_r')
sns.countplot(x='isUpgradedUser',hue='isChurned',data=train,palette='RdBu_r')
sns.countplot(x='blocked',hue='isChurned',data=train,palette='RdBu_r')
len(train[train['blocked']==1].index)/len(train.index)
sns.countplot(x='premium',hue='isChurned',data=train,palette='RdBu_r')
sns.countplot(x='super',hue='isChurned',data=train,palette='RdBu_r')
len(train[train['super']==True].index)/len(train.index)
sns.countplot(x='userLevel',hue='isChurned',data=train,palette='RdBu_r')
sns.countplot(x='pinEnabled',hue='isChurned',data=train,palette='RdBu_r')
sns.distplot(train['average_recharge_trx'])
sns.distplot(train['total_transaction'])
sns.countplot(x='isActive',hue='isChurned',data=train,palette='RdBu_r')
#Some Analysis
# -column date_collected and column date are exactly the same, I will drop either one
# -column random_number has no explicitly meaningful information and it has very low correlation towars all others column according to correlation matrix so I will drop it
# -there are a lot of column contains null value, i will impute some of them which contains a lot of null value and have high correlation towards column isChurned which are
#  column 'max_reharge_trx','average_topup_trx','total_transaction', and 'average_topup_trx'. Meanwhile for columns that contain very few null value, i will just fill null with 0
# -Value in 'average_transfer_trx' is either 0 or null, and furthermore using data visualization value for 'num_topup_trx' is constant 0 and since the num of tranfer is 0 
#  it makes sense that max_transfer_trx and min_topup_trx are all constant 0. So instead of impute 'average_transfer_trx', i will just drop all 'transfer' related column
# -Based on correlation matrix and data visualization column 'isVerifiedPhone','super','blocked','isActive','isUpgradedUser' is almost constant (has a very low variance)
#  and furthermore all of them doesn't have very high correlation towards column 'isChurned' so I will drop them.
# -UserId can be replace with idx which is also a unique identification so i will drop column userId
#Imputation Function
def imputeMaxRecharge(data):
    MinRecharge = data[0]
    AvgRecharge = data[1]
    MaxRecharge = data[2]
    if pd.isnull(MaxRecharge):
        return AvgRecharge + 2.5 *(AvgRecharge-MinRecharge) 
    return MaxRecharge
def imputeAverageTopUpTrx(data):
    minTopUp = data[0]
    averageTopUp = data[1]
    maxTopUp = data[2]
    if pd.isnull(averageTopUp):   
        return (maxTopUp + minTopUp)/2
    return averageTopUp
def imputeTotalTransaction(data):
    avgRecharge = data[0]
    avgTopUp = data[1]
    totalTransaction = data[2]
    if pd.isnull(totalTransaction):
        return avgRecharge + avgTopUp
    return totalTransaction
train['max_recharge_trx'] = train[['min_recharge_trx','average_recharge_trx','max_recharge_trx']].apply(imputeMaxRecharge,axis=1)
train['average_topup_trx'] = train[['min_topup_trx','average_topup_trx','max_topup_trx']].apply(imputeAverageTopUpTrx,axis=1)
train['total_transaction'] = train[['average_recharge_trx','average_topup_trx','total_transaction']].apply(imputeTotalTransaction,axis=1)
train.drop(['isActive','isUpgradedUser','random_number','date'],axis=1,inplace=True)
train.drop(['isVerifiedPhone','super','blocked','userId'],axis=1,inplace=True)
train.drop(['min_transfer_trx','average_transfer_trx','max_transfer_trx','num_transfer_trx'],axis=1,inplace=True)
train.fillna(0,inplace=True)
columns = list(filter(lambda x: x != 'isChurned',list(train.columns.values)))
test = test[columns]
print(list(test.columns))
print(list(train.columns))
data = pd.concat([train,test],ignore_index=True)
date = ['date_collected']
bin = ['premium','pinEnabled']
col = date + bin
le = LabelEncoder()
for i in col: 
    data[i] = le.fit_transform(list(data[i].values))
# Optional step for some algorithm
scaled_feature = ['num_recharge_trx','min_recharge_trx','average_recharge_trx','max_recharge_trx','num_topup_trx','min_topup_trx','average_topup_trx','max_topup_trx','num_transaction','total_transaction']
data[scaled_feature] = StandardScaler().fit_transform(data[scaled_feature])
train = data[~data.isChurned.isnull()]
test = data[data.isChurned.isnull()]
#Splitting train dataset for analysis purpose
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(train.drop(['isChurned','idx'],axis=1), 
                                                    train['isChurned'], test_size=0.30, 
                                                    random_state=101)
model = LogisticRegression()
model.fit(X_train_train,y_train_train)
train_pred = model.predict(X_train_test)
print(confusion_matrix(y_train_test,train_pred))
print(classification_report(y_train_test,train_pred))
#KNN
# error_rate = []
# for i in range(1,25):
#     model = KNeighborsClassifier(n_neighbors=i)
#     model.fit(X_train_train,y_train_train)
#     pred_i = model.predict(X_train_test)
#     error_rate.append(np.mean(pred_i != y_train_test))
# plt.figure(figsize=(10,6))
# plt.plot(range(1,25),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate') 
# With visualization above, i found that with n_neighbors=22 result in the least error_rate 
model = KNeighborsClassifier(n_neighbors=22)
model.fit(X_train_train,y_train_train)
train_pred = model.predict(X_train_test)
print(confusion_matrix(y_train_test,train_pred))
print(classification_report(y_train_test,train_pred))
params = {'n_estimators':[100,200,300,500,800],'max_depth': [10,20,30],'min_samples_split':[2,5,10],'min_samples_leaf':[2,5,10] }
# grid_search = RandomizedSearchCV(estimator= RandomForestClassifier(),param_distributions = params, n_iter = 30,cv=3,verbose=2)
# grid_search.fit(X_train_train,y_train_train)
# grid_search.best_params_
model = RandomForestClassifier(n_estimators=300,max_depth=30,min_samples_split=5,min_samples_leaf=5)
model.fit(X_train_train,y_train_train)
train_pred = model.predict(X_train_test)
print(confusion_matrix(y_train_test,train_pred))
print(classification_report(y_train_test,train_pred))
params = {'n_estimators': [100,200,300,500,800],'gamma': [1,10,50,100],'max_depth': [3, 6,10],'learning_rate': [0.1, 0.01, 0.05]}
# grid_search =GridSearchCV(estimator= xgb.XGBClassifier(),param_grid=params,cv=3,verbose=2)
# grid_search.fit(X_train_train,y_train_train)
# grid_search.best_params_
#XGBoosting
model = xgb.XGBClassifier(gamma=1,learning_rate=0.05,max_depth=6,n_estimators=800)
model.fit(X_train_train,y_train_train)
train_pred = model.predict(X_train_test)
print(confusion_matrix(y_train_test,train_pred))
print(classification_report(y_train_test,train_pred))
X_train = train.drop(['isChurned','idx'],axis=1)
y_train = train['isChurned']
#Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(test.drop(['isChurned','idx'],axis=1))
#KNN
# model = KNeighborsClassifier(n_neighbors=22)
# model.fit(X_train,y_train)
# pred = model.predict(test.drop(['isChurned','idx'],axis=1))
#Random Forest
# model = RandomForestClassifier(n_estimators=300,max_depth=30,min_samples_split=5,min_samples_leaf=5)
# model.fit(X_train,y_train)
# pred = model.predict(test.drop(['isChurned','idx'],axis=1))
#XG Boosting
# model = xgb.XGBClassifier(gamma=1,learning_rate=0.05,max_depth=6,n_estimators=800)
# model.fit(X_train,y_train)
# pred = model.predict(test.drop(['isChurned','idx'],axis=1))
submission = pd.DataFrame({'idx':test['idx'],'isChurned':pred.astype(int)})
submission.to_csv('submission13.csv',index=False)
submission
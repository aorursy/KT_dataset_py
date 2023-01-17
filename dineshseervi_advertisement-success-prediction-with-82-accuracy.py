import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/advertsuccess/Train.csv')
test=pd.read_csv('/kaggle/input/advertsuccess/Test.csv')
print('total train data: ' + str(train.shape[0]))
print('total test data: ' + str(test.shape[0]))
train.describe()
train.info()
train.isnull().sum()
from sklearn import preprocessing

le1 = preprocessing.LabelEncoder()
le1.fit(train['realtionship_status'])
list(le1.classes_)
train['realtionship_status'] = le1.transform(train['realtionship_status'])
train.head()
le2 = preprocessing.LabelEncoder()
le2.fit(train['industry'])
list(le2.classes_)
train['industry'] = le2.transform(train['industry']) 
train.head()
le3 = preprocessing.LabelEncoder()
le3.fit(train['genre'])
list(le3.classes_)
train['genre'] = le3.transform(train['genre']) 
train.head()
le4 = preprocessing.LabelEncoder()
le4.fit(train['targeted_sex'])
list(le4.classes_)
train['targeted_sex'] = le4.transform(train['targeted_sex']) 
train.head()
le5 = preprocessing.LabelEncoder()
le5.fit(train['airtime'])
list(le5.classes_)
train['airtime'] = le5.transform(train['airtime']) 
train.head()
le6 = preprocessing.LabelEncoder()
le6.fit(train['airlocation'])
list(le6.classes_)
train['airlocation'] = le6.transform(train['airlocation']) 
train.head()
le7 = preprocessing.LabelEncoder()
le7.fit(train['expensive'])
list(le7.classes_)
train['expensive'] = le7.transform(train['expensive'])
train.head()
le8 = preprocessing.LabelEncoder()
le8.fit(train['money_back_guarantee'])
list(le8.classes_)
train['money_back_guarantee'] = le8.transform(train['money_back_guarantee'])
train.head()
le9 = preprocessing.LabelEncoder()
le9.fit(train['netgain'])
list(le9.classes_)
train['netgain'] = le9.transform(train['netgain'])
train.head()
#Considering all available features for decision tree classifier
features = ['realtionship_status','industry','genre','targeted_sex','average_runtime(minutes_per_week)','airtime','airlocation','ratings','expensive','money_back_guarantee']
X = train[features]
y = train['netgain']
# Decision tree classifier and model evaluation using kFold cross validation

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

kfold_mae_train=0
kfold_mae_test=0
kfold_f_imp_dic = 0

no_of_folds = 5

kf = KFold(no_of_folds,True,1)
for train_index, test_index in kf.split(X):
    
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    
    dt_classifier = DecisionTreeClassifier(random_state=1)
    dt_classifier.fit(X_train,y_train)
    
    mae_train = mean_absolute_error(dt_classifier.predict(X_train),y_train)
    kfold_mae_train=(kfold_mae_train+mae_train)
    
    mae_test = mean_absolute_error(dt_classifier.predict(X_test),y_test)
    kfold_dt_mae_test = (kfold_mae_test+mae_test)
    
    kfold_f_imp_dic = kfold_f_imp_dic + dt_classifier.feature_importances_
    
print('Decision Tree classifier train set mean absolute error =',kfold_mae_train/no_of_folds)
print('Decision Tree classifier test set mean absolute error  =',kfold_dt_mae_test/no_of_folds)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,accuracy_score,mean_squared_error
from sklearn.model_selection import KFold

kfold_mae_train=0
kfold_mae_test=0
kfold_mse_train=0
kfold_mse_test=0
kfold_f_imp_dic = 0

no_of_folds = 5
kf = KFold(no_of_folds,True,1)
for train_index, test_index in kf.split(X):
    
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    
    rf_classifier = RandomForestClassifier(random_state=1)
    rf_classifier.fit(X_train,y_train)
    
    mae_train = mean_absolute_error(rf_classifier.predict(X_train),y_train)
    kfold_mae_train=(kfold_mae_train+mae_train)
    
    mse_train = mean_squared_error(rf_classifier.predict(X_train),y_train)
    kfold_mse_train=(kfold_mse_train+mse_train)
    
    mae_test = mean_absolute_error(rf_classifier.predict(X_test),y_test)
    kfold_rf_mae_test = (kfold_mae_test+mae_test)
    
    mse_test = mean_squared_error(rf_classifier.predict(X_test),y_test)
    kfold_mse_test=(kfold_mse_test+mse_test)
    
    kfold_f_imp_dic = kfold_f_imp_dic + rf_classifier.feature_importances_
    
print('Random Forest classifier train set mean absolute error =',kfold_mae_train/no_of_folds)
print('Random Forest classifier test set mean absolute error  =',kfold_rf_mae_test/no_of_folds)
print('Random Forest Classifier train set mean Squared error =',kfold_mse_train/no_of_folds)
print('Random Forest Classifier test set mean Squared error =',kfold_mse_test/no_of_folds)
#rfc2=RandomForestClassifier()
#rfc2.fit(X_train,y_train)
#model on train using all the independent values in df
rfc_prediction = rf_classifier.predict(X_train)
rfc_score= accuracy_score(y_train,rfc_prediction)
print('Random Forest classifier Train set accuracy score ',rfc_score)
#model on test using all the indpendent values in df
rfc_prediction = rf_classifier.predict(X_test)
rfc_score= accuracy_score(y_test,rfc_prediction)
print('Random Forest Classifier Test Set accuracy score ',rfc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,accuracy_score,mean_squared_error
from sklearn.model_selection import KFold

kfold_mae_train=0
kfold_mae_test=0
kfold_mse_train=0
kfold_mse_test=0
kfold_f_imp_dic = 0

no_of_folds = 5
kf = KFold(no_of_folds,True,1)
for train_index, test_index in kf.split(X):
    
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    
    lr_classifier = LogisticRegression(random_state=1)
    lr_classifier.fit(X_train,y_train)
    
    mae_train = mean_absolute_error(lr_classifier.predict(X_train),y_train)
    kfold_mae_train=(kfold_mae_train+mae_train)
    
    mse_train = mean_squared_error(lr_classifier.predict(X_train),y_train)
    kfold_mse_train=(kfold_mse_train+mse_train)
    
    mae_test = mean_absolute_error(lr_classifier.predict(X_test),y_test)
    kfold_rf_mae_test = (kfold_mae_test+mae_test)
    
    mse_test = mean_squared_error(lr_classifier.predict(X_test),y_test)
    kfold_mse_test=(kfold_mse_test+mse_test)
    
    kfold_f_imp_dic = kfold_f_imp_dic + rf_classifier.feature_importances_
    
print('Logistic Regression train set mean absolute error =',kfold_mae_train/no_of_folds)
print('Logisitic Regression test set mean absolute error  =',kfold_rf_mae_test/no_of_folds)
print('Logistic Regression train set mean Squared error =',kfold_mse_train/no_of_folds)
print('logistic Regression test set mean Squared error =',kfold_mse_test/no_of_folds)
#rfc2=RandomForestClassifier()
#rfc2.fit(X_train,y_train)
#model on train using all the independent values in df
lr_prediction = lr_classifier.predict(X_train)
lr_score= accuracy_score(y_train,lr_prediction)
print('logisitic Regression Train set accuracy score ',rfc_score)
#model on test using all the indpendent values in df
lr_prediction = lr_classifier.predict(X_test)
lr_score= accuracy_score(y_test,lr_prediction)
print('logistic Regression Test Set accuracy score ',rfc_score)
import xgboost as xgb
kfold_mae_train=0
kfold_mae_test=0
kfold_mse_train=0
kfold_mse_test=0
kfold_f_imp_dic = 0

no_of_folds = 5
kf = KFold(no_of_folds,True,1)
for train_index, test_index in kf.split(X):
    
    X_train,X_test = X.iloc[train_index],X.iloc[test_index]
    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
    xgb_classifier = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05)
    
    xgb_classifier.fit(X_train,y_train)
    
    mae_train = mean_absolute_error(xgb_classifier.predict(X_train),y_train)
    kfold_mae_train=(kfold_mae_train+mae_train)
    
    mse_train = mean_squared_error(xgb_classifier.predict(X_train),y_train)
    kfold_mse_train=(kfold_mse_train+mse_train)
    
    mae_test = mean_absolute_error(xgb_classifier.predict(X_test),y_test)
    kfold_rf_mae_test = (kfold_mae_test+mae_test)
    
    mse_test = mean_squared_error(xgb_classifier.predict(X_test),y_test)
    kfold_mse_test=(kfold_mse_test+mse_test)
    
    kfold_f_imp_dic = kfold_f_imp_dic + xgb_classifier.feature_importances_
    
print('XGBoost train set mean absolute error =',kfold_mae_train/no_of_folds)
print('XGBoost test set mean absolute error  =',kfold_rf_mae_test/no_of_folds)
print('XGBoost train set mean Squared error =',kfold_mse_train/no_of_folds)
print('XGBoost test set mean Squared error =',kfold_mse_test/no_of_folds)
#rfc2=RandomForestClassifier()
#rfc2.fit(X_train,y_train)
#model on train using all the independent values in df
xgb_prediction = xgb_classifier.predict(X_train)
xgb_score= accuracy_score(y_train,xgb_prediction)
print('XGBoost Train set accuracy score ',xgb_score)
#model on test using all the indpendent values in df
xgb_prediction = xgb_classifier.predict(X_test)
xgb_score= accuracy_score(y_test,xgb_prediction)
print('XGBoost Test Set accuracy score ',xgb_score)
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
gnb =GaussianNB()
gnb.fit(X_train,y_train)
#model on train using all the independent values in df
gnb_prediction = gnb.predict(X_train)
gnb_score= accuracy_score(y_train,gnb_prediction)
print('Navie Bayes Train Set accuracy score',gnb_score)
#model on test using all the independent values in df
gnb_prediction = gnb.predict(X_test)
gnb_score= accuracy_score(y_test,gnb_prediction)
print('Navie Bayes Test Set accuracy score',gnb_score)
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
estimator = [] 
estimator.append(('LR',  
                  LogisticRegression(solver ='lbfgs',  
                                     multi_class ='multinomial',  
                                     max_iter = 200))) 
estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
estimator.append(('DTC', DecisionTreeClassifier()))

vc=VotingClassifier(estimators = estimator, voting ='hard') 
vc.fit(X_train,y_train)
#model on train using all the independent values in df
vc_prediction = vc.predict(X_train)
vc_score= accuracy_score(y_train,vc_prediction)
print('voting classifier train set accuracy score :',vc_score)
#model on test using all the independent values in df
vc_prediction = vc.predict(X_test)
vc_score= accuracy_score(y_test,vc_prediction)
print('voting classifier train set accuracy score :',vc_score)
vc=VotingClassifier(estimators = estimator, voting ='soft') 
vc.fit(X_train,y_train)
#model on train using all the independent values in df
vc_prediction = vc.predict(X_train)
vc_score= accuracy_score(y_train,vc_prediction)
print('voting classifier train set accuracy score :',vc_score)
#model on test using all the independent values in df
vc_prediction = vc.predict(X_test)
vc_score= accuracy_score(y_test,vc_prediction)
print('voting classifier train set accuracy score :',vc_score)
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

f_importance_dic = dict(zip(features,kfold_f_imp_dic/no_of_folds))
df_imp_features = pd.DataFrame(list(f_importance_dic.items()),columns=['feature','score'])

plt.figure(figsize=(30,10))
plt.bar(df_imp_features['feature'], df_imp_features['score'],color='green',align='center', alpha=0.5)
plt.xlabel('Mobile features', fontsize=20)
plt.ylabel('Relative feature score',fontsize=20)
plt.title('Relative Feature importance in determining price',fontsize=30)


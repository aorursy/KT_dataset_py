import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing,svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from collections import Counter
train=pd.read_csv('/kaggle/input/loan-eligibility/train.csv')
test=pd.read_csv('/kaggle/input/loan-eligibility/test.csv')
train.head(20)
train.info()
train.describe()
print(Counter(train['Loan_Status']))
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)
train.head()
cat_features=[feature for feature in train.columns if train[feature].dtype == 'object']
cat_features.remove('Loan_Status')
cat_features
num_features=[feature for feature in train.columns if train[feature].dtype != 'O']
num_features
con_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
discrete_features = ['Loan_Amount_Term','Credit_History']
for feature in ['ApplicantIncome','CoapplicantIncome','LoanAmount']:
    print(feature,'\nMean \n',train.groupby('Loan_Status')[feature].mean(),'\n')
    print('Median \n',train.groupby('Loan_Status')[feature].median(),'\n')
for i,feature in enumerate(cat_features):
    print(train.groupby([feature,'Loan_Status'])['Loan_Status'].count())
for i,feature in enumerate(cat_features):
    plt.figure(i)
    sns.countplot(x=feature,data=train,hue='Loan_Status')
for i,feature in enumerate(con_features):
    plt.figure(i)
    #plt.hist(train[feature])
    sns.boxplot(train[feature])
for feature in cat_features:
    train[feature]=train[feature].fillna(train[feature].mode()[0])
    test[feature]=test[feature].fillna(test[feature].mode()[0])   
for feature in num_features:
    train[feature]=train[feature].fillna(train[feature].mean())
    test[feature]=test[feature].fillna(test[feature].mean())
train.info()
#### Encoding categrical Features: ##########

dict_1 = {'Urban':3, 'Semiurban':2 , 'Rural':1}
dict_2 = {'0':0,'1':1,'2':2,'3+':3}


train['Property_Area'] = train['Property_Area'].map(dict_1)
test['Property_Area'] = test['Property_Area'].map(dict_1)

train['Dependents'] = train['Dependents'].map(dict_2)
test['Dependents'] = test['Dependents'].map(dict_2)

train['Education'] = train['Education'].map({'Graduate':1,'Not Graduate':0})
test['Education'] = test['Education'].map({'Graduate':1,'Not Graduate':0})

train = pd.get_dummies(train,drop_first=True)
test = pd.get_dummies(test,drop_first=True)

train['Total_income'] = train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_income'] = test['ApplicantIncome']+test['CoapplicantIncome']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,chi2
X=train.copy()
y=X['Loan_Status_Y']
X.drop(['Loan_Status_Y'],axis=1,inplace=True)
X.head()
train['LoanAmount'] = np.log(train['LoanAmount'])
test['LoanAmount'] = np.log(test['LoanAmount'])
X= X[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Education',
       'Total_income','Dependents','Property_Area']]
test= test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Education',
       'Total_income','Dependents','Property_Area']]
scale=preprocessing.StandardScaler()
X=scale.fit_transform(X)
test=scale.transform(test)
X_train,X_eval,y_train,y_eval=train_test_split(X,y,test_size=0.2,random_state=7,stratify =y)
svc=svm.SVC(probability=True)
svc.fit(X_train,y_train)
acc=svc.score(X_eval,y_eval)
acc1=svc.score(X_train,y_train)
print(acc1,acc)
train_preds=svc.predict(X_train)
test_preds=svc.predict(X_eval)
print('log loss\n', metrics.log_loss(y_eval,test_preds))
print("cost of training model\n",metrics.confusion_matrix(y_train,train_preds))
print("cost of testing model\n",metrics.confusion_matrix(y_eval,test_preds))
print("cost of training model\n",metrics.classification_report(y_train,train_preds))
print("cost of testing model\n",metrics.classification_report(y_eval,test_preds))
y_test1=svc.predict(test)
y_eval_probs = svc.predict_proba(X_eval)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_eval , y_eval_probs)
auc = metrics.roc_auc_score(y_eval, y_eval_probs)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
ran=RandomForestClassifier(n_estimators=25,min_samples_split=5)
ran.fit(X_train,y_train)
acc=ran.score(X_eval,y_eval)
acc1=ran.score(X_train,y_train)
print(acc1,acc)
train_preds=ran.predict(X_train)
test_preds=ran.predict(X_eval)
print('log loss\n', metrics.log_loss(y_eval,test_preds))
print("cost of training model\n",metrics.confusion_matrix(y_train,train_preds))
print("cost of testing model\n",metrics.confusion_matrix(y_eval,test_preds))
print("cost of training model\n",metrics.classification_report(y_train,train_preds))
print("cost of testing model\n",metrics.classification_report(y_eval,test_preds))
y_test1=ran.predict(test)
y_eval_probs = ran.predict_proba(X_eval)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_eval , y_eval_probs)
auc = metrics.roc_auc_score(y_eval, y_eval_probs)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
clf=LogisticRegression()
clf.fit(X_train,y_train)
eval_score=clf.score(X_eval,y_eval)
train_score=clf.score(X_train,y_train)
print('train data score\n',train_score,'\ntest data score\n',eval_score)

train_preds=clf.predict(X_train)
test_preds=clf.predict(X_eval)
print("confusion matrix of training data\n",metrics.confusion_matrix(y_train,train_preds))
print("confusion matrix of testing data\n",metrics.confusion_matrix(y_eval,test_preds))
print("classification report of training data\n",metrics.classification_report(y_train,train_preds))
print("classification report of testing data\n",metrics.classification_report(y_eval,test_preds))
y_test1=clf.predict(test)
svm_params={'C':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,25],
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
           'gamma' : ['scale','auto']}
svc=svm.SVC()
svm_tun=RandomizedSearchCV(svc,svm_params,scoring='roc_auc',cv=5)
svm_tun.fit(X_train,y_train)
print(svm_tun.best_params_)
print(svm_tun.best_score_)
svc=svm.SVC(kernel= svm_tun.best_params_['kernel'], gamma= 'auto', C= svm_tun.best_params_['C'],probability=True)
svc.fit(X_train,y_train)
acc=svc.score(X_eval,y_eval)
acc1=svc.score(X_train,y_train)
print(acc1,acc)
train_preds=svc.predict(X_train)
test_preds=svc.predict(X_eval)
print('log loss\n', metrics.log_loss(y_eval,test_preds))
print("cost of training model\n",metrics.confusion_matrix(y_train,train_preds))
print("cost of testing model\n",metrics.confusion_matrix(y_eval,test_preds))
print("cost of training model\n",metrics.classification_report(y_train,train_preds))
print("cost of testing model\n",metrics.classification_report(y_eval,test_preds))
y_test1=svc.predict(test)
y_eval_probs = svc.predict_proba(X_eval)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_eval , y_eval_probs)
auc = metrics.roc_auc_score(y_eval, y_eval_probs)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
ran_params={'n_estimators':[100,75,50,150,125,175,50,200,250,40],
        'max_depth':[15,18,20,25,30,35,40,45],
        'min_samples_split':[5,3,4,6,7,8,9,11,4,12],
           'min_samples_leaf' : [2,3,4,5,6,7,8]}
ran=RandomForestClassifier()
ran_tun=RandomizedSearchCV(ran,ran_params,scoring='roc_auc',cv=5)
ran_tun.fit(X_train,y_train)
print(ran_tun.best_params_)
print(ran_tun.best_score_)
ran=RandomForestClassifier(n_estimators=ran_tun.best_params_['n_estimators'],
                           min_samples_split= ran_tun.best_params_['min_samples_split'],
                           min_samples_leaf= ran_tun.best_params_['min_samples_leaf'],
                           max_depth=ran_tun.best_params_['max_depth'])
ran.fit(X_train,y_train)
acc=ran.score(X_eval,y_eval)
acc1=ran.score(X_train,y_train)
print(acc1,acc)
ran_train_preds=ran.predict(X_train)
ran_test_preds=ran.predict(X_eval)

print('log loss\n', metrics.log_loss(y_eval,ran_test_preds))
print("cost of training model\n",metrics.confusion_matrix(y_train,ran_train_preds))
print("cost of testing model\n",metrics.confusion_matrix(y_eval,ran_test_preds))
print("cost of training model\n",metrics.classification_report(y_train,ran_train_preds))
print("cost of testing model\n",metrics.classification_report(y_eval,ran_test_preds))
y_test1=ran.predict(test)
y_eval_probs = ran.predict_proba(X_eval)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_eval , y_eval_probs)
auc = metrics.roc_auc_score(y_eval, y_eval_probs)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
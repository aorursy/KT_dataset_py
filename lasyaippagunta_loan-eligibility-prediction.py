import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
train = pd.read_csv('../input/finance-company-loan-data/train_ctrUa4K.csv')

test = pd.read_csv('../input/finance-company-loan-data/test_lAUu6dG.csv')

train.head()
train.head()
# Join both the train and test dataset

train['source']='train'

test['source']='test'



dataset = pd.concat([train,test], ignore_index = True)

print("Train dataset shape:",train.shape)

print("Test dataset shape:",test.shape)

print("Concatenated dataset shape:",dataset.shape)
dataset.info()
dataset.isnull().sum()
print(dataset['Gender'].unique())

print(dataset['Married'].unique())

print(dataset['Dependents'].unique())

print(dataset['Self_Employed'].unique())

print(dataset['LoanAmount'].unique())

print(dataset['Loan_Amount_Term'].unique())

print(dataset['Credit_History'].unique())
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)

dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)

dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)

dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)

dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].median(), inplace=True)

dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
dataset.isnull().sum()
dataset.info()
print(len(dataset['Gender'].unique()))

print(len(dataset['Married'].unique()))

print(len(dataset['Dependents'].unique()))

print(len(dataset['Self_Employed'].unique()))

print(len(dataset['LoanAmount'].unique()))

print(len(dataset['Loan_Amount_Term'].unique()))

print(len(dataset['Credit_History'].unique()))

print(len(dataset['Loan_ID'].unique()))

print(len(dataset['Education'].unique()))

print(len(dataset['ApplicantIncome'].unique()))

print(len(dataset['CoapplicantIncome'].unique()))

print(len(dataset['Property_Area'].unique()))

print(len(dataset['source'].unique()))
#Divide into test and train:

train = dataset.loc[dataset['source']=="train"]

test = dataset.loc[dataset['source']=="test"]

#Drop unnecessary columns:

test.drop(['source'],axis=1,inplace=True)

train.drop(['source'],axis=1,inplace=True)
train.head()
plt.title('Loan Status Bar Plot')

plt.xlabel('Loan Status Y - Yes or N- No')

plt.ylabel('Loan Status Count')



train['Loan_Status'].value_counts().plot.bar(color=['green', 'red'],edgecolor='blue')
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

train['Gender'].value_counts(normalize=True).plot.bar(title='Gender')

plt.subplot(2,2,2)

train['Married'].value_counts(normalize=True).plot.bar(title='Married')

plt.subplot(2,2,3)

train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self Employed')

plt.subplot(2,2,4)

train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit History')
fig, ax = plt.subplots(2,4,figsize = (15,15))

Gender = pd.crosstab(train['Gender'],train['Loan_Status'])

Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=ax[0,0])



Married = pd.crosstab(train['Married'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[0,1])



Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])

Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[0,2])



Education = pd.crosstab(train['Education'],train['Loan_Status'])

Education.div(Education.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,ax=ax[0,3])



Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[1,0])



Credit_History = pd.crosstab(train['Credit_History'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True, ax=ax[1,1])



Property_Area = pd.crosstab(train['Property_Area'],train['Loan_Status'])

Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,ax=ax[1,2])
#cols=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed','Property_Area']

#for label in cols:

#    dataset[label]=LabelEncoder().fit_transform(dataset[label])

#dataset.head()
X=train.drop(["Loan_Status",'Loan_ID'],axis=1)

y=train["Loan_Status"]



X = pd.get_dummies(X,drop_first=True)

X.head()
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
logistic_Regression = LogisticRegression(max_iter=1000,random_state=0)

logistic_Regression.fit(x_train,y_train)
y_pred = logistic_Regression.predict(x_test)
log = accuracy_score(y_pred,y_test)*100
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))
knn = KNeighborsClassifier(n_neighbors=200)

knn.fit(x_train,y_train)
pred_knn = knn.predict(x_test)
KNN = accuracy_score(pred_knn,y_test)*100
print(confusion_matrix(pred_knn,y_test))
print(classification_report(pred_knn,y_test))
error=[]

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred1=knn.predict(x_test)

    error.append(np.mean(pred1!=y_test))

print(error)
plt.figure(figsize=(10,6))

plt.plot(range(1,50),error,color='blue',linestyle='dashed',marker = 'o',markerfacecolor='red',markersize=10)

plt.title('Error rate vs K value')

plt.xlabel('k')

plt.ylabel('error rate')
gnb=GaussianNB()

gnb.fit(x_train,y_train)
pred_gnb = gnb.predict(x_test)
GNB = accuracy_score(pred_gnb,y_test)*100
print(confusion_matrix(pred_gnb,y_test))
print(classification_report(pred_gnb,y_test))
svc = SVC()

svc.fit(x_train,y_train)

pred_svc = svc.predict(x_test)
SVC = accuracy_score(pred_svc,y_test)*100
print(confusion_matrix(pred_svc,y_test))
print(classification_report(pred_svc,y_test))
dtree_en = DecisionTreeClassifier(criterion='entropy',splitter='random',max_leaf_nodes=5,min_samples_leaf=10,max_depth=3)
clf = dtree_en.fit(x_train,y_train)
pred_dt = clf.predict(x_test)
DTREE = accuracy_score(pred_dt,y_test)*100
cm=confusion_matrix(y_test,pred_dt)

print(cm)

print(classification_report(y_test,pred_dt))
dtree = DecisionTreeClassifier(criterion='gini',splitter='random',max_leaf_nodes=5,min_samples_leaf=10,max_depth=5)

dtree.fit(x_train,y_train)
pred_g = dtree.predict(x_test)
DTREE_G = accuracy_score(y_test,pred_g)*100
cm=confusion_matrix(y_test,pred_g)

print(cm)

print(classification_report(y_test,pred_g))
rfc = RandomForestClassifier(criterion='entropy',n_estimators=400)

rfc.fit(x_train, y_train)
pred_rf= rfc.predict(x_test)
RFC = accuracy_score(y_test,pred_rf)*100

RFC
print(confusion_matrix(pred_rf,y_test))
print(classification_report(pred_rf,y_test))
model = DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=0)

adaboost = AdaBoostClassifier(n_estimators=80, base_estimator=model,random_state=0)

adaboost.fit(x_train,y_train)
pred = adaboost.predict(x_test)
ada = accuracy_score(y_test,pred)*100
model_g = DecisionTreeClassifier(criterion='gini',max_depth=1,random_state=0)

adaboost1 = AdaBoostClassifier(n_estimators=90, base_estimator=model_g,random_state=0)

adaboost1.fit(x_train,y_train)
pred_gini = adaboost.predict(x_test)
g = accuracy_score(y_test,pred_gini)*100
xgb =  XGBClassifier(learning_rate =0.000001,n_estimators=1000,max_depth=5,min_child_weight=1,subsample=0.8,colsample_bytree=0.8,nthread=4,scale_pos_weight=1,seed=27)
xgb.fit(x_train, y_train)

predxg = xgb.predict(x_test)

xg = accuracy_score(y_test,predxg)*100
print("1)  Logistic Regression    :",log)

print("2)  AdaBoost - Entropy     :",ada)

print("3)  AdaBoost - Gini        :",g)

print("4)  XGBoost                :",xg)

print("5)  Decision Tree - Entropy:",DTREE)

print("6)  Decision Tree - Gini   :",DTREE_G)

print("7)  Random Forest          :",RFC)

print("8)  Naive-Bayes            :",GNB)

print("9)  KNN                    :",KNN)

print("10) SVC                    :",SVC)
test.head()
Xt = test.drop(["Loan_Status","Loan_ID"],axis=1)

Xt = pd.get_dummies(Xt,drop_first=True)



Xt.head()
test_pred = logistic_Regression.predict(Xt)
test["Loan_Status"] = test_pred
test.head()
submission = test[["Loan_ID","Loan_Status"]].copy()
submission.to_csv('testLR.csv')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# To find if there are any null values in given dataset
for i in df.select_dtypes(exclude='object'):
    if(df[i].isnull().sum()>0):
        print(i)
pd.options.display.max_columns=None
df.drop('customerID',axis=1,inplace=True) # Dropping customerID, as it wont be of much use to use
df.shape
df.select_dtypes(include='object').head(2)
# This copy is only for visualization of data before encoding!
df1=df.copy()
# MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,
# StreamingTV,StreamingMovies,Contract [multiclass features]

df['gender']=df['gender'].replace({'Male':1,'Female':0})
df['Partner']=df['Partner'].replace({'Yes':1,'No':0})
df['Dependents']=df['Dependents'].replace({'Yes':1,'No':0})
df['PhoneService']=df['PhoneService'].replace({'Yes':1,'No':0})
df['PaperlessBilling']=df['PaperlessBilling'].replace({'Yes':1,'No':0})
df['Churn']=df['Churn'].replace({'Yes':1,'No':0})
################################ Problems ####################################
#   1-We have to convert all categorical variables by using manual label encoding.
#   2-fix the datatype of TotalCharges which is numerical feature but it appears to be object datatype.
#   3- Replace null values by respective mean/median.

# for i in df.select_dtypes(include='object'):
#     if df[i].nunique()>2:
#         print(i,':\n',df[i].value_counts())
# Fixing data type of totalcharges to float and Imputing missing values with median.
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())

##### using get_dummies to encode multiclass features
dfcopy=pd.get_dummies(df.select_dtypes(include='object'),drop_first=True) #creating dummies with 3 categories.
df=pd.concat([df,dfcopy],axis=1)
df.select_dtypes(include='object').columns
df.drop(['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaymentMethod'],axis=1,inplace=True)
# df is clean dataframe
df1.head()
for i in df1.select_dtypes(include='object'):
    sns.countplot(df1[i])
    plt.show()
# plt.figure(figsize=(15,18))
# for i in range(1, len(df1.select_dtypes(include=object).columns)-1):
#     plt.subplot(10, 2, i)
#     sns.countplot(df1[df1.columns[i]])
# plt.show()

## As we can observe from given data,
## 1-There are equal number of males and females,equal number of people with and without partner in the given data.
## 2-There are around roughly 2000 people who are dependents
## 3-Almost 90% of people do have access to phone service.But,there are a few people who dont have access to phone service.
## 4-There are people who have access to landline internet n/w as well mobile n/w
## 5-People with Fiber optic,DSL access are more.
## 6-Its very surprising that majority of people from the sample dont have online security.
## 7-Again majority of people also dont onlinebackup
## 8-But,Its good to see that people care enough about their devices so as to protect them.
df1.select_dtypes(exclude='object').columns
# convert tenure to years!
df1['tenure']=np.round((df1['tenure']/12),1)
df1.head(1)
df1['TotalCharges']=pd.to_numeric(df1['TotalCharges'],errors='coerce')
fig,axes=plt.subplots(1,3,figsize=(15,5))
sns.distplot(df1['tenure'],ax=axes[0])
sns.distplot(df1['MonthlyCharges'],ax=axes[1])
sns.distplot(df1['TotalCharges'],ax=axes[2])
plt.show()
#So,majority of people stick around with their operator from 0 to 6 years.
#Monthly charges range around from 20 to 120 dollars.
#Total charges incurred by customers are around 0 to 8000 dollars per year.
sns.scatterplot(df1['MonthlyCharges'],df1['TotalCharges'],hue=df1['Churn'])
plt.show()
# People who are churning to other operator seem mostly below 4000 only.There are a few people with monthly charges
# and with total charges above 5000,But,the frequency to churn is low.
df1['Churn']=df1['Churn'].replace({'Yes':1,'No':0})
df1['Churn']=df1['Churn'].astype('int')
sns.heatmap(df1.corr(),annot=True)
# Total charges,
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# from sklearn.preprocessing import LabelEncoder,OneHotEncoder #this is optional

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from scipy.stats import zscore

from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix


import warnings
warnings.filterwarnings('ignore')
X=df.drop('Churn',axis=1)
y=df['Churn']

lr = LogisticRegression()
gb = GaussianNB()
models = []
models.append(('LogisticRegression',lr))
models.append(('NaviveBayes',gb))
results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True, n_splits=5, random_state=0)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
    results.append(1 - cv_results)
    names.append(name)
    print('%s : %f(%f)' %(name,np.mean(cv_results), np.var(cv_results,ddof=1)))
# boxplot algorithm comparision
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#Splitting X&y using train_test:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=8)

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_prob_train = logreg.predict_proba(X_train)
y_pred_train = logreg.predict(X_train)
y_prob_test = logreg.predict_proba(X_test)
y_pred_test = logreg.predict(X_test)

print('Confusion Matrix - Train:', '\n', confusion_matrix(y_train, y_pred_train))
print('Overall Accuracy', accuracy_score(y_train, y_pred_train))

print('Confusion Matrix - Test:', '\n', confusion_matrix(y_test, y_pred_test))
print('Overall Accuracy', accuracy_score(y_test, y_pred_test))
      
from sklearn.metrics import log_loss
print('log loss: ',log_loss(y_test,y_prob_test))
X=df.drop('Churn',axis=1)
y=df['Churn']
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
knn = KNeighborsClassifier()
knn_params = {'n_neighbors':np.arange(3,20), 'weights':['uniform','distance']}
gscv = GridSearchCV(knn, knn_params, cv=5, scoring='roc_auc')
gscv.fit(X_scaled, y)
print(gscv.best_params_)
gscv_best_knn=gscv.best_params_

KNN=KNeighborsClassifier(**gscv_best_knn)

KNN.fit(X_scaled,y)
KNN.score(X_scaled,y)
#Splitting X&y using train_test:
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.2,random_state=8)
y_prob_train = KNN.predict_proba(X_train)
y_pred_train = KNN.predict(X_train)
y_prob_test = KNN.predict_proba(X_test)
y_pred_test = KNN.predict(X_test)

print('Confusion Matrix - Train:', '\n', confusion_matrix(y_train, y_pred_train))
print('Overall Accuracy', accuracy_score(y_train, y_pred_train))

print('Confusion Matrix - Test:', '\n', confusion_matrix(y_test, y_pred_test))
print('Overall Accuracy', accuracy_score(y_test, y_pred_test))
      
from sklearn.metrics import log_loss
print('log loss: ',log_loss(y_test,y_prob_test))
X=df.drop('Churn',axis=1)
y=df['Churn']
dt = DecisionTreeClassifier()
dt_params = {'max_depth':np.arange(1,10), 'min_samples_leaf':np.arange(2,100), 'criterion':['entropy','gini']}
gscv = GridSearchCV(dt, dt_params, cv=5, scoring='roc_auc')
gscv.fit(X, y)
print(gscv.best_params_)
gscv_best_DT=gscv.best_params_
DT=DecisionTreeClassifier(**gscv_best_DT)
DT.fit(X,y)
DT.score(X,y)
#Splitting X&y using train_test:
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.2,random_state=8)


DT.fit(X_train,y_train)
y_prob_train = DT.predict_proba(X_train)
y_pred_train = DT.predict(X_train)
y_prob_test = DT.predict_proba(X_test)
y_pred_test = DT.predict(X_test)

print('Confusion Matrix - Train:', '\n', confusion_matrix(y_train, y_pred_train))
print('Overall Accuracy', accuracy_score(y_train, y_pred_train))

print('Confusion Matrix - Test:', '\n', confusion_matrix(y_test, y_pred_test))
print('Overall Accuracy', accuracy_score(y_test, y_pred_test))
      
from sklearn.metrics import log_loss
print('log loss: ',log_loss(y_test,y_prob_test))
X=df.drop('Churn',axis=1)
y=df['Churn']
auc_avg = []
auc_var = []
for ne in np.arange(1,30):
    RF=RandomForestClassifier(n_estimators=ne,random_state=0)
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    auc = cross_val_score(RF, X, y, cv=kfold, scoring='roc_auc')
    auc_avg.append(1 - np.mean(auc))
    auc_var.append(np.var(auc,ddof=1))
print('Min Bias Error:',np.min(auc_avg),' n_estimator:',np.argmin(auc_avg)+1,' Variance Error:',auc_var[np.argmin(auc_avg)])
print('Bias Error:',auc_avg[np.argmin(auc_var)],' n_estimator:',np.argmin(auc_var)+1,'Min Variance Error:',np.min(auc_var))
#Splitting X&y using train_test:
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.2,random_state=8)
RF=RandomForestClassifier(n_estimators=16)
RF.fit(X,y)
RF.score(X,y)
RF.fit(X_train,y_train)
y_prob_train = RF.predict_proba(X_train)
y_pred_train = RF.predict(X_train)
y_prob_test = RF.predict_proba(X_test)
y_pred_test = RF.predict(X_test)

print('Confusion Matrix - Train:', '\n', confusion_matrix(y_train, y_pred_train))
print('Overall Accuracy', accuracy_score(y_train, y_pred_train))

print('Confusion Matrix - Test:', '\n', confusion_matrix(y_test, y_pred_test))
print('Overall Accuracy', accuracy_score(y_test, y_pred_test))
      
from sklearn.metrics import log_loss
print('log loss: ',log_loss(y_test,y_prob_test))
X=df.drop('Churn',axis=1)
y=df['Churn']
lr = LogisticRegression()
gb = GaussianNB()
knn=KNeighborsClassifier(**gscv_best_knn)
dt = DecisionTreeClassifier(**gscv_best_DT)
rf=RandomForestClassifier(n_estimators=17,random_state=0)
models = []
models.append(('LogisticRegression',lr))
models.append(('NaiveBayes',gb))
models.append(('KNeighborsClassifier',knn))
models.append(('DecisionTreeClassifier',dt))
models.append(('RandomForestClassifier',rf))
results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True, n_splits=5, random_state=0)
    cv_results = cross_val_score(model, X_scaled, y, cv=kfold, scoring='roc_auc')
    results.append(1-cv_results)
    names.append(name)
    print('%s : %f(%f)' %(name,1 - np.mean(cv_results), np.var(cv_results,ddof=1)))
# boxplot algorithm comparision
fig = plt.figure(figsize=(15,5))
fig.suptitle('Algorithm Comparision')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
X=df.drop('Churn',axis=1)
y=df['Churn']

auc_avg = []
auc_var = []
for ne in np.arange(1,20):
    ab_rf = AdaBoostClassifier(base_estimator=rf,n_estimators= ne,random_state=0)
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    auc = cross_val_score(ab_rf, X, y, cv=kfold, scoring='roc_auc')
    auc_avg.append(1 - np.mean(auc))
    auc_var.append(np.var(auc,ddof=1))

print('Min Bias Error:',np.min(auc_avg),' n_estimator:',np.argmin(auc_avg)+1,' Variance Error:',auc_var[np.argmin(auc_avg)])
print('Bias Error:',auc_avg[np.argmin(auc_var)],' n_estimator:',np.argmin(auc_var)+1,'Min Variance Error:',np.min(auc_var))
# From ada boosting the random forest,we can observe that,bias reduced slightly and variance was already reduced
# because of random forest itself
X=df.drop('Churn',axis=1)
y=df['Churn']

auc_avg = []
auc_var = []
for ne in np.arange(1,30):
    ab_dt = AdaBoostClassifier(n_estimators= ne,random_state=0)
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    auc = cross_val_score(ab_dt, X, y, cv=kfold, scoring='roc_auc')
    auc_avg.append(1 - np.mean(auc))
    auc_var.append(np.var(auc,ddof=1))

print('Min Bias Error:',np.min(auc_avg),' n_estimator:',np.argmin(auc_avg)+1,' Variance Error:',auc_var[np.argmin(auc_avg)])
print('Bias Error:',auc_avg[np.argmin(auc_var)],' n_estimator:',np.argmin(auc_var)+1,'Min Variance Error:',np.min(auc_var))
# Again,here as well,bias error of decision tree was reduced slightly.
X=df.drop('Churn',axis=1)
y=df['Churn']
auc_avg = []
auc_var = []
for ne in np.arange(1,30):
    ab_nb = AdaBoostClassifier(base_estimator=gb,n_estimators=ne, random_state=0)
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    auc = cross_val_score(ab_nb, X_scaled, y, cv=kfold, scoring='roc_auc')
    auc_avg.append(1 - np.mean(auc))
    auc_var.append(np.var(auc,ddof=1))
print('Min Bias Error:',np.min(auc_avg),' n_estimator:',np.argmin(auc_avg)+1,' Variance Error:',auc_var[np.argmin(auc_avg)])
print('Bias Error:',auc_avg[np.argmin(auc_var)],' n_estimator:',np.argmin(auc_var)+1,'Min Variance Error:',np.min(auc_var))

X=df.drop('Churn',axis=1)
y=df['Churn']
auc_avg = []
auc_var = []
for ne in np.arange(1,30):
    ab_lr = AdaBoostClassifier(base_estimator=lr,n_estimators=ne, random_state=0)
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    auc = cross_val_score(ab_lr, X_scaled, y, cv=kfold, scoring='roc_auc')
    auc_avg.append(1 - np.mean(auc))
    auc_var.append(np.var(auc,ddof=1))
print('Min Bias Error:',np.min(auc_avg),' n_estimator:',np.argmin(auc_avg)+1,' Variance Error:',auc_var[np.argmin(auc_avg)])
print('Bias Error:',auc_avg[np.argmin(auc_var)],' n_estimator:',np.argmin(auc_var)+1,'Min Variance Error:',np.min(auc_var))
X=df.drop('Churn',axis=1)
y=df['Churn']
auc_avg = []
auc_var = []
for ne in np.arange(1,30):
    bgcl_lr = BaggingClassifier(base_estimator=lr, random_state=0, n_estimators=ne)
    kfold = KFold(shuffle=True,n_splits=5,random_state=0)
    auc = cross_val_score(bgcl_lr, X_scaled, y, cv=kfold, scoring='roc_auc')
    auc_avg.append(1 - np.mean(auc))
    auc_var.append(np.var(auc,ddof=1))
print('Min Bias Error:',np.min(auc_avg),' n_estimator:',np.argmin(auc_avg)+1,' Variance Error:',auc_var[np.argmin(auc_avg)])
print('Bias Error:',auc_avg[np.argmin(auc_var)],' n_estimator:',np.argmin(auc_var)+1,'Min Variance Error:',np.min(auc_var))
lr = LogisticRegression()
gb = GaussianNB()
knn=KNeighborsClassifier(**gscv_best_knn)
dt = DecisionTreeClassifier(**gscv_best_DT)
rf=RandomForestClassifier(n_estimators=17,random_state=0)

ab_rf = AdaBoostClassifier(base_estimator=rf,n_estimators=2,random_state=0)
ab_dt = AdaBoostClassifier(base_estimator=dt,n_estimators=21,random_state=0)
ab_nb=  AdaBoostClassifier(base_estimator=gb,n_estimators=3,random_state=0)
ab_lr=  AdaBoostClassifier(base_estimator=lr,n_estimators=29,random_state=0)
bgcl_lr = BaggingClassifier(base_estimator=lr, random_state=0, n_estimators=17)


#gbcl = GradientBoostingClassifier(random_state=0, n_estimators=27)
#stacked = VotingClassifier(estimators=[('BoostedDT',ab_dt),('BaggedLR',bgcl_lr)], voting='soft')
models = []
models.append(('LogisticRegression',lr))
models.append(('NaiveBayes',gb))
models.append(('KNeighborsClassifier',knn))
models.append(('DecisionTreeClassifier   ',dt))
models.append(('RandomForestClassifier',rf))
models.append(('BoostedRF',ab_rf))
models.append(('BoostedDT',ab_dt))
models.append(('BoostedNB',ab_nb))
models.append(('BoostedLR',ab_lr))
models.append(('BaggedLR',bgcl_lr))

#models.append(('GBoostClassifier',gbcl))
#models.append(('VotingClassifier',stacked))
results = []
names = []
for name, model in models:
    kfold = KFold(shuffle=True, n_splits=5, random_state=0)
    cv_results = cross_val_score(model, X_scaled, y, cv=kfold, scoring='roc_auc')
    results.append(1 - cv_results)
    names.append(name)
    print('%s : %f(%f)' %(name,1 - np.mean(cv_results), np.var(cv_results,ddof=1)))
# boxplot algorithm comparision
fig = plt.figure(figsize=(20,5))
fig.suptitle('Algorithm Comparision')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
X=df.drop('Churn',axis=1)
y=df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'liblinear')

logreg.fit(X_train, y_train)



y_pred_train = logreg.predict(X_train) # actual prediction of train 
y_pred_test = logreg.predict(X_test) # actual prediction of test
y_test_pred_new=logreg.predict_proba(X_test)
y_test_pred_new=y_test_pred_new[:,1]



print('Confusion Matrix - Train:', '\n', confusion_matrix(y_train, y_pred_train))
print('Overall Accuracy', accuracy_score(y_train, y_pred_train))

print('Confusion Matrix - Test:', '\n', confusion_matrix(y_test, y_pred_test))
print('Overall Accuracy', accuracy_score(y_test, y_pred_test))


# print('auc score train: ',roc_auc_score(X_train,y_prob_train_pro))
print('auc score test: ',roc_auc_score(y_test,y_test_pred_new))

from sklearn.metrics import log_loss
print('log loss: ',log_loss(y_test,y_test_pred_new))

#set seed for same results everytime
seed=0
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics



X=df.drop('Churn',axis=1)
y=df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =3)

#declare the models
lr = LogisticRegression()
rf=RandomForestClassifier(n_estimators=17,random_state=0)
adb=ensemble.AdaBoostClassifier()
bgc=ensemble.BaggingClassifier()
gnb = GaussianNB()
knn=KNeighborsClassifier(**gscv_best_knn)
dt = DecisionTreeClassifier(**gscv_best_DT)
ab_rf = AdaBoostClassifier(base_estimator=rf,n_estimators=2,random_state=0)
ab_dt = AdaBoostClassifier(base_estimator=dt,n_estimators=21,random_state=0)
ab_nb=  AdaBoostClassifier(base_estimator=gnb,random_state=0)
ab_lr=  AdaBoostClassifier(base_estimator=lr,n_estimators=29,random_state=0)
bgcl_lr = BaggingClassifier(base_estimator=lr, random_state=0, n_estimators=17)
xgb = XGBClassifier()

models=[lr,rf,adb,bgc,gnb,knn,dt,ab_rf,ab_dt,ab_nb,ab_lr,bgcl_lr,xgb]
sctr,scte,auc,ps,rs=[],[],[],[],[]
def ens(X_train,X_test, y_train, y_test):
    for model in models:
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_test_pred_new=model.predict_proba(X_test)
            y_test_pred_new=y_test_pred_new[:,1]
            train_score=model.score(X_train,y_train)
            test_score=model.score(X_test,y_test)
            p_score=metrics.precision_score(y_test,y_test_pred)
            r_score=metrics.recall_score(y_test,y_test_pred)
            
            ac=metrics.roc_auc_score(y_test,y_test_pred_new)
            
            sctr.append(train_score)
            scte.append(test_score)
            ps.append(p_score)
            rs.append(r_score)
            auc.append(ac)
    return sctr,scte,auc,ps,rs
ens(X_train,X_test, y_train, y_test)

ensemble=pd.DataFrame({'names':['Logistic Regression','Random Forest','Ada boost','Bagging',
                                'Naive-Bayes','KNN','Decistion Tree','ab_rf','ab_dt','ab_nb','ab_lr','bgcl_lr','XGB'],
                       'auc_score':auc,'training':sctr,'testing':scte,'precision':ps,'recall':rs})
ensemble=ensemble.sort_values(by=['auc_score','precision','recall'],ascending=False).reset_index(drop=True)
ensemble


from imblearn.over_sampling import SMOTE

smote_X=df.drop('Churn',axis=1)
smote_Y=df['Churn']
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =3)

# smote_X = telcom[cols]
# smote_Y = telcom[target_col]

#Split train and test data
smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,
                                                                         test_size = .25 ,
                                                                         random_state = 111)

#oversampling minority class using smote
os = SMOTE(random_state = 0)
os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)
os_smote_X = pd.DataFrame(data = os_smote_X,columns=smote_X.columns)
os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=['Churn'])
#set seed for same results everytime
seed=0
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics



X=os_smote_X
y=os_smote_Y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =3)

#declare the models
lr = LogisticRegression()
rf=RandomForestClassifier(n_estimators=17,random_state=0)
adb=ensemble.AdaBoostClassifier()
bgc=ensemble.BaggingClassifier()
gnb = GaussianNB()
knn=KNeighborsClassifier(**gscv_best_knn)
dt = DecisionTreeClassifier(**gscv_best_DT)
ab_rf = AdaBoostClassifier(base_estimator=rf,n_estimators=2,random_state=0)
ab_dt = AdaBoostClassifier(base_estimator=dt,n_estimators=21,random_state=0)
ab_nb=  AdaBoostClassifier(base_estimator=gnb,random_state=0)
ab_lr=  AdaBoostClassifier(base_estimator=lr,n_estimators=29,random_state=0)
bgcl_lr = BaggingClassifier(base_estimator=lr, random_state=0, n_estimators=17)
xgb = XGBClassifier()

models=[lr,rf,adb,bgc,gnb,knn,dt,ab_rf,ab_dt,ab_nb,ab_lr,bgcl_lr,xgb]
sctr,scte,auc,ps,rs=[],[],[],[],[]
def ens(X_train,X_test, y_train, y_test):
    for model in models:
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_test_pred_new=model.predict_proba(X_test)
            y_test_pred_new=y_test_pred_new[:,1]
            train_score=model.score(X_train,y_train)
            test_score=model.score(X_test,y_test)
            p_score=metrics.precision_score(y_test,y_test_pred)
            r_score=metrics.recall_score(y_test,y_test_pred)
            
            ac=metrics.roc_auc_score(y_test,y_test_pred_new)
            
            sctr.append(train_score)
            scte.append(test_score)
            ps.append(p_score)
            rs.append(r_score)
            auc.append(ac)
    return sctr,scte,auc,ps,rs
ens(X_train,X_test, y_train, y_test)

ensemble=pd.DataFrame({'names':['Logistic Regression','Random Forest','Ada boost','Bagging',
                                'Naive-Bayes','KNN','Decistion Tree','ab_rf','ab_dt','ab_nb','ab_lr','bgcl_lr','XGB'],
                       'auc_score':auc,'training':sctr,'testing':scte,'precision':ps,'recall':rs})
ensemble=ensemble.sort_values(by=['auc_score','precision','recall'],ascending=False).reset_index(drop=True)
ensemble
X=os_smote_X
y=os_smote_Y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'liblinear')

logreg.fit(X_train, y_train)


y_pred_train = logreg.predict(X_train) # actual prediction of train 
y_pred_test = logreg.predict(X_test) # actual prediction of test
y_test_pred_new=logreg.predict_proba(X_test)
y_test_pred_new=y_test_pred_new[:,1]



print('Confusion Matrix - Train:', '\n', confusion_matrix(y_train, y_pred_train))
print('Overall Accuracy', accuracy_score(y_train, y_pred_train))

print('Confusion Matrix - Test:', '\n', confusion_matrix(y_test, y_pred_test))
print('Overall Accuracy', accuracy_score(y_test, y_pred_test))


# print('auc score train: ',roc_auc_score(X_train,y_prob_train_pro))
print('auc score test: ',roc_auc_score(y_test,y_test_pred_new))
print('classification report:\n',classification_report(y_test,y_pred_test))
from sklearn.metrics import log_loss
print('log loss: ',log_loss(y_test,y_test_pred_new))
print('Before SMOTE:\n')
X=df.drop('Churn',axis=1)
y=df['Churn']
print('X shape: ',X.shape)
print('y shape: ',y.shape)
print('Target variable distribution before smote:\n',y.value_counts())

print('\n\n\n')

print('After SMOTE:\n')
X=os_smote_X
y=os_smote_Y
print('X shape: ',X.shape)
print('y shape: ',y.shape)
print('Target variable distribution after smote:\n',y['Churn'].value_counts())
# As we can observe from the result of our function precision and recall along with auc_roc score improved drastically 
# when we applied smote to our data to make it more balanced.
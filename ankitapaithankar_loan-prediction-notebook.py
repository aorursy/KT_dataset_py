import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestRegressor,BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import sklearn.metrics as metrics
from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix
### read datasets
train=pd.read_csv("../input/loan-prediction/train_ctrUa4K.csv")
test=pd.read_csv("../input/loan-prediction/test_lAUu6dG.csv")

test_cpy=test.copy()
round((test.isnull().sum()/len(test.index))*100,2)
train.head()
test.head()
train.shape
test.shape
train.info()
train.describe()
train['Loan_Status'].value_counts().plot.bar()
# Independent Variable (Categorical)

plt.figure(1) 
plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()
# It can be inferred from the above bar plots that:

#     80% applicants in the dataset are male.
#     Around 65% of the applicants in the dataset are married.
#     Around 15% applicants in the dataset are self employed.
#     Around 85% applicants have repaid their debts.
plt.figure(1) 
plt.subplot(131) 
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 
plt.subplot(132) 
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(133) 
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
plt.show()
# Following inferences can be made from the above bar plots:

#     Most of the applicants don’t have any dependents.
#     Around 80% of the applicants are Graduate.
#     Most of the applicants are from Semiurban area.
plt.figure(1) 
plt.subplot(121) 
sns.distplot(train['ApplicantIncome']); 
plt.subplot(122) 
train['ApplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()
train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")
plt.figure(1) 
plt.subplot(121) 
sns.distplot(train['CoapplicantIncome']); 
plt.subplot(122) 
train['CoapplicantIncome'].plot.box(figsize=(16,5)) 
plt.show()
# Let’s look at the distribution of LoanAmount variable.

train['LoanAmount'].plot.box(figsize=(16,5)) 
Gender=pd.crosstab(train['Gender'],train['Loan_Status'],normalize=True) 
Gender.plot(kind="bar", stacked=True, figsize=(4,4))

# It can be inferred that the proportion of male and female applicants 
# is more or less same for both approved and unapproved loans.
Married=pd.crosstab(train['Married'],train['Loan_Status'],normalize=True) 
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'],normalize=True) 
Education=pd.crosstab(train['Education'],train['Loan_Status'],normalize=True) 
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'],normalize=True) 

Married.plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Dependents.plot(kind="bar", stacked=True) 
plt.show() 
Education.plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Self_Employed.plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show()


#  Proportion of married applicants is higher for the approved loans.
#  Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.
#  There is nothing significant we can infer from Self_Employed vs Loan_Status plot.
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'],normalize=True) 
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'],normalize=True) 
Credit_History.plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Property_Area.plot(kind="bar", stacked=True) 
plt.show()

#  It seems people with credit history as 1 are more likely to get their loans approved.
#  Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'],normalize=True) 
Income_bin.plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')
bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'],normalize=True) 
Coapplicant_Income_bin.plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')


# It shows that if coapplicant’s income is less the chances of loan approval are high. 
# But this does not look right. The possible reason behind this may be that most of the applicants don’t 
# have any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not 
# dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s 
# income to visualize the combined effect of income on loan approval.
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']

bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'],normalize=True) 
Total_Income_bin.plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


# It can be seen that the proportion of approved loans is higher for Low and Average 
# Loan Amount as compared to that of High Loan Amount which supports our hypothesis in which we considered that 
# the chances of loan approval will be high when the loan amount is less.
bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'],normalize=True) 
LoanAmount_bin.plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')
train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
matrix = train.corr() 
f, ax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20) 

test['LoanAmount_log'] = np.log(test['LoanAmount'])
train['Gender'].value_counts()
train['Married'].value_counts()
train['Self_Employed'].value_counts()
train['Loan_Status'].value_counts()
train['Dependents'].value_counts()
train['Dependents']=train['Dependents'].fillna(0)
train['Dependents'].replace({'3+':3},inplace=True)

test['Dependents']=test['Dependents'].fillna(0)
test['Dependents'].replace({'3+':3},inplace=True)
train['Gender']=train['Gender'].map({'Male':1, 'Female': 0})
test['Gender']=test['Gender'].map({'Male':1, 'Female': 0})
train['Gender']=train['Gender'].fillna(1).astype('int')
test['Gender']=test['Gender'].fillna(1).astype('int')

train['Self_Employed']=train['Self_Employed'].map({'Yes':1, 'No': 0})
test['Self_Employed']=test['Self_Employed'].map({'Yes':1, 'No': 0})
train['Self_Employed']=train['Self_Employed'].fillna(0).astype(int)
test['Self_Employed']=test['Self_Employed'].fillna(0).astype(int)

train['Married']=train['Married'].map({'Yes':1, 'No': 0})
test['Married']=test['Married'].map({'Yes':1, 'No': 0})
train['Married']=train['Married'].fillna(1).astype('int')


train['Education']=train['Education'].map({'Graduate':1, 'Not Graduate': 0})
test['Education']=test['Education'].map({'Graduate':1, 'Not Graduate': 0})

train['Loan_Status']=train['Loan_Status'].map({'Y':1, 'N': 0})
train['Dependents']=train['Dependents'].astype(int)
test['Dependents']=test['Dependents'].astype(int)
# train_pa=pd.get_dummies(train['Property_Area'],drop_first=True)
# test_pa=pd.get_dummies(test['Property_Area'],drop_first=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

train['Property_Area']=le.fit_transform(train['Property_Area'])
test['Property_Area']=le.fit_transform(test['Property_Area'])
train.drop(['Loan_ID'],axis=1,inplace=True)
test.drop(['Loan_ID'],axis=1,inplace=True)
train.head()
round((train.isnull().sum()/len(train.index))*100,2)
test.head()
test.isnull().sum()
sns.boxplot(x=train['ApplicantIncome'])
sns.boxplot(x=train['CoapplicantIncome'])
train.Loan_Status.value_counts()
train['TotalIncome']=train['ApplicantIncome']+train['CoapplicantIncome']
test['TotalIncome']=test['ApplicantIncome']+test['CoapplicantIncome']

train.head()
train['Total_Income_log'] = np.log(train['TotalIncome']) 
sns.distplot(train['Total_Income_log']); 
test['Total_Income_log'] = np.log(test['TotalIncome'])
train.drop(['ApplicantIncome','CoapplicantIncome','TotalIncome','LoanAmount'],axis=1,inplace=True)
test.drop(['ApplicantIncome','CoapplicantIncome','TotalIncome','LoanAmount'],axis=1,inplace=True)
train.head()
train['LoanAmount_log']=train['LoanAmount_log'].fillna(train['LoanAmount_log'].median())
train['Loan_Amount_Term']=train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean())
train['Credit_History']=train['Credit_History'].fillna(train['Credit_History'].mean())

test['LoanAmount_log']=test['LoanAmount_log'].fillna(test['LoanAmount_log'].median())
test['Loan_Amount_Term']=test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean())
test['Credit_History']=test['Credit_History'].fillna(test['Credit_History'].mean())
### new variable EMI

train['EMI']=train['LoanAmount_log']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount_log']/test['Loan_Amount_Term']
# Normalising continuous features
df = train[['EMI','Total_Income_log']]
df_test = test[['EMI','Total_Income_log']]

df.head()
normalized_df=(df-df.mean())/df.std()
normalized_df_test=(df_test-df_test.mean())/df_test.std()
normalized_df.head()
train=train.drop(['EMI','Total_Income_log','LoanAmount_log','Loan_Amount_Term'],axis=1)
# train_out2=train_out2.drop(['LoanAmount','Loan_Amount_Term'],axis=1)
df=pd.concat([normalized_df,train],axis=1)

df.head(10)

test=test.drop(['EMI','Total_Income_log','LoanAmount_log','Loan_Amount_Term'],axis=1)

test_df=pd.concat([normalized_df_test,test],axis=1)

test_df.head()
round((test_df.isnull().sum()/len(test_df.index))*100,2)
#set seed for same results everytime
seed=0
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics

X=df.drop('Loan_Status',1)
y=df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =1)

#declare the models
lr = LogisticRegression()
rf=RandomForestClassifier()
adb=ensemble.AdaBoostClassifier()
bgc=ensemble.BaggingClassifier()
gnb = GaussianNB()
knn=KNeighborsClassifier()
dt = DecisionTreeClassifier()
bgcl_lr = BaggingClassifier(base_estimator=lr, random_state=0)

# ,ab_rf,ab_dt,ab_nb,ab_lr,bgcl_lr

models=[lr,rf,adb,bgc,gnb,knn,dt,bgcl_lr]
sctr,scte,auc,ps,rs,acc=[],[],[],[],[],[]
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
            accr=metrics.accuracy_score(y_test,y_test_pred)
            ac=metrics.roc_auc_score(y_test,y_test_pred_new)
            
            sctr.append(train_score)
            scte.append(test_score)
            ps.append(p_score)
            rs.append(r_score)
            auc.append(ac)
            acc.append(accr)
    return sctr,scte,auc,ps,rs,acc

ens(X_train,X_test, y_train, y_test)
# 'ab_rf','ab_dt','ab_nb','ab_lr','bgcl_lr'
ensemble=pd.DataFrame({'names':['Logistic Regression','Random Forest','Ada boost','Bagging',
                                'Naive-Bayes','KNN','Decistion Tree',
                                'bagged LR'],
                       'auc_score':auc,'training':sctr,'testing':scte,'precision':ps,'recall':rs,'accuracy':acc})
ensemble=ensemble.sort_values(by='auc_score',ascending=False).reset_index(drop=True)
ensemble
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()


rf.fit(X,y)

y_test_pred_rf=rf.predict(test_df)
finalpred=pd.concat([test_cpy['Loan_ID'],pd.DataFrame(y_test_pred_rf,columns=['Loan_Status'])],1)
finalpred.to_csv("sub.csv",index=False)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

dt = DecisionTreeClassifier()
dt_params = {'max_depth':np.arange(1,10), 'min_samples_leaf':np.arange(2,30), 'criterion':['entropy','gini']}
rscv = RandomizedSearchCV(dt, dt_params, cv=5, scoring='roc_auc')
rscv.fit(X, y)
print(rscv.best_params_)
rscv_best_DT=rscv.best_params_

DT=DecisionTreeClassifier(**rscv_best_DT)
DT.fit(X,y)
y_test_pred_DT=DT.predict(test_df)
finalpred=pd.concat([test_cpy['Loan_ID'],pd.DataFrame(y_test_pred_DT,columns=['Loan_Status'])],1)
finalpred.to_csv("pred.csv",index=False)
import xgboost as xgb 
from xgboost.sklearn import XGBClassifier
xgb=XGBClassifier(learning_rate=0.09,n_estimators=125,max_depth=4,min_child_weight=4,colsample_bytree=0.5,reg_alpha=0.000001 )
xgb.fit(X,y)
y_test_pred_xgb=xgb.predict(test_df)
finalpred=pd.concat([test_cpy['Loan_ID'],pd.DataFrame(y_test_pred_DT,columns=['Loan_Status'])],1)
finalpred.to_csv("xgb.csv",index=False)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,          
                      intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,          
                      penalty='l2', random_state=1, solver='liblinear', tol=0.0001,          
                      verbose=0, warm_start=False)

lr.fit(X,y)


y_test_pred_lr=lr.predict(test_df)
finalpred=pd.concat([test_cpy['Loan_ID'],pd.DataFrame(y_test_pred_lr,columns=['Loan_Status'])],1)
finalpred.to_csv("LR_1.csv",index=False)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
print("Before OverSampling, counts of label '1': {}".format(sum(y == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0))) 

from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X, y.ravel()) 
  
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,          
                      intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,          
                      penalty='l2', random_state=1, solver='liblinear', tol=0.0001,          
                      verbose=0, warm_start=False)


lr.fit(X_train_res,y_train_res)

y_test_pred_lr=lr.predict(test_df)
finalpred=pd.concat([test_cpy['Loan_ID'],pd.DataFrame(y_test_pred_lr,columns=['Loan_Status'])],1)

finalpred.to_csv("LR_2.csv",index=False)
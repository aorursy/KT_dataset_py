import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
#ignore warning
import warnings
warnings.filterwarnings("ignore")
#read data
pdd=pd.read_csv('../input/logistic-regression-coupon-used-prediction/.csv')
pdd.head()
#data info
pdd.info()
#missing value check
pd.DataFrame(pdd.isnull().sum()).reset_index().rename(columns={'index':'field_name',0:'missing number'})
#Numerical Variable Outlier analysis
pdd.describe().drop(['ID','coupon_ind'],axis=1)
sum(pdd.duplicated())
#see the proportion of the counpon_ind
pdd.coupon_ind.value_counts(1).reset_index().rename(columns={'index':'Category','coupon_ind':'Percentage'}).set_index('Category')
pdd.groupby('coupon_ind').mean().drop('ID',axis=1)
#Customer age.
plt.figure(figsize=(8,6))
sns.distplot(pdd['age'],bins=20)
plt.tick_params(labelsize=16)
plt.xlabel('age',size=18)
plt.figure(figsize=(8,6))
pdd['age'].plot(kind='box')
plt.tick_params(labelsize=16)
plt.xlabel('age',size=18)
#Number of coupons used by users in the past six months
plt.figure(figsize=(8,6))
pdd['coupon_used_in_last6_month'].plot(kind='box')
plt.tick_params(labelsize=16)
plt.xlabel('Coupon used in last six months',size=18)
plt.figure(figsize=(8,6))
pdd['coupon_used_in_last6_month'].plot(kind='hist',bins=50,density=True)
plt.tick_params(labelsize=16)
plt.xlabel('Coupon used in last six months',size=18)
plt.ylabel('frequency',size=18)
plt.figure(figsize=(8,6))
pdd['coupon_used_in_last_month'].plot(kind='hist',bins=10,density=True)
plt.tick_params(labelsize=16)
plt.xlabel('Coupon used in last month',size=18)
plt.ylabel('frequency',size=18)
#Job distribution
plt.figure(figsize=(8,6))
sns.countplot(x='job',data=pdd,hue='coupon_ind')
plt.xticks(rotation=45)
plt.tick_params(labelsize=16)
plt.xlabel('Job',size=18)
plt.ylabel('Count',size=18)
plt.legend(fontsize=16)
pdd.groupby('coupon_ind')['job'].value_counts()
plt.figure(figsize=(20,10))
temp=pd.DataFrame(pdd.groupby('job')['coupon_ind'].value_counts(1))
temp=temp.rename(columns={'coupon_ind':'rate'}).reset_index()
tmp_order=temp[temp['coupon_ind']==1].sort_values(by='rate')['job'].tolist()
sns.barplot(x='job',y='rate',hue='coupon_ind',data=temp,order=tmp_order)
#Marriage condition
plt.figure(figsize=(8,6))
sns.countplot(x='marital',data=pdd,hue='coupon_ind')
plt.xticks(rotation=45)
plt.tick_params(labelsize=16)
plt.xlabel('Marriage Condition',size=18)
plt.ylabel('counts',size=18)
plt.legend(fontsize=16)
plt.figure(figsize=(5,8))
temp=pd.DataFrame(pdd.groupby('marital')['coupon_ind'].value_counts(1))
temp=temp.rename(columns={'coupon_ind':'rate'}).reset_index()
tmp_order=temp[temp['coupon_ind']==1].sort_values(by='rate')['marital'].tolist()
sns.barplot(x='marital',y='rate',hue='coupon_ind',data=temp,order=tmp_order)
#Credit Card Condition
plt.figure(figsize=(8,6))
sns.countplot(x='default',data=pdd,hue='coupon_ind')
plt.xticks(rotation=45)
plt.tick_params(labelsize=16)
plt.xlabel('Credit Card Condition',size=18)
plt.ylabel('counts',size=18)
plt.legend(fontsize=16)
plt.figure(figsize=(5,8))
temp=pd.DataFrame(pdd.groupby('default')['coupon_ind'].value_counts(1))
temp=temp.rename(columns={'coupon_ind':'rate'}).reset_index()
tmp_order=temp[temp['coupon_ind']==1].sort_values(by='rate')['default'].tolist()
sns.barplot(x='default',y='rate',hue='coupon_ind',data=temp,order=tmp_order)
#Return condition
plt.figure(figsize=(8,6))
sns.countplot(x='returned',data=pdd,hue='coupon_ind')
plt.xticks(rotation=45)
plt.tick_params(labelsize=16)
plt.xlabel('Return Condition',size=18)
plt.ylabel('counts',size=18)
plt.legend(fontsize=16)
plt.figure(figsize=(5,8))
temp=pd.DataFrame(pdd.groupby('returned')['coupon_ind'].value_counts(1))
temp=temp.rename(columns={'coupon_ind':'rate'}).reset_index()
tmp_order=temp[temp['coupon_ind']==1].sort_values(by='rate')['returned'].tolist()
sns.barplot(x='returned',y='rate',hue='coupon_ind',data=temp,order=tmp_order)
#Loan condition
plt.figure(figsize=(8,6))
sns.countplot(x='loan',data=pdd,hue='coupon_ind')
plt.xticks(rotation=45)
plt.tick_params(labelsize=16)
plt.xlabel('Loan Condition',size=18)
plt.ylabel('counts',size=18)
plt.legend(fontsize=16)
plt.figure(figsize=(5,8))
temp=pd.DataFrame(pdd.groupby('loan')['coupon_ind'].value_counts(1))
temp=temp.rename(columns={'coupon_ind':'rate'}).reset_index()
tmp_order=temp[temp['coupon_ind']==1].sort_values(by='rate')['loan'].tolist()
sns.barplot(x='loan',y='rate',hue='coupon_ind',data=temp,order=tmp_order)
#copy date for backups
pdd_clean=pdd.copy()
#drop id
pdd_clean.drop('ID',axis=1,inplace=True)
pdd_clean=pdd_clean.rename(columns={'coupon_ind':'flag'})
#Age segementation
bins=[0,25,45,65,100]
labels=['<25','25-45','45-65','>65']
pdd_clean['age_range']=pd.cut(pdd_clean['age'],bins,right=False,labels=labels)
#Coupon used last 6 months segementation
bins=[0,10,100]
labels=['<10','>=10']
pdd_clean['coupon_last6_range']=pd.cut(pdd_clean['coupon_used_in_last6_month'],bins,right=False,labels=labels)
#get dummies for all category variable
pdd_clean=pd.get_dummies(pdd_clean,drop_first=True)
pdd_clean.head()
pdd_clean.corr()[['flag']].sort_values('flag',ascending=False)
plt.figure(figsize=(8,6))
sns.heatmap(pdd_clean[pdd_clean.corr()[['flag']].sort_values('flag',ascending=True).index].corr(),cmap="Greys")
plt.tick_params(labelsize=16)
#Save the precision, recall, f1, AUC of each scheme to facilitate subsequent visualization
names=[]
scores_train=[]
scores_test=[]
precisions=[]
recalls=[]
f1=[]
auc=[]
#Define independent variables and extract variables with correlation coefficient greater than +/-0.05
#Adjustable threshold during optimization

threshold=0.05
tmp=pdd_clean.corr()[['flag']].sort_values('flag',ascending=False)
var=tmp.where(np.abs(tmp)>threshold).dropna(how='any').index.values
var=var[1:]
x=pdd_clean[var]
y=pdd_clean['flag']
print(var)
x.head()
#Split the data set
#Test_size can be adjusted during optimization
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

#Training model
LR= LogisticRegression()
LR.fit(X_train,y_train)

#Model parameters
for a,b in zip(np.append(var,'intercept'), np.append(LR.coef_,LR.intercept_)):
   
    print(f'parameter{a}:{b}, Probability ratio:{np.exp(b)}')
#Model evaluation
print("\n Original model:")
print("\n")
print(f'Training set scoring:{LR.score(X_train,y_train)}')
print(f'Testing set scoring:{LR.score(X_test,y_test)}')

y_pred =LR.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['flag=0','flag=1']))
      
#ROC/AUC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr,marker='o')
plt.show()


names.append('initial')
scores_train.append(LR.score(X_train,y_train))
scores_test.append(LR.score(X_test,y_test))
precisions.append(metrics.precision_score(y_test,y_pred)) #flag=1
recalls.append(metrics.recall_score(y_test,y_pred))#flag=1
f1.append(metrics.f1_score(y_test,y_pred))#flag=1
auc.append(metrics.auc(fpr,tpr))
#1.Data standardization

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)



LR= LogisticRegression()
LR.fit(X_train,y_train)

for a,b in zip(np.append(var,'intercept'), np.append(LR.coef_,LR.intercept_)):
    print(f'parameter{a}:{b}, Probability ratio:{np.exp(b)}')

print("\nData standardization:")
print("\n")

print(f'Training set scoring:{LR.score(X_train,y_train)}')
print(f'Testing set scoring:{LR.score(X_test,y_test)}')

y_pred =LR.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['flag=0','flag=1']))
      
#ROC/AUC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr,marker='o')
plt.show()


names.append('initial')
scores_train.append(LR.score(X_train,y_train))
scores_test.append(LR.score(X_test,y_test))
precisions.append(metrics.precision_score(y_test,y_pred)) #flag=1
recalls.append(metrics.recall_score(y_test,y_pred))#flag=1
f1.append(metrics.f1_score(y_test,y_pred))#flag=1
auc.append(metrics.auc(fpr,tpr))

#2.SMOTE upsampling

from imblearn.over_sampling import SMOTE
smo=SMOTE(random_state=100)
X_smo,y_smo=smo.fit_sample(x,y)

X_train,X_test,y_train,y_test=train_test_split(X_smo,y_smo,test_size=0.3,random_state=100)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


LR= LogisticRegression()
LR.fit(X_train,y_train)


for a,b in zip(np.append(var,'intercept'), np.append(LR.coef_,LR.intercept_)):
     print(f'parameter{a}:{b}, Probability ratio:{np.exp(b)}')

print("\nSMOTE:")
print("\n")
print(f'Training set scoring:{LR.score(X_train,y_train)}')
print(f'Testing set scoring:{LR.score(X_test,y_test)}')

y_pred =LR.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['flag=0','flag=1']))
      
#ROC/AUC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr,marker='o')
plt.show()


names.append('initial')
scores_train.append(LR.score(X_train,y_train))
scores_test.append(LR.score(X_test,y_test))
precisions.append(metrics.precision_score(y_test,y_pred)) #flag=1
recalls.append(metrics.recall_score(y_test,y_pred))#flag=1
f1.append(metrics.f1_score(y_test,y_pred))#flag=1
auc.append(metrics.auc(fpr,tpr))

#2.RandomOverSampler Upsampling
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=100)
X_ros,y_ros=smo.fit_sample(x,y)

X_train,X_test,y_train,y_test=train_test_split(X_ros,y_ros,test_size=0.3,random_state=100)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


LR= LogisticRegression()
LR.fit(X_train,y_train)


for a,b in zip(np.append(var,'intercept'), np.append(LR.coef_,LR.intercept_)):
    print(f'parameter{a}:{b}, Probability ratio:{np.exp(b)}')

print("\nRandomOverSampler:")
print("\n")
print(f'Training set scoring:{LR.score(X_train,y_train)}')
print(f'Testing set scoring:{LR.score(X_test,y_test)}')

y_pred =LR.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['flag=0','flag=1']))
      
#ROC/AUC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr,marker='o')
plt.show()


names.append('initial')
scores_train.append(LR.score(X_train,y_train))
scores_test.append(LR.score(X_test,y_test))
precisions.append(metrics.precision_score(y_test,y_pred)) #flag=1
recalls.append(metrics.recall_score(y_test,y_pred))#flag=1
f1.append(metrics.f1_score(y_test,y_pred))#flag=1
auc.append(metrics.auc(fpr,tpr))
#2.RandomUnderSampler Upsampling
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=100)
X_rus,y_rus=smo.fit_sample(x,y)

X_train,X_test,y_train,y_test=train_test_split(X_rus,y_rus,test_size=0.3,random_state=100)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


LR= LogisticRegression()
LR.fit(X_train,y_train)

for a,b in zip(np.append(var,'intercept'), np.append(LR.coef_,LR.intercept_)):
    print(f'parameter{a}:{b}, Probability ratio:{np.exp(b)}')

print("\nRandomUnderSampler:")
print("\n")
print(f'Training set scoring:{LR.score(X_train,y_train)}')
print(f'Testing set scoring:{LR.score(X_test,y_test)}')

y_pred =LR.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['flag=0','flag=1']))
      
#ROC/AUC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr,marker='o')
plt.show()


names.append('initial')
scores_train.append(LR.score(X_train,y_train))
scores_test.append(LR.score(X_test,y_test))
precisions.append(metrics.precision_score(y_test,y_pred)) #flag=1
recalls.append(metrics.recall_score(y_test,y_pred))#flag=1
f1.append(metrics.f1_score(y_test,y_pred))#flag=1
auc.append(metrics.auc(fpr,tpr))
#2.NearMiss downsampling

from imblearn.under_sampling import NearMiss
nml=NearMiss(version=2) 
#Select the sample with the shortest average distance from the N furthest samples among the positive samples
X_nml,y_nml=nml.fit_sample(x,y)

X_train,X_test,y_train,y_test=train_test_split(X_nml,y_nml,test_size=0.3,random_state=100)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


LR= LogisticRegression()
LR.fit(X_train,y_train)


for a,b in zip(np.append(var,'intercept'), np.append(LR.coef_,LR.intercept_)):
     print(f'parameter{a}:{b}, Probability ratio:{np.exp(b)}')
#模型评估
print("\nNearMiss:")
print("\n")
print(f'Training set scoring:{LR.score(X_train,y_train)}')
print(f'Testing set scoring:{LR.score(X_test,y_test)}')

y_pred =LR.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['flag=0','flag=1']))
      
#ROC/AUC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr,marker='o')
plt.show()



names.append('initial')
scores_train.append(LR.score(X_train,y_train))
scores_test.append(LR.score(X_test,y_test))
precisions.append(metrics.precision_score(y_test,y_pred)) #flag=1
recalls.append(metrics.recall_score(y_test,y_pred))#flag=1
f1.append(metrics.f1_score(y_test,y_pred))#flag=1
auc.append(metrics.auc(fpr,tpr))
#2.SMOTEENN down sampling

from imblearn.combine import SMOTEENN
smote_enn=SMOTEENN(random_state=100) 
X_smote_enn,y_smote_enn=smote_enn.fit_sample(x,y)

X_train,X_test,y_train,y_test=train_test_split(X_smote_enn,y_smote_enn,test_size=0.3,random_state=100)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


LR= LogisticRegression()
LR.fit(X_train,y_train)

for a,b in zip(np.append(var,'intercept'), np.append(LR.coef_,LR.intercept_)):
    print(f'parameter{a}:{b}, Probability ratio:{np.exp(b)}')

print("\nNearMiss:")
print("\n")
print(f'Training set scoring:{LR.score(X_train,y_train)}')
print(f'Testing set scoring:{LR.score(X_test,y_test)}')

y_pred =LR.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['flag=0','flag=1']))
      
#ROC/AUC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_pred,pos_label=1)
plt.plot(fpr,tpr,marker='o')
plt.show()



names.append('initial')
scores_train.append(LR.score(X_train,y_train))
scores_test.append(LR.score(X_test,y_test))
precisions.append(metrics.precision_score(y_test,y_pred)) #flag=1
recalls.append(metrics.recall_score(y_test,y_pred))#flag=1
f1.append(metrics.f1_score(y_test,y_pred))#flag=1
auc.append(metrics.auc(fpr,tpr))
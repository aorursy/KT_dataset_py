import pandas as pd
import numpy as np
import seaborn as sns
import random
import math
!pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import SGDClassifier,LogisticRegression,LinearRegression,SGDRegressor
from scipy.stats import norm 
!pip install missingno
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
import missingno as msno
import plotly.express as px
from scipy.stats import boxcox
from imblearn.under_sampling import NearMiss
from sklearn.metrics import auc,roc_curve,roc_auc_score,confusion_matrix
from tqdm import notebook as tqdm
from sklearn.metrics import classification_report,accuracy_score,log_loss
data=pd.read_csv("../input/adultcsv/adult.csv")
data.head()
msno.bar(data)
data.isna().sum()
data.dropna(inplace=True)
data['target'].replace({' <=50K\n':0,' >50K\n':1},inplace=True)

data.drop(columns=['Unnamed: 0','fnlwgt','education-num'],inplace=True)
data.columns

#workclass
print(data.workclass.value_counts())
#  Without-pay             8
#  Never-worked            5


#education
print(data.education.value_counts())

#  5th-6th          243
#  1st-4th          117
#  Preschool         33

#marital-status
print(data['marital-status'].value_counts())

#  Married-AF-spouse           16

#occupation
print(data.occupation.value_counts())

# ?                    1355
# Armed-Forces            7
#relationship
print(data.relationship.value_counts())
#race
print(data.race.value_counts())
#sex
print(data.sex.value_counts())
#native country
print(data.native_country.value_counts())

#  Trinadad&Tobago                  11
#  Hong                             11
#  Yugoslavia                       11
#  Hungary                          10
#  Scotland                          9
#  Honduras                          8
#  Outlying-US(Guam-USVI-etc)        8
#  Laos                              8
#  Holand-Netherlands                1
data=data[data['workclass']!=' Without-pay']
data=data[data['workclass']!=' Never-worked']
data.workclass.replace({' ?':' Others'},inplace=True)
print('Workclass')
print(data['workclass'].value_counts())


data=data[data['education']!=' 5th-6th']
data=data[data['education']!=' 1st-4th']
data=data[data['education']!=' Preschool']
data['education'].replace({' Some-college':' Bachelors'},inplace=True)
data['education'].replace({' 11th':' School',' 10th':' School',' 7th-8th':' School',' 9th':' School',' 12th':' School'},inplace=True)
print(data.education.value_counts())


data=data[data['marital-status']!= ' Married-AF-spouse']
print(data['marital-status'].value_counts())

data['occupation'].replace({' ?':' Other-service'},inplace=True)
data=data[data.occupation!=' Armed-Forces']
print(data.occupation.value_counts())



# Trinadad&Tobago                  11
#  Hong                             11
#  Yugoslavia                       11
#  Hungary                          10
#  Scotland                          9
#  Honduras                          8
#  Outlying-US(Guam-USVI-etc)        8
#  Laos                              8
#  Holand-Netherlands                1

data=data[data['native_country']!=' Trinadad&Tobago']

data=data[data['native_country']!=' Hong']

data=data[data['native_country']!=' Yugoslavia']
data=data[data['native_country']!=' Hungary']

data=data[data['native_country']!=' Scotland']

data=data[data['native_country']!=' Honduras']

data=data[data['native_country']!=' Outlying-US(Guam-USVI-etc)']

data=data[data['native_country']!=' Laos']
data=data[data['native_country']!=' Holand-Netherlands']
data['native_country'].replace({' ?':' Others'},inplace=True)
print(data.native_country.value_counts())
#age column analysis
plt.figure(num='data',figsize=(18,12),dpi=200,frameon=False,edgecolor='blue')
plt.subplot(2,3,1)
sns.distplot(data.age,kde=True,fit=norm)
plt.subplot(2,3,2)
sns.scatterplot(data=data.age)
plt.subplot(2,3,3)
sns.boxplot(x='target',y='age',data=data,fliersize=10,color='green',hue='target')
plt.subplot(2,3,4)
sns.boxplot(x='target',y='age',data=data,color='green',hue='target',showfliers = False)
plt.subplot(2,3,5)
sns.violinplot(y='age',x='target',data=data)
plt.show()
fig=px.box(data,x='age',color='target')
fig.show()
#workclass column analysis
print(data.workclass.describe())
print()
print(data.workclass.value_counts())
print()

plt.figure(1,figsize=(20,9))
plt.subplot(2,2,1)
plt.xticks(rotation=45)
sns.countplot(data.workclass)

plt.subplot(2,2,2)
plt.xticks(rotation=45)
g=sns.countplot(data.workclass,hue=data.target)


plt.show()
plt.figure(2,figsize=(12,5))
sns.catplot(hue='workclass',x='target',data=data,height=8,kind='count')
plt.show()
plt.figure(3,figsize=(12,5))

sns.catplot(x='workclass',y='target',data=data,height=8,kind='point',)

plt.show()
plt.figure(4,figsize=(12,5))
sns.violinplot(data=pd.get_dummies(data.workclass))
plt.show()


plt.figure(5,figsize=(14,6))
plt.subplot(1,2,1)
sns.boxenplot(y='age',data=data,x='target',)
plt.subplot(1,2,2)
sns.boxenplot(x='workclass',y='age',data=data,hue='target')
plt.show()

print(pd.crosstab(data.workclass,data.target))
#education analysis
print(data.education.head())
print(data.education.value_counts())
plt.figure(1,figsize=(12,8))
sns.countplot(data.education,hue=data.target)
plt.show()

plt.figure(2,figsize=(12,8))
sns.violinplot(data=pd.get_dummies(data.education))
plt.show()

plt.figure(3,figsize=(24,10))
plt.subplot(2,1,1)
sns.violinplot(data=data,y='target',hue='workclass',x='education')
plt.subplot(2,1,2)
sns.boxenplot(x='education',y='age',data=data,hue='target')
plt.show()

print(pd.crosstab(data.education,data.target))
#marital status
# data.columns
print(data['marital-status'].value_counts())
def replace(X):
    if X in range(20):
        return 'adult'
    if X in range(20,50):
        return 'mature'
    if X in range(50,100):
        return 'old'
plt.figure(1,figsize=(20,12))
plt.subplot(3,2,1)
sns.countplot(data['marital-status'])
plt.subplot(3,2,2)
sns.countplot(data['marital-status'],hue=data.target)

plt.subplot(3,2,3)

sns.countplot(data['marital-status'],hue=data.workclass)
plt.subplot(3,2,4)
sns.countplot(data.age.map(lambda x:replace(x)),hue=data['education'])
plt.subplot(3,2,5)
sns.countplot(data.age.map(lambda x:replace(x)),hue=data['marital-status'])
plt.subplot(3,2,6)
sns.countplot(data.age.map(lambda x:replace(x)),hue=data['workclass'])
# ages=
plt.show()
# data.age.replace({[i for i in range(20)]:'adult',[i for i in range(20,50)]:'mature',[i for i in range(50,100)]:'old_age'})
# sns.countplot(data.age.map(lambda x:replace(x)),hue=data['workclass'])

#occupation

# print(data.occupation)
print(data.occupation.value_counts())
plt.figure(1,figsize=(20,16))
plt.subplot(3,2,1)
plt.xticks(rotation=20)
sns.countplot(data.occupation)
plt.subplot(3,2,2)
plt.xticks(rotation=20)
sns.countplot(data.occupation,hue=data.target)
plt.subplot(3,2,3)
plt.xticks(rotation=20)
sns.countplot(data.occupation,hue=data.age.map(lambda x: replace(x)))
plt.subplot(3,2,4)
plt.xticks(rotation=20)
sns.countplot(data.occupation,hue=data['marital-status'])
plt.subplot(3,2,5)

sns.countplot(data.occupation,hue=data['workclass'])
plt.xticks(rotation=30)
plt.subplot(3,2,5)

plt.xticks(rotation=30)
plt.show()
# relationship analysis
# print(data.relationship)
print(data.relationship.value_counts())
plt.figure(1,figsize=(20,16))
plt.subplot(3,2,1)
plt.xticks(rotation=20)
sns.countplot(data.relationship)
plt.subplot(3,2,2)
plt.xticks(rotation=20)
sns.countplot(data.relationship,hue=data.target)
plt.subplot(3,2,3)
plt.xticks(rotation=20)
sns.countplot(data.relationship,hue=data.age.map(lambda x: replace(x)))
plt.subplot(3,2,4)
plt.xticks(rotation=20)
sns.countplot(data.relationship,hue=data['marital-status'])
plt.subplot(3,2,5)

sns.countplot(data.relationship,hue=data['workclass'])
plt.xticks(rotation=30)
plt.subplot(3,2,6)
sns.countplot(data.relationship,hue=data['education'])
plt.xticks(rotation=30)
plt.show()
# race analysis
print(data.race.value_counts())
plt.figure(1,figsize=(20,16))
plt.subplot(4,2,1)
plt.xticks(rotation=20)
sns.countplot(data.race)
plt.subplot(4,2,2)
plt.xticks(rotation=20)
sns.countplot(data.race,hue=data.target)
plt.subplot(4,2,3)
plt.xticks(rotation=20)
sns.countplot(data.race,hue=data.age.map(lambda x: replace(x)))
plt.subplot(4,2,4)
plt.xticks(rotation=20)
sns.countplot(data.race,hue=data['marital-status'])
plt.subplot(4,2,5)

sns.countplot(data.race,hue=data['workclass'])
plt.xticks(rotation=30)
plt.subplot(4,2,6)
sns.countplot(data.race,hue=data['education'])
plt.xticks(rotation=30)
plt.subplot(4,2,7)
sns.countplot(data.race,hue=data['relationship'])
plt.xticks(rotation=30)
plt.show()
# sex analysis
print(data.sex.value_counts())
plt.figure(1,figsize=(20,16))
plt.subplot(4,2,1)
plt.xticks(rotation=20)
sns.countplot(data.sex)
plt.subplot(4,2,2)
plt.xticks(rotation=20)
sns.countplot(data.sex,hue=data.target)
plt.subplot(4,2,3)
plt.xticks(rotation=20)
sns.countplot(data.sex,hue=data.age.map(lambda x: replace(x)))
plt.subplot(4,2,4)
plt.xticks(rotation=20)
sns.countplot(data.sex,hue=data['marital-status'])
plt.subplot(4,2,5)

sns.countplot(data.sex,hue=data['workclass'])
plt.xticks(rotation=30)
plt.subplot(4,2,6)
sns.countplot(data.sex,hue=data['education'])
plt.xticks(rotation=30)
plt.subplot(4,2,7)
sns.countplot(data.sex,hue=data['relationship'])
plt.xticks(rotation=30)
plt.subplot(4,2,8)
sns.countplot(data.sex,hue=data['race'])
plt.xticks(rotation=30)
plt.show()
data['capital-gain'].value_counts()
def return_cal(x):
    if x in range(100):
        return 'hundreds'
    if x in range(100,50000):
        return '50_thousands<'
    if x in range(50000,100000):
        return '50_thousands>'
    if x in range(100000,5000000):
        return '50_lakhs<'
    if x in range(5000000,10000000):
        return '50_lakhs>'
    if x >10000000:
        return 'crores'

sns.distplot(data['capital-gain'].value_counts(),fit=norm)
print(data['capital-gain'].describe())
fig=px.histogram(data,x='capital-gain',color='target')
fig.show()
fig=px.histogram(data,x='capital-gain',color='target',range_x=[0,1000])
fig.show()
fig=px.box(data,x='capital-gain')
fig.show()
fig=px.box(data,x='capital-gain',color='target',range_x=[0,7000])
fig.show()
# capital-gain analysis
data['modified_age']=data.age.map(lambda x : replace(x))
print(data['capital-gain'].describe())
# plt.figure(1,figsize=(20,15))
# plt.subplot(5,2,1)
# sns.scatterplot(data=data['capital-gain'])
# plt.xticks(rotation=20)
# plt.subplot(5,2,2)
# sns.distplot(data['capital-gain'],fit=norm)
# plt.xticks(rotation=20)
# plt.subplot(5,2,3)
# sns.boxplot(x='target',y='capital-gain',data=data)
# plt.xticks(rotation=20)
# plt.subplot(5,2,4)
# sns.boxplot(x='target',y='capital-gain',data=data,showfliers=False)
# plt.xticks(rotation=20)
# plt.subplot(5,2,5)
# sns.boxenplot(x='target',y='capital-gain',data=data,showfliers=True)
# plt.xticks(rotation=20)
# plt.subplot(5,2,6)
# sns.violinplot(x='target',y='capital-gain',data=data,showfliers=True)
# plt.xticks(rotation=20)
# plt.subplot(5,2,7)
# sns.countplot(data['target'],hue=data['capital-gain'].map(lambda x : return_cal(x)),)
# plt.xticks(rotation=20)

# plt.show()


fig = px.scatter(data, x="modified_age", y="capital-gain",color='target',size='age')
fig.show()

fig = px.scatter(data, x="education", y="capital-gain",color='target')
fig.show()

fig = px.scatter(data, x="marital-status", y="capital-gain",color='target')
fig.show()

fig = px.scatter(data, x="marital-status", y="capital-gain",color='sex')
fig.show()

fig = px.scatter(data, x="race", y="capital-gain",color='target')
fig.show()

fig=px.box(data,y='capital-gain',x='modified_age',color='target',points='suspectedoutliers')
fig.show()

fig=px.box(data,y='capital-gain',x='modified_age',color='target',points='suspectedoutliers',range_y=[0,40000])
fig.show()

data['modified_cp']=data['capital-gain'].map(lambda x : return_cal(x))
fig=px.box(data,x='modified_cp',y='target')
fig.show()

data['modified_cp']=data['capital-gain'].map(lambda x : return_cal(x))
fig=px.box(data,x='modified_cp',y='target',color='sex')


# capital-gain analysis
data['modified_age']=data.age.map(lambda x : replace(x))
print(data['capital-loss'].describe())
# plt.figure(1,figsize=(20,15))
# plt.subplot(5,2,1)
# sns.scatterplot(data=data['capital-loss'])
# plt.xticks(rotation=20)
# plt.subplot(5,2,2)
# sns.distplot(data['capital-loss'],kde=True,fit=norm)
# plt.xticks(rotation=20)
# plt.subplot(5,2,3)
# sns.boxplot(x='target',y='capital-loss',data=data)
# plt.xticks(rotation=20)
# plt.subplot(5,2,4)
# sns.boxplot(x='target',y='capital-loss',data=data,showfliers=False)
# plt.xticks(rotation=20)
# plt.subplot(5,2,5)
# sns.boxenplot(x='target',y='capital-loss',data=data,showfliers=True)
# plt.xticks(rotation=20)
# plt.subplot(5,2,6)
# sns.violinplot(x='target',y='capital-loss',data=data,showfliers=True)
# plt.xticks(rotation=20)
# plt.subplot(5,2,7)
# sns.countplot(data['target'],hue=data['capital-loss'].map(lambda x : return_cal(x)),)
# plt.xticks(rotation=20)

# plt.show()


fig = px.scatter(data, x="modified_age", y="capital-loss",color='target',size='age')
fig.show()

fig = px.scatter(data, x="education", y="capital-loss",color='target')
fig.show()

fig = px.scatter(data, x="marital-status", y="capital-loss",color='target')
fig.show()

fig = px.scatter(data, x="marital-status", y="capital-loss",color='sex')
fig.show()

fig = px.scatter(data, x="race", y="capital-loss",color='target')
fig.show()

fig=px.box(data,y='capital-loss',x='modified_age',color='target',points='suspectedoutliers')
fig.show()

fig=px.box(data,y='capital-loss',x='modified_age',color='target',points='suspectedoutliers',range_y=[0,40000])
fig.show()

data['modified_cp']=data['capital-loss'].map(lambda x : return_cal(x))
fig=px.box(data,x='modified_cp',y='target')
fig.show()

data['modified_cp']=data['capital-loss'].map(lambda x : return_cal(x))
fig=px.box(data,y='capital-loss',x='target',color='sex')
fig.show()

fig=px.box(data,x='capital-gain',)
fig.show()
# hours-per-week analysis
data['modified_age']=data.age.map(lambda x : replace(x))
print(data['hours-per-week'].describe())
# plt.figure(1,figsize=(20,15))
# plt.subplot(5,2,1)
# sns.scatterplot(data=data['hours-per-week'])
# plt.xticks(rotation=20)
# plt.subplot(5,2,2)
# sns.distplot(data['hours-per-week'],kde=True,fit=norm)
# plt.xticks(rotation=20)
# plt.subplot(5,2,3)
# sns.boxplot(x='target',y='hours-per-week',data=data)
# plt.xticks(rotation=20)
# plt.subplot(5,2,4)
# sns.boxplot(x='target',y='hours-per-week',data=data,showfliers=False)
# plt.xticks(rotation=20)
# plt.subplot(5,2,5)
# sns.boxenplot(x='target',y='hours-per-week',data=data,showfliers=True)
# plt.xticks(rotation=20)
# plt.subplot(5,2,6)
# sns.violinplot(x='target',y='hours-per-week',data=data,showfliers=True)
# plt.xticks(rotation=20)
# plt.subplot(5,2,7)
# sns.countplot(data['hours-per-week'],hue=data['target'])
# plt.xticks(rotation=20)

# plt.show()


# plt.subplot(2,1,1)
fig = px.scatter(data, x="modified_age", y="hours-per-week",color='target',size='age')
fig.show()

fig = px.scatter(data, x="education", y="hours-per-week",color='target')
fig.show()

fig = px.scatter(data, x="marital-status", y="hours-per-week",color='target')
fig.show()

fig = px.scatter(data, x="marital-status", y="hours-per-week",color='sex')
fig.show()

fig = px.scatter(data, x="race", y="hours-per-week",color='target')
fig.show()

fig=px.box(data,y='hours-per-week',x='modified_age',color='target',points='suspectedoutliers')
fig.show()

data['modified_cl']=data['capital-loss'].map(lambda x : return_cal(x))
fig=px.box(data,y='hours-per-week',x='target',color='modified_cl')
fig.show()

data['modified_cg']=data['capital-gain'].map(lambda x : return_cal(x))
fig=px.box(data,y='hours-per-week',x='target',color='modified_cg')
fig.show()

# data['modified_cp']=data['hours-per-week'].map(lambda x : return_cal(x))
fig=px.box(data,y='hours-per-week',x='target',color='sex')
fig.show()

fig=px.box(data,x='hours-per-week')
fig.show()
print(data['native_country'].value_counts())
# print(data['native_country'].value_counts().shape)
# print(data['native_country'].value_counts())
# print(data['native_country'].unique())
from collections import Counter
ct=Counter(data['native_country'])
# print(max(ct))
for i,j in ct.items():
    print(f" {i}  {j} {j/28958:.3f}")
    
#too many outliers in this column we will drop it initially,later we may use it
#outlier removal and normalization
#age 
data=data[(data['age']<=75) & (data['age']>=17)]

#capital-gain
data=data[data['capital-gain']<=15000]

#capital-loss
data=data[(data['capital-loss']>=1300) & (data['capital-loss']<=2600)]

#hours-per-week
data=data[(data['hours-per-week']<=60) & (data['hours-per-week']>=30)]


#Normalization


data['age']=data['age'].map(lambda x: np.sqrt(x))


data['capital-gain']=data['capital-gain'].map(lambda x: np.sqrt(x))


data['capital-loss']=data['capital-loss'].map(lambda x: np.sqrt(x))


data['hours-per-week']=data['hours-per-week'].map(lambda x: np.sqrt(x))
data.drop(columns=['native_country'],inplace=True)
data.columns

# #data transformation
# # education --> Label Encoding
# # everything else is One Hot Encode

encoder=LabelEncoder()
data['education']=encoder.fit_transform(data['education'])




df=pd.get_dummies(data,sparse=False,columns=['workclass','marital-status','occupation','relationship','race','sex'],drop_first=True)
df['education']=data['education']
df['target']=data['target']

df.reset_index(inplace=True)
df.drop(columns='index',inplace=True)
df.head()
#Standardization
# scaler=StandardScaler()
dt=pd.read_csv('../input/adultcsv/adult.csv')
print(dt.describe())
print(df.describe())
# #removing imbalancing using SMOTE

Y=df['target']
X=df.drop(columns=['target'])
print(X.head(),Y.head())
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=232)
ros = SMOTE(random_state = 33)

xtrain2,ytrain2 = ros.fit_resample(xtrain,ytrain)
print('Before smote xtrain and ytrain shape ',xtrain.shape,ytrain.shape)
print('After smote xtrain and ytrain shape ',xtrain2.shape,ytrain2.shape)
#checking y classes
print(pd.Series(ytrain2).value_counts())
from sklearn.svm import SVC
svc=SVC(verbose=1,random_state=23,probability=True)
svc.fit(xtrain2,ytrain2)
pred=svc.predict_proba(xtest)
pred_tr=svc.predict_proba(xtrain2)
pred1=svc.predict(xtrain2)
pred2=svc.predict(xtest)
print('Log-loss')
print(log_loss(ytrain2,pred_tr))
print(log_loss(ytest,pred))
print('Accuracy Score')
print(accuracy_score(ytrain2,pred1))
print(accuracy_score(ytest,pred2))
print('ROC_AUC')
print(roc_auc_score(ytrain2,pred1))
print(roc_auc_score(ytest,pred2))

svc=SVC(verbose=5,random_state=23,C=0.9,gamma=0.5,kernel='rbf',probability=True)
svc.fit(xtrain2,ytrain2)
pred=svc.predict_proba(xtest)
pred_tr=svc.predict_proba(xtrain2)
pred1=svc.predict(xtrain2)
pred2=svc.predict(xtest)
print('Log loss')
print(log_loss(ytrain2,pred_tr))
print(log_loss(ytest,pred))
print('Accuracy Score')
print(accuracy_score(ytrain2,pred1))
print(accuracy_score(ytest,pred2))
print('ROC_AUC')
print(roc_auc_score(ytrain2,pred1))
print(roc_auc_score(ytest,pred2))
svc=svc=SVC(verbose=5,random_state=23,probability=True,kernel='rbf')
params={'gamma':[0.01,0.1,0.5,0.9,2,5,10],'C':[0.1,0.5,1,5,10,50,100]}
grs=GridSearchCV(svc,params,verbose=1)
grs.fit(xtrain2,ytrain2)
pred1=grs.predict(xtrain2)
pred2=grs.predict(xtest)
print('Log loss')
print(log_loss(ytrain2,grs.predict_proba(xtrain2)))
print(log_loss(ytest,grs.predict_proba(xtest)))
print('Accuracy Score')
print(accuracy_score(ytrain2,pred1))
print(accuracy_score(ytest,pred2))
print('ROC_AUC')
print(roc_auc_score(ytrain2,pred1))
print(roc_auc_score(ytest,pred2))
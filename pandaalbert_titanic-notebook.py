import pandas as pd

import numpy as np 

import matplotlib as mpl 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import power_transform,MinMaxScaler,OneHotEncoder,StandardScaler

from sklearn.model_selection import GridSearchCV,cross_val_score

from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier

from sklearn.linear_model import Ridge,LogisticRegression,RidgeClassifier

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,VotingClassifier,ExtraTreesClassifier

from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.impute import KNNImputer

from lightgbm.sklearn import LGBMClassifier

from xgboost.sklearn import XGBClassifier

import pandas_profiling

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

train_y=train['Survived']

train=train.drop(['PassengerId','Survived'],axis=1)

test=test.drop(['PassengerId'],axis=1)

print(train.shape)
df=pd.concat([train,test],axis=0,ignore_index=True)
df.Name=df.Name.str.rsplit(', ',expand=True)[1].str.rsplit('. ',expand=True)[0]
df.info()
df.profile_report()
for col in list(df.columns):

    tmp=pd.DataFrame(df[col].value_counts()).sort_values(col,ascending=False).values

    if tmp[0,0]/tmp[1,0]>20:

        print('possible degenerate:',col)

print('finished')
df.Pclass=df.Pclass.astype(object)

num_=[]

for col in list(df.columns):

    if df[col].dtype != object:

        num_.append(col)

num_
df_num=df[num_]

df_obj=df.drop(num_,axis=1)
df_num.iloc[:,:4].hist(figsize=(10,10),bins=10)
numage=pd.DataFrame({'Age':list(df_num.Age)[:len(train_y)],'y':list(train_y)})

numage.Age=pd.cut(numage.Age,8)

numage=numage.groupby('Age').sum().reset_index()

numage.Age=numage.Age.astype(object)
plt.bar(x=range(1,9),height=numage.y)
df_num.describe().T
fare_mean=np.mean(df_num.Fare)

df_num.Fare=df_num.Fare.fillna(fare_mean)
tmp=df_num.dropna(axis=0)

tmp_age=tmp['Age']

tmp_other=tmp.drop(['Age'],axis=1)
knn=KNeighborsRegressor()

reg=GridSearchCV(knn,cv=5,scoring='neg_root_mean_squared_error',n_jobs=1,param_grid={'n_neighbors':[16,32,64]})

reg.fit(tmp_other,tmp_age)

reg.best_score_
rid=Ridge()

reg=GridSearchCV(rid,cv=5,scoring='neg_root_mean_squared_error',n_jobs=1,param_grid={'alpha':[0.001,0.01,0.1,1]})

reg.fit(tmp_other,tmp_age)

reg.best_score_
tree=DecisionTreeRegressor()

reg=GridSearchCV(tree,cv=5,scoring='neg_root_mean_squared_error',n_jobs=1,param_grid={'max_depth':[3,4,5]})

reg.fit(tmp_other,tmp_age)

reg.best_score_
reg.best_params_
# rd=RandomForestRegressor()

# reg=GridSearchCV(rd,cv=5,scoring='neg_root_mean_squared_error',n_jobs=2,param_grid={'n_estimators':[100,200,300]})

# reg.fit(tmp_other,tmp_age)

# print(reg.best_score_)

# print(reg.best_params_)
df_na_age=df_num.loc[pd.isnull(df_num.Age)]

df_nona_age=df_num.dropna()
tree=DecisionTreeRegressor(max_depth=5)

tree.fit(X=df_nona_age[['SibSp','Parch','Fare']],y=df_nona_age['Age'])
df_na_age['Age']=tree.predict(X=df_na_age[['SibSp','Parch','Fare']])
df_num=pd.concat([df_na_age,df_nona_age],axis=0)
df_num=df_num.sort_index()
df_num['Pclass']=df_obj.Pclass.values

df_num.Pclass=df_num.Pclass.astype(int)
minmax=MinMaxScaler()

df_num=pd.DataFrame(minmax.fit_transform(df_num.values),columns=df_num.columns)
df_num.info()
df_obj=df_obj.drop('Pclass',axis=1)

df_obj.info()
df_obj.Embarked=df_obj.Embarked.fillna(df_obj.Embarked.value_counts().index[0])
df_obj.head()
295/1309
tmp=df_obj.Ticket.str.rsplit(' ',expand=True,n=0)

tmp[0][pd.isnull(tmp[2])==False]=tmp[0][pd.isnull(tmp[2])==False]+tmp[1][pd.isnull(tmp[2])==False]

tmp[2][pd.isnull(tmp[2])]=tmp[1][pd.isnull(tmp[2])]

tmp[2][pd.isnull(tmp[2])]=tmp[0][pd.isnull(tmp[2])]

tmp[0][pd.isnull(tmp[1])]='Pure.number'

tmp.columns=['Tic1','Ticno','Tic2']

tmp=tmp[['Tic1','Tic2']]
df_obj=pd.concat([df_obj,tmp],axis=1)

df_obj=df_obj.drop('Ticket',axis=1)
df_obj.info()
def count_values(df):

    return df['Cabin'].value_counts()

tmp=df_obj[['Sex','Cabin']]

tmp['Cabin']=tmp['Cabin'].str[0]

tmp_sex=tmp[['Sex','Cabin']].groupby('Sex',as_index=True).apply(count_values).reset_index()
sns.barplot(x='level_1',y='Cabin',hue='Sex',data=tmp_sex)

plt.legend(loc='upper right')
tmp=df_obj[['Tic1','Cabin']]

tmp['Cabin']=tmp['Cabin'].str[0]

tmp_tic1=tmp[['Tic1','Cabin']].groupby('Tic1',as_index=True).apply(count_values).reset_index()
sns.barplot(x='level_1',y='Cabin',hue='Tic1',data=tmp_tic1.loc[(tmp_tic1.Tic1=='Pure.number')|(tmp_tic1.Tic1=='PC')])

plt.legend(loc='upper right')
tmp=df_obj[['Embarked','Cabin']]

tmp['Cabin']=tmp['Cabin'].str[0]

tmp_emb=tmp[['Embarked','Cabin']].groupby('Embarked',as_index=True).apply(count_values).reset_index()
sns.barplot(x='level_1',y='Cabin',hue='Embarked',data=tmp_emb.loc[(tmp_emb.Embarked=='C')|(tmp_emb.Embarked=='S')])

plt.legend(loc='upper right')
df_obj.Cabin=df_obj.Cabin.str[0]
fill1=df_obj.copy()
fillcabin=fill1.Cabin

fill1=fill1.drop('Cabin',axis=1)

onehot=OneHotEncoder()

fill_col=list(fill1.columns)

fill1=pd.DataFrame(onehot.fit_transform(fill1).toarray(),columns=onehot.get_feature_names(fill_col))
fill1['Cabin']=fillcabin
fill1=pd.merge(fill1,df_num,left_index=True,right_index=True,how='left')
fill1_na=fill1.loc[pd.isnull(fill1.Cabin)]

fill1_nona=fill1.dropna()
for n in range(2,11):

    clf=DecisionTreeClassifier(max_depth=n)

    mul=OneVsOneClassifier(clf,n_jobs=1)

    mul.fit(fill1_nona.drop('Cabin',axis=1),fill1_nona.Cabin)

    print(n,accuracy_score(fill1_nona.Cabin,mul.predict(fill1_nona.drop('Cabin',axis=1))))
for n in range(2,10):

    clf=DecisionTreeClassifier(max_depth=n)

    mul=OneVsOneClassifier(clf,n_jobs=2)

    cs=cross_val_score(mul,X=fill1_nona.drop('Cabin',axis=1),y=fill1_nona.Cabin,scoring='accuracy',cv=10)

    print(n,np.mean(cs))
clf=DecisionTreeClassifier(max_depth=9)

mul=OneVsOneClassifier(clf,n_jobs=2)

mul.fit(fill1_nona.drop('Cabin',axis=1),fill1_nona.Cabin)

cabin_pred=mul.predict(fill1_na.drop('Cabin',axis=1))
fill1_na.Cabin=cabin_pred
na_cabin=fill1_na['Cabin']
df_obj['Cabin'][pd.isnull(df_obj.Cabin)]=na_cabin
df_obj['y']=train_y
tmp=df_obj.copy()
tmp['Tic2'][tmp.Tic2=='LINE']=0
tmp.Tic2=tmp.Tic2.astype(int)

def cal_mean(df):

    return np.mean(df['Tic2'])

tmp[['y','Tic2']].groupby('y').apply(cal_mean)
sns.boxplot(x=tmp['y'],y=tmp['Tic2'])

plt.ylim((-10000,450000))
df_num.Pclass=df_num.Pclass.astype(int)

df_num['Tic2']=df_obj.Tic2.values

df_num['Tic2'][df_num.Tic2=='LINE']=0

df_num.Tic2=df_num.Tic2.astype(int)
minmax=MinMaxScaler()

df_num.Tic2=minmax.fit_transform(df_num.Tic2.values.reshape(-1,1))
df_obj=df_obj.drop('Tic2',axis=1)

df_obj
def cal_mean(df):

    return np.mean(df['y'])

for col in list(df_obj.columns)[:-1]:

    tmp=df_obj[[col,'y']].groupby(col).apply(cal_mean).reset_index()

    if max(tmp[0])-min(tmp[0])>=0.5:

        change={}

        for i,j in zip(tmp[col],tmp[0]):

            change[i]=j

        df_obj[col]=df_obj[col].replace(change)
df_obj=df_obj.drop('y',axis=1)

# onehot=OneHotEncoder()

# emb=pd.DataFrame(onehot.fit_transform(df_obj['Embarked'].values.reshape(-1,1)).toarray(),columns=onehot.get_feature_names(['Embarked']))
# df_obj=pd.concat([df_obj.drop('Embarked',axis=1),emb],axis=1)

df_obj=df_obj.drop('Embarked',axis=1)
df_obj.info()
df=pd.concat([df_num,df_obj],axis=1)
knnimp=KNNImputer(n_neighbors=8)

df=pd.DataFrame(knnimp.fit_transform(df),columns=df.columns)
train=df.iloc[:891,:]

test=df.iloc[891:,:]
train.shape
lg=LGBMClassifier()

best=GridSearchCV(lg,cv=10,scoring='accuracy',param_grid={'n_estimators':[30,50,80],

                                                          'reg_lambda':[0.006,0.008,0.01]})

best.fit(train,train_y)

print(best.best_score_)

print(best.best_params_)
xgb=XGBClassifier()

best=GridSearchCV(xgb,cv=10,scoring='accuracy',param_grid={'max_depth':[1,2,3],

                                                          'reg_lambda':[0.06,0.08,0.1]})

best.fit(train,train_y)

print(best.best_score_)

print(best.best_params_)
rd=RandomForestClassifier(random_state=42)

best=GridSearchCV(rd,cv=10,scoring='accuracy',param_grid={'n_estimators':[520,550,580]},n_jobs=-1)

best.fit(train,train_y)

print(best.best_score_)

print(best.best_params_)
rd=ExtraTreesClassifier(random_state=42)

best=GridSearchCV(rd,cv=10,scoring='accuracy',param_grid={'n_estimators':[80,100,120]})

best.fit(train,train_y)

print(best.best_score_)

print(best.best_params_)
rd=RandomForestClassifier(n_estimators=550)

rd.fit(train,train_y)
pd.DataFrame({'Feature':list(train.columns),'importance':rd.feature_importances_}).sort_values('importance',ascending=False)
svc=SVC(random_state=42)

best=GridSearchCV(svc,cv=10,scoring='accuracy',param_grid={'C':[0.001,0.01,0.1,1],

                                                          'kernel':['rbf','poly'],

                                                          'degree':[1,2,3],

                                                          'gamma':['auto','scale',0.1,0.2],

                                                          },n_jobs=2)

best.fit(train,train_y)

print(best.best_score_)

print(best.best_params_)
clf=DecisionTreeClassifier(random_state=42)

best=GridSearchCV(clf,cv=10,scoring='accuracy',param_grid={'max_depth':[2,4,8,16,32]},n_jobs=1)

best.fit(train,train_y)

print(best.best_score_)

print(best.best_params_)
voting=VotingClassifier(estimators=[('rf',RandomForestClassifier(n_estimators=550)),

                                   ('lg',LGBMClassifier(n_estimators=80,reg_lambda=0.008)),

                                   ('xgb',XGBClassifier(max_depth=2,reg_lambda=0.1))],

                       voting='hard')

cv=cross_val_score(voting,X=train,y=train_y,cv=10)
cv.mean()
voting=VotingClassifier(estimators=[('rf',RandomForestClassifier(n_estimators=550)),

                                   ('lg',LGBMClassifier(n_estimators=80,reg_lambda=0.008)),

                                   ('xgb',XGBClassifier(max_depth=2,reg_lambda=0.1))],

                       voting='soft')

voting.fit(train,train_y)

sur_pred=voting.predict(test)
sur_pred=pd.DataFrame({'PassengerId':range(892,1310),'Survived':sur_pred})
sur_pred.to_csv('./pred.csv',index=False)
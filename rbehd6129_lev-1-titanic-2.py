# 기본 패키지 임포트

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# train 데이터 불러오기

train=pd.read_csv('../input/titanic/train.csv')

train.head()
# train 데이터 결측값 확인

train.isnull().sum()
# Survived 그래프

f,ax=plt.subplots(1,2,figsize=(18,8))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
# 성별과 생존 간의 관계

train.groupby(['Sex','Survived'])['Survived'].count()
# 그래프 그리기

f,ax=plt.subplots(1,2,figsize=(18,8))

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
pd.crosstab(train.Pclass,train.Survived,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
# Sex, Pclass와 생존 간의 관계

pd.crosstab([train.Sex,train.Survived],train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
# Sex, Pclass와 생존 간의 그래프 그리기

sns.factorplot('Pclass','Survived',hue='Sex',data=train)

plt.show()
# 승객의 나이 분석

print('Oldest Passenger was of:',train['Age'].max(),'Years')

print('Youngest Passenger was of:',train['Age'].min(),'Years')

print('Average Age on the ship:',train['Age'].mean(),'Years')
# Age, Pclass, Sex와 생존 간의 그래프

f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
# Initial 변수 생성, 특정 이름 추출

train['Initial'] = 0

for i in train:

    train['Initial'] = train.Name.str.extract('([A-Za-z]+)\.')
# 성별과 이름 간의 관계 파악

pd.crosstab(train.Initial,train.Sex).T.style.background_gradient(cmap='summer_r')
# 고유 이름을 Miss, Mr, Other로 변경

train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',

                          'Jonkheer','Col','Rev','Capt','Sir','Don'],

                         ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',

                          'Other','Other','Other','Mr','Mr','Mr'], inplace=True)
# 이니셜별 평균 나이 확인

train.groupby('Initial')['Age'].mean()
# Age 결측값 대체

train.loc[(train.Age.isnull())&(train.Initial=='Mr'),'Age']=33

train.loc[(train.Age.isnull())&(train.Initial=='Mrs'),'Age']=36

train.loc[(train.Age.isnull())&(train.Initial=='Master'),'Age']=5

train.loc[(train.Age.isnull())&(train.Initial=='Miss'),'Age']=22

train.loc[(train.Age.isnull())&(train.Initial=='Other'),'Age']=46
# 연령별 생존자와 연령별 사망자 그래프

f,ax=plt.subplots(1,2,figsize=(20,10))

train[train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

train[train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
# 각 이니셜 별 Pclass와 생존 간의 그래프

sns.factorplot('Pclass','Survived',col='Initial',data=train)

plt.show()
# Embarked, Pclass, Sex와 생존 간의 관계

pd.crosstab([train.Embarked,train.Pclass],[train.Sex,train.Survived],margins=True).style.background_gradient(cmap='summer_r')
# 입항항구와 생존 간의 그래프

sns.factorplot('Embarked','Survived',data=train)

fig=plt.gcf()

fig.set_size_inches(5,3)

plt.show()
# Embarked, Sex, Pclass와 생존 각각의 그래프

f,ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked',data=train,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=train,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=train,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=train,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
# Embarked, Sex, Pclass, 생존 간 그래프 2

sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=train)

plt.show()
# Embarked 변수 결측치 제거

train['Embarked'].fillna('S',inplace=True)
# SibSp와 생존 간의 관계

pd.crosstab([train.SibSp],train.Survived).style.background_gradient(cmap='summer_r')
# SibSp와 생존 간의 그래프

f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('SibSp','Survived',data=train,ax=ax[0])

ax[0].set_title('SibSp vs Survived')

sns.factorplot('SibSp','Survived',data=train,ax=ax[1])

ax[1].set_title('SibSp vs Survived')

plt.close(2)

plt.show()
# SibSp와 Pclass 간의 관계

pd.crosstab(train.SibSp,train.Pclass).style.background_gradient(cmap='summer_r')
pd.crosstab(train.Parch,train.Pclass).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('Parch','Survived',data=train,ax=ax[0])

ax[0].set_title('Parch vs Survived')

sns.factorplot('Parch','Survived',data=train,ax=ax[1])

ax[1].set_title('Parch vs Survived')

plt.close(2)

plt.show()
print('Highest Fare was:', train['Fare'].max())

print('Lowest Fare was:',train['Fare'].min())

print('Average Fare was:',train['Fare'].mean())
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()
# 상관관계 분석 그래프(train.corr : 상관관계 행렬)

sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# Age 카테고리화

train['Age_band'] = 0

train.loc[train['Age']<=10,'Age_band']=0 # 0세

train.loc[(train['Age']>10)&(train['Age']<=20),'Age_band']=1 # 10대

train.loc[(train['Age']>20)&(train['Age']<=30),'Age_band']=2 # 20대

train.loc[(train['Age']>30)&(train['Age']<=40),'Age_band']=3 # 30대

train.loc[(train['Age']>40)&(train['Age']<=50),'Age_band']=4 # 40대

train.loc[(train['Age']>50)&(train['Age']<=60),'Age_band']=5 # 50대

train.loc[(train['Age']>60)&(train['Age']<=70),'Age_band']=6 # 60대

train.loc[train['Age']>70,'Age_band']=7 # 70대

train.head(2)
# 각 연령대별 승객 수 비교

train['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
# 연령대별 Pclass별 생존률 비교

sns.factorplot('Age_band','Survived',data=train,col='Pclass')

plt.show()
# 변수 생성

train['Family_Size']=0

train['Family_Size']=train['Parch']+train['SibSp']

train['Alone']=0

train.loc[train.Family_Size==0,'Alone']=1



# 변수들의 비교

f,ax=plt.subplots(1,2,figsize=(18,6))

sns.factorplot('Family_Size','Survived',data=train,ax=ax[0])

ax[0].set_title('Family_Size vs Survived')

sns.factorplot('Alone','Survived',data=train,ax=ax[1])

ax[1].set_title('Alone vs Survived')

plt.close(2)

plt.close(3)

plt.show()
sns.factorplot('Alone','Survived',data=train,hue='Sex',col='Pclass')

plt.show()
train['Fare_Range']=pd.qcut(train['Fare'],4)

train.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
# Fare 카테고리화

train['Fare_cat']=0

train.loc[train['Fare']<=7.91,'Fare_cat']=0

train.loc[(train['Fare']>7.91)&(train['Fare']<=14.454),'Fare_cat']=1

train.loc[(train['Fare']>14.454)&(train['Fare']<=31),'Fare_cat']=2

train.loc[train['Fare']>31,'Fare_cat']=3
# Fare_cat, Sex별 생존률 그래프

sns.factorplot('Fare_cat','Survived',data=train,hue='Sex')

plt.show()
# 문자열 수치화

train['Sex'].replace(['male','female'],[0,1],inplace=True)

train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

train['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
# 불필요 변수 제거 후 각 변수별 상관관계 분석

train.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId', 'SibSp', 'Parch'],axis=1,inplace=True)

sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
# ML 전용 패키지 모두 임포트

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
# 훈련데이터, 테스트데이터 분리

train_a,test_a=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])

train_X=train_a[train_a.columns[1:]]

train_Y=train_a[train_a.columns[:1]]

test_X=test_a[test_a.columns[1:]]

test_Y=test_a[test_a.columns[:1]]

X=train[train.columns[1:]]

Y=train['Survived']
# 1. Logiistic Regression

model = LogisticRegression()

model.fit(train_X,train_Y)

prediction1 = model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction1, test_Y))
# 2-1. Linear SVM(Support Vector Machines)

model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)

model.fit(train_X,train_Y)

prediction2=model.predict(test_X)

print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2, test_Y))
# 2-2. Radial SVM(Support Vector Machines)

model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(train_X,train_Y)

prediction3=model.predict(test_X)

print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction3,test_Y))
# 3. Random Forest

model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_Y)

prediction4=model.predict(test_X)

print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction4,test_Y))
# 4. K-Nearest Neighbors

model=KNeighborsClassifier() 

model.fit(train_X,train_Y)

prediction5=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))
a_index=list(range(1,11))

a=pd.Series()

x=[0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_Y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))

plt.plot(a_index, a)

plt.xticks(x)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()

print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
# 5. Gaussian Naive Bayes

model=GaussianNB()

model.fit(train_X,train_Y)

prediction6=model.predict(test_X)

print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction6,test_Y))
# 6. Decision Tree

model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction7=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction7,test_Y))

# 교차검증 패키지 임포트

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction
# 교차검증

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Logistic Regression','Linear Svm','Radial Svm','Random Forest','KNN','Naive Bayes','Decision Tree']

models=[LogisticRegression(),svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),RandomForestClassifier(n_estimators=100),

        KNeighborsClassifier(n_neighbors=9),GaussianNB(),DecisionTreeClassifier()]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

new_models_dataframe2
# 각 모델별 교차검증 결과 그래프

plt.subplots(figsize=(12,6))

box=pd.DataFrame(accuracy,index=[classifiers])

box.T.boxplot()
# 각 모델별 교차검증 시행 후 평균 정확도 비교

new_models_dataframe2['CV Mean'].plot.barh(width=0.8)

plt.title('Average CV Mean Accuracy')

fig=plt.gcf()

fig.set_size_inches(8,5)

plt.show()
# 모델별 교차검증 후 오차행렬 비교 그래프

f,ax=plt.subplots(3,3,figsize=(12,10))

y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')

ax[0,0].set_title('Matrix for rbf-SVM')

y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')

ax[0,1].set_title('Matrix for Linear-SVM')

y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')

ax[0,2].set_title('Matrix for KNN')

y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')

ax[1,0].set_title('Matrix for Random-Forests')

y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')

ax[1,1].set_title('Matrix for Logistic Regression')

y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')

ax[1,2].set_title('Matrix for Decision Tree')

y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')

ax[2,0].set_title('Matrix for Naive Bayes')

plt.subplots_adjust(hspace=0.2,wspace=0.2)

plt.show()
# SVM

from sklearn.model_selection import GridSearchCV

C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel=['rbf','linear']

hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
# Random Forests

n_estimators=range(100,1000,100)

hyper={'n_estimators':n_estimators}

gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
# Voting Classifier

from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),

                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),

                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),

                                              ('LR',LogisticRegression(C=0.05)),

                                              ('DT',DecisionTreeClassifier(random_state=0)),

                                              ('NB',GaussianNB()),

                                              ('svm',svm.SVC(kernel='linear',probability=True))

                                             ], 

                       voting='soft').fit(train_X,train_Y)

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))

cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")

print('The cross validated score is',cross.mean())
# Bagging for KNN

from sklearn.ensemble import BaggingClassifier

model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))

result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for bagged KNN is:',result.mean())
# Bagging for Decision Tree

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))

result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for bagged Decision Tree is:',result.mean())
# 1. AdaBoost(Adaptive Boosting)

from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for AdaBoost is:',result.mean())
# 2. Stochastic Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)

result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boosting is:',result.mean())
# 3. XGBoost

import xgboost as xg

xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')

print('The cross validated score for XGBoost is:',result.mean())
n_estimators=list(range(100,1100,100))

learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}

gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
# AdaBoost 오차행렬 

ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)

result=cross_val_predict(ada,X,Y,cv=10)

sns.heatmap(confusion_matrix(Y,result),cmap='winter',annot=True,fmt='2.0f')

plt.show()
f,ax=plt.subplots(2,2,figsize=(15,12))

model=RandomForestClassifier(n_estimators=500,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])

ax[0,0].set_title('Feature Importance in Random Forests')

model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')

ax[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')

ax[1,0].set_title('Feature Importance in Gradient Boosting')

model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')

ax[1,1].set_title('Feature Importance in XgBoost')

plt.show()
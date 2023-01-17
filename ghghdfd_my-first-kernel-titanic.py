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

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5)



#ignore warning

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')
df_train.describe()
df_test.describe()
for col in df_train.columns:

    msg='columns: {:>10}\t Percent of NaN value: {: 2f}%'.format(col,100*(df_train[col].isnull().sum()/df_train[col].shape[0]))

    print(msg)
for col in df_test.columns:

    msg='columns: {:>10}\t Percent of NaN value: {: 2f}%'.format(col,100*(df_test[col].isnull().sum()/df_test[col].shape[0]))

    print(msg)
f, ax= plt.subplots(1,2,figsize=(18,8)) # 1행2열, 가로 18 세로 8의 plot 만들기 

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],# pie plot 간에 간격 벌려줌

                                             autopct='%1.1f%%',ax=ax[0],shadow=True) # 퍼센트 표시와 그림자 설정

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel(' ')

sns.countplot('Survived',data=df_train,ax=ax[1])

ax[1].set_title('Countplot - Survived')

ax[1].set_ylabel(' ')

plt.show()
df_train.shape
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).count()
df_train[['Pclass','Survived']]
pd.crosstab(df_train['Pclass'],df_train['Survived'],margins=True).style.background_gradient(cmap='cool')
y_position=1.02

f, ax= plt.subplots(1,2,figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('number of passenger by pclass')

ax[0].set_ylabel('count')

sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('pclass: survived vs dead',y=y_position)

plt.show()
f, ax= plt.subplots(1,2,figsize=(18,8))

df_train[['Sex','Survived']].groupby(['Sex'],as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

ax[0].set_ylabel(' ')

sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()

pd.crosstab(df_train['Sex'],df_train['Survived'],margins=True).style.background_gradient(cmap='summer')
sns.factorplot('Pclass','Survived',hue='Sex',data=df_train,size=6,aspect=1.5)

plt.show()
sns.factorplot('Sex','Survived',col='Pclass',data=df_train,size=6,aspect=1.5)

plt.show()
print('oldest people age: {:.1f} Years'.format(df_train['Age'].max()))

print('youngest people age: {:.1f} Years'.format(df_train['Age'].min()))

print('mean age: {:.1f} Years'.format(df_train['Age'].mean()))
f,ax= plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[df_train['Survived']==1]['Age'],ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])

plt.show()
cummulate_suvival_ratio=[]

for i in range(1, 80):

    cummulate_suvival_ratio.append(df_train[df_train['Age']<i]['Survived'].sum()/len(df_train[df_train['Age']<i]['Survived']))



plt.figure(figsize=(7,7))

plt.plot(cummulate_suvival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
f, ax= plt.subplots(1,2,figsize=(18,8))

sns.violinplot('Pclass','Age',hue='Survived',data=df_train,sclae='count',split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))



sns.violinplot('Sex','Age',hue='Survived',data=df_train,sclae='count',split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))



plt.show()

##나이가 어릴수록 생존 높다 특히 여성과 아이들을 먼저 챙겼다는 걸 알 수 있다.

f,ax= plt.subplots(1,1,figsize=(7,7))

df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax)

##생존률은 탑승항구별로 비슷한 걸로 보임
f,ax = plt.subplots(2,2,figsize=(15,10))

sns.countplot('Embarked',data=df_train,ax=ax[0,0])



sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])



sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,0])



sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,1])



plt.subplots_adjust(wspace=0.5, hspace=0.6) #좌우상하 간격 설정해줌

plt.show()
df_train['FamilySize']=df_train['SibSp']+df_train['Parch']+1 #자기 자신 포함해야 해서 1 더함, 총 가족 수 구한 것.

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 
print("Maximum size of Family: ", df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())
f,ax= plt.subplots(1,3,figsize=(40,10))

sns.countplot('FamilySize',data=df_train,ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
f, ax = plt.subplots(1,1,figsize=(8,8))

g=sns.distplot(df_train['Fare'],color='b',label='skewness : {:.2f}'.format(df_train['Fare'].skew()),ax=ax)

g=g.legend(loc='best')
print('max: ',df_train['Fare'].max())

print('min: ',df_train['Fare'].min())

print('mean: ',df_train['Fare'].mean())

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환
df_train['Fare']=df_train['Fare'].map(lambda i : np.log(i) if i>0 else 0)

df_test['Fare']=df_test['Fare'].map(lambda i : np.log(i) if i>0 else 0)
f, ax = plt.subplots(1,1,figsize=(8,8))

g=sns.distplot(df_train['Fare'],color='b',label='skewness : {:.2f}'.format(df_train['Fare'].skew()),ax=ax)

g=g.legend(loc='best')
df_train.isnull().sum()
df_train['Name']
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.')

df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap='summer')
pd.crosstab(df_test['Initial'],df_test['Sex']).T.style.background_gradient(cmap='summer')
df_train['Initial'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir','Dona'],

                           ['Mr','Mr','Other','Mr','Mr','Other','Miss','Mr','Master','Miss','Miss','Mrs','Mr','Mrs','Miss','Other','Other','Other'],inplace=True)
df_test['Initial'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir','Dona'],

                           ['Mr','Mr','Other','Mr','Mr','Other','Miss','Mr','Master','Miss','Miss','Mrs','Mr','Mrs','Miss','Other','Other','Other'],inplace=True)
df_train.groupby(['Initial']).mean().sort_values(by='Survived',ascending=False)
df_train.groupby('Initial')['Survived'].mean().plot.bar()
df_all=pd.concat([df_train,df_test])
df_all
df_all.reset_index(drop=True)
df_all.groupby('Initial').mean()
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age']=33

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'),'Age']=37

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age']=22

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age']=5

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age']=33



df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age']=33

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age']=37

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age']=22

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age']=5

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age']=33
df_train.isnull().sum()
df_train['Embarked'].isnull().sum()
df_train['Embarked'].fillna('S',inplace=True)
df_train['Embarked'].value_counts()
def age_cat(x):

    if x<10:

        return 0

    elif x<20:

        return 1

    elif x<30:

        return 2

    elif x<40:

        return 3

    elif x<50:

        return 4

    elif x<60:

        return 5

    elif x<70:

        return 6

    else:

        return 7

    

df_train['Age_cat']=df_train['Age'].apply(age_cat)
df_test['Age_cat']=df_train['Age'].apply(age_cat)
df_train.drop(['Age'],axis=1,inplace=True)

df_test.drop(['Age'],axis=1,inplace=True)
df_train['Initial']=df_train['Initial'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Other':4})
df_test['Initial']=df_test['Initial'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Other':4})
df_train['Embarked'].unique()
df_train['Embarked']=df_train['Embarked'].map({'C':0,'Q':1,'S':2})
df_test['Embarked']=df_test['Embarked'].map({'C':0,'Q':1,'S':2})
df_train.head()
df_train.isnull().any()
df_train['Sex']=df_train['Sex'].map({'female':0,'male':1})

df_test['Sex']=df_test['Sex'].map({'female':0,'male':1})
df_train.head()
heatmap_data=df_train[['Survived','Pclass','Sex','SibSp','Fare','Embarked','FamilySize','Initial','Age_cat']]



col_map=plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(),linewidths=0.1,vmax=1.0,

           square=True,cmap=col_map,linecolor='White',annot=True,annot_kws={'size':16})



del heatmap_data

df_train=pd.get_dummies(df_train,columns=['Initial'],prefix='Initial')
df_test=pd.get_dummies(df_test,columns=['Initial'],prefix='Initial')
df_train=pd.get_dummies(df_train,columns=['Embarked'],prefix='Embarked')

df_test=pd.get_dummies(df_test,columns=['Embarked'],prefix='Embarked')
df_train.head()
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
df_test.head()
#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
train,valid=train_test_split(df_train,test_size=0.3,random_state=0,stratify=df_train['Survived'])

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

vld_X=valid[valid.columns[1:]]

vld_Y=valid[valid.columns[:1]]

X=df_train[df_train.columns[1:]]

Y=df_train['Survived']
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(train_X,train_Y)

prediction1=model.predict(vld_X)

print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,vld_Y))
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)

model.fit(train_X,train_Y)

prediction2=model.predict(vld_X)

print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,vld_Y))
model = LogisticRegression()

model.fit(train_X,train_Y)

prediction3=model.predict(vld_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,vld_Y))
model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction4=model.predict(vld_X)

print('The accuract of Decision Tree is' ,metrics.accuracy_score(prediction4,vld_Y))
model=KNeighborsClassifier()

model.fit(train_X,train_Y)

prediction5=model.predict(vld_X)

print('The accuracy of KNN is ',metrics.accuracy_score(prediction5,vld_Y))
a_index=list(range(1,11))

a=pd.Series()

x=[0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_Y)

    prediction=model.predict(vld_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,vld_Y)))

plt.plot(a_index, a)

plt.xticks(x)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()

print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
model=GaussianNB()

model.fit(train_X,train_Y)

prediction6=model.predict(vld_X)

print('The accuracy of the GaussianNB is', metrics.accuracy_score(prediction6,vld_Y))

model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_Y)

prediction7=model.predict(vld_X)

print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,vld_Y))
from pandas import Series
feature_importance=model.feature_importances_

Series_feat_imp=Series(feature_importance,index=df_test.columns)
plt.figure(figsize=(8,8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature Importance')

plt.ylabel('Feature')

plt.show()
from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

new_models_dataframe2
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)

plt.title('Average CV Mean Accuracy')

fig=plt.gcf()

fig.set_size_inches(8,5)

plt.show()
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)

model.fit(train_X,train_Y)
submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission.head()

prediction_final=model.predict(df_test)
submission['Survived']=prediction_final
submission.to_csv('./myfirstsubmission.csv',index=False)
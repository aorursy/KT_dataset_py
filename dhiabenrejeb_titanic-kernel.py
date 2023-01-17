import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
gender_sub=pd.read_csv('../input/gender_submission.csv')
gender_sub
train=pd.read_csv('../input/train.csv')
train
train.isnull().sum()
train.groupby(['Sex','Survived'])['Survived'].count()
print('minimum age',train['Age'].min())
print('maximum age ',train['Age'].max())
print('average age',train['Age'].mean())
train['Name']
train['Salutation']=0
for i in train:
    train['Salutation']=train.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(train.Salutation,train.Sex.count())
train.groupby('Salutation')['Age'].mean()
train.loc[(train.Age.isnull())&(train.Salutation=='capt'),'Age']=70
train.loc[(train.Age.isnull())&(train.Salutation=='col'),'Age']=58
train.loc[(train.Age.isnull())&(train.Salutation=='countess'),'Age']=33
train.loc[(train.Age.isnull())&(train.Salutation=='Don'),'Age']=40
train.loc[(train.Age.isnull())&(train.Salutation=='Dr'),'Age']=42
train.loc[(train.Age.isnull())&(train.Salutation=='Jonkheer'),'Age']=38
train.loc[(train.Age.isnull())&(train.Salutation=='Lady'),'Age']=48
train.loc[(train.Age.isnull())&(train.Salutation=='Major'),'Age']=48
train.loc[(train.Age.isnull())&(train.Salutation=='Master'),'Age']=5
train.loc[(train.Age.isnull())&(train.Salutation=='Miss'),'Age']=22
train.loc[(train.Age.isnull())&(train.Salutation=='Mlle'),'Age']=24
train.loc[(train.Age.isnull())&(train.Salutation=='Mme'),'Age']=24
train.loc[(train.Age.isnull())&(train.Salutation=='Mr'),'Age']=32
train.loc[(train.Age.isnull())&(train.Salutation=='Mrs'),'Age']=36
train.loc[(train.Age.isnull())&(train.Salutation=='Ms'),'Age']=28
train.loc[(train.Age.isnull())&(train.Salutation=='Rev'),'Age']=43
train.loc[(train.Age.isnull())&(train.Salutation=='Sir'),'Age']=49
train.Age.isnull().sum()
sns.countplot('Embarked',hue='Survived',data=train)

train['Embarked'].fillna('S',inplace=True)
train.Embarked.isnull().sum()
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%')
sns.countplot('Sex',hue='Survived',data=train)
f,ax=plt.subplots(1,2,figsize=(18,8))
train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By class')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('class:Survived vs Dead')
plt.show()
sns.factorplot("Pclass", "Survived", "Sex", data=train, kind="bar", size=6, aspect=2, palette="muted")
plt.title("Survived: Sex vs Social Class")
f,ax=plt.subplots(1,2,figsize=(20,10))
train[train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Dead')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')
ax[1].set_title('Survived')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
sns.factorplot('Embarked','Survived',data=train, kind="bar", size=6, aspect=2, palette="muted")
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()
sns.countplot('Embarked',hue='Pclass',data=train)
plt.title('Embarked vs Social class')
pd.crosstab([train.SibSp],train.Survived)

sns.barplot('SibSp','Survived',data=train)
sns.factorplot("SibSp", "Survived", "Pclass", data=train, kind="bar", size=6, aspect=2, palette="muted")
plt.title("Survived: Family vs Social Class")
sns.barplot('Parch','Survived',data=train)
plt.title('Parch vs Survived')
mask=sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

train['Sex'].replace(['male','female'],[0,1],inplace=True)
train['child'] = [1 if i<16 else 0 for i in train.Age]
train['Family_Size']=0
train['Family_Size']=train['Parch']+train['SibSp']
train['Alone']=0
train.loc[train.Family_Size==0,'Alone']=1
f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('Family_Size','Survived',data=train,kind='bar',ax=ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=train,kind='bar',ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()
train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train['Age_group']=0
train.loc[train['Age']<=16,'Age_group']=0
train.loc[(train['Age']>16)&(train['Age']<=32),'Age_group']=1
train.loc[(train['Age']>32)&(train['Age']<=48),'Age_group']=2
train.loc[(train['Age']>48)&(train['Age']<=64),'Age_group']=3
train.loc[train['Age']>64,'Age_group']=4
train.head(2)
train.drop(['Name','Ticket','Cabin','PassengerId','Fare','Salutation','Age'],axis=1,inplace=True)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,12))
sns.heatmap(train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20);
train
test=pd.read_csv('../input/test.csv')
test
test.isnull().sum()
test['Salutation']=0
for i in test:
    test['Salutation']=test.Name.str.extract('([A-Za-z]+)\.')
test.loc[(test.Age.isnull())&(test.Salutation=='capt'),'Age']=70
test.loc[(test.Age.isnull())&(test.Salutation=='col'),'Age']=58
test.loc[(test.Age.isnull())&(test.Salutation=='countess'),'Age']=33
test.loc[(test.Age.isnull())&(test.Salutation=='Don'),'Age']=40
test.loc[(test.Age.isnull())&(test.Salutation=='Dr'),'Age']=42
test.loc[(test.Age.isnull())&(test.Salutation=='Jonkheer'),'Age']=38
test.loc[(test.Age.isnull())&(test.Salutation=='Lady'),'Age']=48
test.loc[(test.Age.isnull())&(test.Salutation=='Major'),'Age']=48
test.loc[(test.Age.isnull())&(test.Salutation=='Master'),'Age']=5
test.loc[(test.Age.isnull())&(test.Salutation=='Miss'),'Age']=22
test.loc[(test.Age.isnull())&(test.Salutation=='Mlle'),'Age']=24
test.loc[(test.Age.isnull())&(test.Salutation=='Mme'),'Age']=24
test.loc[(test.Age.isnull())&(test.Salutation=='Mr'),'Age']=32
test.loc[(test.Age.isnull())&(test.Salutation=='Mrs'),'Age']=36
test.loc[(test.Age.isnull())&(test.Salutation=='Ms'),'Age']=28
test.loc[(test.Age.isnull())&(test.Salutation=='Rev'),'Age']=43
test.loc[(test.Age.isnull())&(test.Salutation=='Sir'),'Age']=49
test.isnull().sum()
test['Family_Size']=0
test['Family_Size']=test['Parch']+test['SibSp']
test['Alone']=0
test.loc[test.Family_Size==0,'Alone']=1
test['Sex'].replace(['male','female'],[0,1],inplace=True)
test['child'] = [1 if i<16 else 0 for i in test.Age]
test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
test
test['Age_group']=0
test.loc[test['Age']<=16,'Age_group']=0
test.loc[(test['Age']>16)&(test['Age']<=32),'Age_group']=1
test.loc[(test['Age']>32)&(test['Age']<=48),'Age_group']=2
test.loc[(test['Age']>48)&(test['Age']<=64),'Age_group']=3
test.loc[test['Age']>64,'Age_group']=4
test.head(2)
pid=test['PassengerId']
test.drop(['Name','Ticket','Cabin','PassengerId','Fare','Salutation','Age'],axis=1,inplace=True)
train
test
from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
train1,test1=train_test_split(train,test_size=0.3,random_state=0,stratify=train['Survived'])
train_X=train1[train1.columns[1:]]
train_Y=train1[train1.columns[:1]]
test_X=test1[test1.columns[1:]]
test_Y=test1[test1.columns[:1]]
X=train1[train1.columns[1:]]
Y=train1['Survived']
model=svm.SVC(kernel='rbf',C=1,gamma=0.05)
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))
model=svm.SVC(kernel='linear',C=0.1,gamma=0.05)
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))

acc=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    acc=acc.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(range(1,11), acc)
plt.show()
model=KNeighborsClassifier(n_neighbors=6) 
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction4,test_Y))
from sklearn.model_selection import cross_val_predict
f,ax=plt.subplots(2,2,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Confusion Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Confusion Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Confusion Matrix for KNN')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Confusion Matrix for Logistic Regression')

prediction = svm.SVC(kernel='rbf',C=1,gamma=0.05)
prediction.fit(train_X,train_Y)
Surv=model.predict(test)

test['PassengerId']=pid
test['Survived']=Surv
test.drop(['Pclass','Sex','Age_group','SibSp','Parch','Embarked','Family_Size','Alone','child'],axis=1,inplace=True)
test
test.to_csv('sub1.csv', encoding='utf-8', index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.head()

data.isnull()
data.isnull().sum()
pd.crosstab(data.Sex,data.Survived,margins=True).style.background_gradient(cmap='summer_r')
data.groupby(['Sex','Survived'])['Survived'].count()

f,ax=plt.subplots(1,2,figsize=(10,5))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.2f%%',ax=ax[0])
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()
f,ax=plt.subplots(1,2,figsize=(10,5))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Survivied vs Dead')
plt.show()
data.groupby(['Pclass', 'Survived'])['Survived'].count()
pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='Wistia')
f,ax=plt.subplots(1,2,figsize=(10,5))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Passengers by Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass-Survived vs Dead')
plt.show()
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='cool')
sns.factorplot('Pclass','Survived', hue = 'Sex', data=data)
plt.show()
print('Oldest Passenger was ',data['Age'].max(), 'years old.' )
print('Youngest Passenger was ',data['Age'].min(), 'years old')
print('Average age for the passengers was ', data['Age'].mean(), 'Years')
f,ax=plt.subplots(1,2,figsize=(10,5))
sns.violinplot('Pclass', 'Age', hue = 'Survived', data=data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot('Sex', 'Age', hue = 'Survived', data=data, split=True, ax=ax[1])
ax[1].set_title('Age and Sex vs Survival')
ax[1].set_yticks(range(0,110,10))
plt.show()
data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')    
pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='hsv')
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

data.groupby('Initial')['Age'].mean()
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Others'),'Age']=46
data.isnull().sum()
f,ax=plt.subplots(1,2,figsize=(15,10))
data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived=0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')
ax[1].set_title('Survived=1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
sns.factorplot('Pclass','Survived',col='Initial',data=data)
plt.show()
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True).style.background_gradient(cmap='autumn')
sns.factorplot('Embarked', 'Survived', data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()
f,ax=plt.subplots(2,2,figsize=(15,10))
sns.countplot('Embarked',data=data,ax=ax[0,0])
ax[0,0].set_title('Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Gender Embarked')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
sns.factorplot('Pclass','Survived', hue='Sex',col='Embarked',data=data)
plt.show()
data['Embarked'].fillna('S', inplace=True)
data.Embarked.isnull().any()
pd.crosstab([data.SibSp],data.Survived,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(15,5))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
ax[0].set_title('Siblings vs Survived')
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()
pd.crosstab(data.SibSp,data.Pclass).style.background_gradient(cmap='summer_r')

pd.crosstab(data.Parch,data.Pclass).style.background_gradient(cmap='summer_r')

f,ax=plt.subplots(1,2,figsize=(15,5))
sns.barplot('Parch','Survived',data=data,ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived',data=data,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)
plt.show()
print('Highest Fare was:',data['Fare'].max())
print('Lowest Fare was:',data['Fare'].min())
print('Average Fare was:',data['Fare'].mean())
f,ax=plt.subplots(1,2,figsize=(15,10))
data[data['Survived']==0].Fare.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,550,50))
ax[0].set_xticks(x1)
data[data['Survived']==1].Fare.plot.hist(ax=ax[1],color='green',bins=30,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,550,50))
ax[1].set_xticks(x2)
plt.show()
f,ax=plt.subplots(1,2,figsize=(10,5))
sns.violinplot("Pclass","Fare", hue="Survived", data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Fare vs Survived')
ax[0].set_yticks(range(0,550,50))
sns.violinplot("Sex","Fare", hue="Survived", data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Fare vs Survived')
ax[1].set_yticks(range(0,550,50))
plt.show()
sns.heatmap(data.corr(),annot=True,cmap='PiYG',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[(data['Age']>64)&(data['Age']<=80),'Age_band']=4
data.head(2)
data['Age_band'].value_counts().to_frame()
sns.factorplot('Age_band','Survived',data=data,col='Pclass')
plt.show()
sns.barplot('Age_band','Survived',data=data)
data['Family']=0
data['Family']=data['Parch']+data['SibSp']
data['Alone']=0
data.loc[data.Family==0,'Alone']=1
data.head(3)
f,ax=plt.subplots(1,2,figsize=(10,5))
sns.factorplot('Family','Survived',data=data,ax=ax[0])
ax[0].set_title('Family vs Survived')
sns.factorplot('Alone','Survived',data=data,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()
sns.factorplot('Alone','Survived',data=data,hue='Sex',col='Pclass')
data['Fare_range']=pd.qcut(data['Fare'],5)
data.groupby(['Fare_range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
data['Fare_cat']=0
data.loc[data['Fare']<=0.7854,'Fare_cat']=0
data.loc[(data['Fare']>0.7854)&(data['Fare']<=10.5),'Fare_cat']=1
data.loc[(data['Fare']>10.5)&(data['Fare']<=21.679),'Fare_cat']=2
data.loc[(data['Fare']>21.679)&(data['Fare']<=39.688),'Fare_cat']=3
data.loc[(data['Fare']>39.688)&(data['Fare']<=512.329),'Fare_cat']=4
sns.factorplot('Fare_cat','Survived',data=data,hue='Sex')
plt.show()
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

data.head(2)
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='PiYG',linewidths=0.2,annot_kws={'size':8})
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()
data.head(2)

from sklearn.linear_model import LogisticRegression 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.cross_validation import train_test_split 
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print('The accuracy is for Logistic Regression is ',metrics.accuracy_score(prediction1,test_Y))
model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction2,test_Y))
model=RandomForestClassifier()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction3,test_Y))
model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction4,test_Y))
from sklearn.model_selection import cross_val_predict 
f,ax=plt.subplots(2,2,figsize=(10,8))
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for KNN')
plt.show()


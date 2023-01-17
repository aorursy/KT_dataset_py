#data analysing n wrangling
import pandas as pd 
import numpy as np
import random as rnd

#
#visualising
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#ml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import os
#reading csv dataset file
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df,test_df]
train_df.columns.values
train_df.head()
test_df_t=test_df


#test_df.head()
x=np.array([1,2,4,5,7,8,99])
z={pid:x for pid in test_df['PassengerId']}
dff=pd.DataFrame(z)

x
#dff['PassengerId']=test_df['PassengerId']
dff
test_df_t.head()

train_df.describe()
#'''
train_df.info()
print('-'*50)
test_df.info()
train_df.describe(include=['O'])        #columns with non numeric values
#'''
#correlate > complete >correcting >creating
train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)
print(train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean())
print('\n' , train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False))
print('\n',train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False))
#print('\n',train_df[['Age','Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived',ascending=False))
    #so for age we will use age grouping featuring extraction
#use plotting for Age
print('\n',train_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',ascending=False))
print('\n',train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',ascending=False))
#print('\n',train_df[['Cabin','Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived',ascending=False))
 #so we can remove cabin feature as it doesnt give much insight
#print('\n',train_df[['Fare','Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived',ascending=False))

g= sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins=30)
#train_df[['Pclass','Survived']].groupby(['Pclass']).describe()
grid=sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=30)
grid.add_legend();
#thus pclass 1 survived more over pclass3
grid2=sns.FacetGrid(train_df,col='Survived',row='Sex',size=2.2,aspect=1.6)
grid2.map(plt.hist,'Age',alpha=.5,bins=30)
grid2.add_legend();
#thus females have more chances of survival specially for age group 18 -40
grid3=sns.FacetGrid(train_df,row='Embarked',size=2.2,aspect=1.6)
grid3.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')

#grid3.map(sns.barplot,'Pclass','Survived','Sex',palette='deep')

grid3.add_legend()
grid4=sns.FacetGrid(train_df,row='Embarked',col='Survived',size=2.2,aspect=1.6)
grid4.map(sns.barplot,'Sex','Fare',palette='deep')
grid4.add_legend()

#Wrangling

#dropping unnecessary cols
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket','Cabin'],axis=1)
test_df=test_df.drop(['Ticket','Cabin'],axis=1)
combine=[test_df,train_df]
print('After',train_df.shape,test_df.shape)
#new feature extraction from existing data

combine[1].head()

for _ in combine:
    _['Title']=_.Name.str.extract('([A-Za-z]+)\.',expand=True)
    
pd.crosstab(train_df['Title'],train_df['Survived'])
#pd.crosstab(train_df['Title'],train_df['Sex'])
#combine[1].head()
train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived',ascending=False)

#train_df.Name.head(20)
for _ in combine:
    _['Title']= _['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Jonkheer','Major'],'Rare')
    _['Title']= _['Title'].replace(['Mlle','Ms',],'Miss')
    _['Title']= _['Title'].replace(['Mme'],'Mrs')
    _['Title']= _['Title'].replace(['Rev','Sir'],'Mr')
    
    #now  mapping titles to numeric values
    _['Title']= _['Title'].map({'Mr':1,'Miss':2,'Mrs':3,'Rare':4,'Master':5})
    #now mapping Sex to numperic
    _['Sex']=_['Sex'].map({'male':0,'female':1}).astype(int)
    _['Embarked']=_['Embarked'].replace({'C':0,'Q':1,'S':24})
    
    
train_df[['Title','Survived']].groupby('Title').describe()
print(train_df.columns.values,"\n",test_df.columns.values)
#now dropping some columns
test_df= test_df.drop(['Name','PassengerId'],axis=1)
train_df= train_df.drop(['Name','PassengerId'],axis=1)


train_df.head()

#fill missing Age values with median
train_df['Age']=train_df['Age'].fillna(train_df['Age'].median())    
test_df['Age']=test_df['Age'].fillna(test_df['Age'].median())    
#creating age band
#checking need of age band
train_df['AgeBand']=pd.cut(train_df['Age'],5)
train_df[['AgeBand','Survived']].groupby(['AgeBand']).mean()
#creating age band as feature engineering
for dta in [train_df,test_df]:
    dta.loc[dta['Age'] <=16,'Age']=0
    dta.loc[(dta['Age'] >16) & (dta['Age']<=30) ,'Age']=1
    dta.loc[(dta['Age'] >30 )& (dta['Age']<=45 ),'Age']=2
    dta.loc[(dta['Age'] >45) & (dta['Age']<=60 ),'Age']=3
    dta.loc[(dta['Age'] >60 ),'Age' ]=4
#drop age band4
train_df=train_df.drop(['AgeBand'],axis=1)
train_df.head()
#train_df[['Parch','Survived']].groupby('Parch').mean()
# creating new feature called family size
train_df['Familysize']=train_df['SibSp'] + train_df['Parch'] + 1
test_df['Familysize']=test_df['SibSp'] + test_df['Parch'] + 1


    
train_df[['Familysize','Survived']].groupby(['Familysize']).mean().sort_values(by='Survived',ascending=False)

#*** creating new features IsAlone
train_df['IsAlone']=0
test_df['IsAlone']=0

train_df.loc[train_df['Familysize']==1 ,'IsAlone']=1
test_df.loc[test_df['Familysize']==1 ,'IsAlone']=1
train_df[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean()
train_df.head()
train_df['Age*Class']=train_df['Age']*train_df['Pclass']
test_df['Age*Class']=test_df['Age']*test_df['Pclass']

#train_df.loc[:,['Age*Class','Age','Pclass','Survived']].head()
#train_df[['Age*Class','Survived']].groupby('Age*Class').mean().sort_values('Survived')
train_df.describe()
#Analysing fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#feature engineering on fare to fareband
train_df['Fareband']=pd.qcut(train_df['Fare'],4)
train_df[['Fareband','Survived']].groupby(['Fareband']).mean()
for dta in [train_df,test_df]:
    dta.loc[dta['Fare']<=8,'Fare']=0
    dta.loc[(dta['Fare'] > 8)  & (dta['Fare'] <=15),'Fare']=1
    dta.loc[(dta['Fare'] > 15 ) &(dta['Fare'] <=30),'Fare']=2
    dta.loc[(dta['Fare'] > 30 ) &(dta['Fare'] <=513),'Fare']=3


#train_df[['Fare','Survived']].groupby(['Fare']).mean()
train_df.head()
train_df['Title'].fillna(4, inplace=True)
test_df['Title'].fillna(4, inplace=True)

train_df['Embarked'].fillna(2, inplace=True)
train_df.shape
#model ,predict,n solve

#x_train=train_df.drop('Survived',axis=1)
x_train=train_df.drop(['Survived','Fareband'],axis=1)

y_train=train_df['Survived']
x_test=test_df
#x_train.columns.values
x_train.head()
x1_train=x_train[:round(len(x_train)*8.5/10)]
x2_train=x_train[len(x1_train):]
print(x1_train.shape,x2_train.shape,y_train.shape,x_test.shape )

#Logistic Regression
logreg=LogisticRegression()
logreg.fit(x_train , y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log


z={'PassengerId':list(test_df_t['PassengerId']) , 'Survived':list(y_pred)}
pd.DataFrame(z)
pd.DataFrame(pd.DataFrame(z)).to_csv('pred_Logistic_Regression.csv',sep=',',index=False)
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc

#since svc predicted best
l2=list(test_df_t['PassengerId'])
z={'PassengerId':list(test_df_t['PassengerId']) , 'Survived':list(y_pred)}
pd.DataFrame(z)
pd.DataFrame(pd.DataFrame(z)).to_csv('predSvc.csv',sep=',',index=False)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn


z={'PassengerId':list(test_df_t['PassengerId']) , 'Survived':list(y_pred)}
pd.DataFrame(z)
pd.DataFrame(pd.DataFrame(z)).to_csv('pred_knn.csv',sep=',',index=False)
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian


z={'PassengerId':list(test_df_t['PassengerId']) , 'Survived':list(y_pred)}
pd.DataFrame(z)
pd.DataFrame(pd.DataFrame(z)).to_csv('pred_naive_bayes.csv',sep=',',index=False)
# Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron


z={'PassengerId':list(test_df_t['PassengerId']) , 'Survived':list(y_pred)}
pd.DataFrame(z)
pd.DataFrame(pd.DataFrame(z)).to_csv('pred_perceptron.csv',sep=',',index=False)

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
acc_linear_svc


z={'PassengerId':list(test_df_t['PassengerId']) , 'Survived':list(y_pred)}
pd.DataFrame(z)
pd.DataFrame(pd.DataFrame(z)).to_csv('pred_linear_svc.csv',sep=',',index=False)
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd


z={'PassengerId':list(test_df_t['PassengerId']) , 'Survived':list(y_pred)}
pd.DataFrame(z)
pd.DataFrame(pd.DataFrame(z)).to_csv('pred_stochastic gradient descent.csv',sep=',',index=False)
#print('Logistic Regression :' ,acc_log ,'\n linear svc :', acc_linear_svc , '\n SVM :' , acc_svc , '\n knn :' ,acc_knn , '\n Gaussian Naive Bayes :' , acc_gaussian ,'\n Perceptron :' , acc_perceptron)
print('Logistic Regression :' ,acc_log ,'\n linear svc :', acc_linear_svc , '\n SVM :' , acc_svc , '\n knn :' ,acc_knn , '\n Gaussian Naive Bayes :' , acc_gaussian ,'\n Perceptron :' , acc_perceptron)

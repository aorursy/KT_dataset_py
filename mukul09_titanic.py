
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
train_data.head()
train_data.describe()
train_data.info()
print("###############################")
test_data.info()
train_data=train_data.drop(['PassengerId','Name','Ticket'],axis=1)
test_data=test_data.drop(['Name','Ticket'],axis=1)
g=train_data.groupby(by='Embarked')['Survived'].count()
g
# fill the NAN values of Embarked with most occured value, which is 'S'
train_data['Embarked']=train_data['Embarked'].fillna('S')

sns.factorplot('Embarked','Survived', data=train_data, size=3, aspect=3)

fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(15,5))

#sns.factorplot('Embarked',data=train_data,kind='count',order=['S','C','Q'],ax=axis1)
#sns.factorplot('Survived',hue='Embarked',data=train_data,order=[0,1],kind='count',ax=axis2)
sns.countplot('Embarked',data=train_data,ax=axis1)
sns.countplot('Survived',hue='Sex',data=train_data,ax=axis2)

embark_perc = train_data[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

embark_dummy_train=pd.get_dummies(train_data['Embarked'])
#embark_dummy_train.drop(['S'],axis=1,inplace=True)

embark_dummy_test=pd.get_dummies(test_data['Embarked'])
#embark_dummy_test.drop(['S'],axis=1,inplace=True)

train_data=train_data.join(embark_dummy_train)
test_data=test_data.join(embark_dummy_test)

train_data.drop(['Embarked'], axis=1,inplace=True)
test_data.drop(['Embarked'], axis=1,inplace=True)

#train_data.head()
#Fare
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

train_data['Fare']=train_data['Fare'].astype(int)
test_data['Fare']=test_data['Fare'].astype(int)

fare_survived=train_data['Fare'][train_data['Survived']==1]
fare_not_survived=train_data['Fare'][train_data['Survived']==0]

average_fare=pd.DataFrame(data=[fare_survived.mean(), fare_not_survived.mean()])
std_fare=pd.DataFrame(data=[fare_survived.std(), fare_not_survived.std()])
sns.set_style('whitegrid')

train_data['Fare'].plot(kind='hist',figsize=(15,5),bins=100, xlim=(0,100),edgecolor='black')


average_fare.index.names=std_fare.index.names=['Survived']
average_fare.plot(yerr=std_fare, kind='bar', legend=False)

#Age

fig, (axis1, axis2)=plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age Value -Titanic')
axis2.set_title('New Age Values - Titanic')
#axis3.set_title('Original Age Value -Titanic')
#axis4.set_title('New Age Values - Titanic')

average_age_train=train_data['Age'].mean()
std_age_train=train_data['Age'].std()
count_nan_age_train=train_data['Age'].isnull().sum()
#count_nan_age_train=train_data['Age'].isnull().value_counts()

average_age_test=test_data['Age'].mean()
std_age_test=test_data['Age'].std()
count_nan_age_test=test_data['Age'].isnull().sum()

rand1=np.random.randint(average_age_train-std_age_train, average_age_train+std_age_train)
rand2=np.random.randint(average_age_test-std_age_test, average_age_test+std_age_test)

train_data['Age'].dropna().astype(int).plot(kind='hist', bins=50, ax=axis1)

train_data['Age'][np.isnan(train_data['Age'])]=rand1
test_data['Age'][np.isnan(test_data['Age'])]=rand2

train_data['Age']=train_data['Age'].astype(int)
test_data['Age']=test_data['Age'].astype(int)

train_data['Age'].hist(bins=50, ax=axis2)

facet=sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

fig,axis1=plt.subplots(1,1,figsize=(18,4))
average_age=train_data[['Age','Survived']].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age',y='Survived', data=average_age)
#Cabin
print(train_data['Cabin'].isnull().value_counts())
train_data.drop(['Cabin'],axis=1, inplace=True)
test_data.drop(['Cabin'], axis=1, inplace=True)
# Family

train_data['Family']=train_data['Parch']+train_data['SibSp']
train_data['Family'].loc[train_data['Family']>0]=1
train_data['Family'].loc[train_data['Family']==0]=0

test_data['Family']=test_data['Parch']+test_data['SibSp']
test_data['Family'].loc[test_data['Family']>0]=1
test_data['Family'].loc[test_data['Family']==0]=0

train_data = train_data.drop(['SibSp','Parch'], axis=1)
test_data    = test_data.drop(['SibSp','Parch'], axis=1)

fig,(axis1,axis2)=plt.subplots(1,2, figsize=(15,4))

sns.countplot(x='Family', data=train_data,ax=axis1,order=[1,0])
axis1.set_xticklabels(['With Family', 'Alone'])

#sns.barplot(x='Family', y='Survived',data=train_data, ax=axis2, order=[1,0])
family_perc = train_data[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)
axis2.set_xticklabels(['With Family', 'Alone'])

# Sex
def get_person(passenger):
    age,sex=passenger
    return 'child' if age<16 else sex

train_data['Person']=train_data[['Age', 'Sex']].apply(get_person,axis=1)
test_data['Person']=test_data[['Age','Sex']].apply(get_person,axis=1)

train_data.drop(['Sex'],axis=1,inplace=True)
test_data.drop(['Sex'],axis=1,inplace=True)

# create dummy variables

person_dummies_train=pd.get_dummies(train_data['Person'])
person_dummies_train.columns=['Child','Female','Male']
#person_dummies_train.drop(['Male'],axis=1,inplace=True)

person_dummies_test=pd.get_dummies(test_data['Person'])
person_dummies_test.columns=['Child','Female','Male']
#person_dummies_test.drop(['Male'],axis=1,inplace=True)

train_data=train_data.join(person_dummies_train)
test_data=test_data.join(person_dummies_test)

fig,(axis1, axis2)=plt.subplots(1,2,figsize=(15,4))

sns.countplot(data=train_data, x='Person', ax=axis1)

person_prec=train_data[['Person','Survived']].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_prec,ax=axis2)

train_data.drop(['Person'],axis=1,inplace=True)
test_data.drop(['Person'],axis=1,inplace=True)

#Pclass
#sns.factorplot('Pclass',data=train_data,kind='count',order=[1,2,3])
sns.factorplot(x='Pclass',y='Survived',data=train_data,order=[1,2,3],size=5)

pclass_dummy_train=pd.get_dummies(train_data['Pclass'])
pclass_dummy_train.columns=['class1','class2','class3']
#pclass_dummy_train.drop(['class3'],axis=1,inplace=True)

pclass_dummy_test=pd.get_dummies(test_data['Pclass'])
pclass_dummy_test.columns=['class1','class2','class3']
#pclass_dummy_test.drop(['class3'],axis=1,inplace=True)

train_data.drop(['Pclass'],axis=1,inplace=True)
test_data.drop(['Pclass'],axis=1,inplace=True)

train_data=train_data.join(pclass_dummy_train)
test_data=test_data.join(pclass_dummy_test)
train_data.head()
test_data.head()
X_train=train_data.drop(['Survived'],axis=1)
y_train=train_data.Survived
X_test=test_data.drop(['PassengerId'],axis=1)
X_test.head()
# logistic Regression
logreg=LogisticRegression()
logreg.fit(X_train, y_train)
predict=logreg.predict(X_test)
logreg.score(X_train, y_train)
# Support vector Machine

svc=SVC()
svc.fit(X_train,y_train)
predict=svc.predict(X_test)
svc.score(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
predict=random_forest.predict(X_test)
#print(predict)
random_forest.score(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predict=knn.predict(X_test)
knn.score(X_train, y_train)
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)
predict=gnb.predict(X_test)
gnb.score(X_train, y_train)

# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = pd.DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df
submission=pd.DataFrame({
    "PassengerId":test_data['PassengerId'],
    "Survived": predict
})
submission.to_csv('titanic.csv', index=False)

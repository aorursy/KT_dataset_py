import os
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')
os.getcwd()
os.chdir("../input")
os.listdir()
import pandas as pd
df=pd.read_csv('train.csv')

X_test =pd.read_csv('test.csv')

Y_test = pd.read_csv('gender_submission.csv')
PassengerId =X_test['PassengerId']
PassengerId.head()
print('X_train')

df.info()

print('-'*70)

print('X_test')

X_test.info()
df.describe()
f,ax=plt.subplots(1,2,figsize=(18,8))

df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=df,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
df['Initial']=0

for i in df.Name:

    df['Initial']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
X_test['Initial']=0

for i in X_test.Name:

    X_test['Initial']=X_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
df['Initial'].unique()

X_test['Initial'].unique()
pd.crosstab(df.Initial,df.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex
df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

df['Pclass'].replace([1,2,3],['first','second','third'],inplace=True)

X_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mrs'],inplace=True)

X_test['Pclass'].replace([1,2,3],['first','second','third'],inplace=True)
pd.crosstab(df.Initial,df.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex





pd.crosstab(X_test.Initial,X_test.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex
df.Initial.unique()
sns.factorplot('Pclass','Survived',col='Initial',data=df)

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=df,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
df.groupby('Initial')['Age'].mean()

X_test.groupby('Initial')['Age'].mean()
df.loc[(df.Age.isnull())&(df.Initial=='Master'),'Age']=5

df.loc[(df.Age.isnull())&(df.Initial=='Miss'),'Age']=22

df.loc[(df.Age.isnull())&(df.Initial=='Mr'),'Age']=33

df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),'Age']=36

df.loc[(df.Age.isnull())&(df.Initial=='other'),'Age']=46
X_test.loc[(X_test.Age.isnull())&(X_test.Initial=='Master'),'Age']=5

X_test.loc[(X_test.Age.isnull())&(X_test.Initial=='Miss'),'Age']=22

X_test.loc[(X_test.Age.isnull())&(X_test.Initial=='Mr'),'Age']=33

X_test.loc[(X_test.Age.isnull())&(X_test.Initial=='Mrs'),'Age']=36

X_test.loc[(X_test.Age.isnull())&(X_test.Initial=='other'),'Age']=46
df.Initial.isnull().any()

X_test.Initial.isnull().any()
print('X_train')

df.info()

print('-'*70)

print('X_test')

X_test.info()
pd.crosstab([df.Embarked,df.Pclass],[df.Sex,df.Survived],margins=True).style.background_gradient(cmap='summer_r')
df.loc[(df.Embarked.isnull())]

X_test.loc[(X_test.Embarked.isnull())]
df.loc[61]
df["Embarked"].fillna("S",inplace=True)

X_test["Embarked"].fillna("S",inplace=True)
df.Embarked.value_counts()
df.keys()
df.tail()
df['FamilySize'] = df['Parch'] + df['SibSp']

X_test['FamilySize'] = df['Parch'] + df['SibSp']
data=df.drop(['PassengerId','Name', 'Cabin','Ticket','Parch','SibSp'],axis=1,)

X_test=X_test.drop(['PassengerId','Name', 'Cabin','Ticket','Parch','SibSp'],axis=1,)
data.head()
data.isnull().sum()



X_test.isnull().sum()
X_test.loc[(X_test.Fare.isnull())]
X_test["Fare"].fillna(X_test.Fare.mean(),inplace=True)
data.head()
data.FamilySize.unique()
data.FamilySize.value_counts().to_frame().style.background_gradient(cmap='summer')
data= pd.get_dummies(data)

X_test=pd.get_dummies(X_test)
data["Age"]=((data["Age"]-data["Age"].min())/(data["Age"].max()-data["Age"].min()))

data["Fare"]=((data["Fare"]-data["Fare"].min())/(data["Fare"].max()-data["Fare"].min()))

data["FamilySize"]=((data["FamilySize"]-data["FamilySize"].min())/(data["FamilySize"].max()-data["FamilySize"].min()))
X_test["Age"]=((X_test["Age"]-X_test["Age"].min())/(X_test["Age"].max()-X_test["Age"].min()))

X_test["Fare"]=((X_test["Fare"]-X_test["Fare"].min())/(X_test["Fare"].max()-X_test["Fare"].min()))

X_test["FamilySize"]=((X_test["FamilySize"]-X_test["FamilySize"].min())/(X_test["FamilySize"].max()-X_test["FamilySize"].min()))
data.head()

X_test.head()
Y_train = data.pop('Survived')
X_train=data
X_test.head()
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, LogisticRegressionCV, PassiveAggressiveClassifier, RidgeClassifierCV, Lasso, Ridge

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, KNeighborsClassifier 

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.svm import LinearSVC, SVC

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.neural_network import MLPClassifier, MLPRegressor   

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
X_train.info()

print('-'*70)

X_test.info()
Y_test= Y_test.pop('Survived')
model= [LogisticRegression(), KNeighborsClassifier(), Perceptron(), SGDClassifier(), LogisticRegressionCV(), PassiveAggressiveClassifier(), RidgeClassifierCV(), Lasso(), Ridge(), KNeighborsRegressor(), KNeighborsClassifier() , BernoulliNB() ,GaussianNB() , LinearSVC(), SVC() , DecisionTreeClassifier(), DecisionTreeRegressor(), MLPClassifier() ,MLPRegressor() , AdaBoostClassifier(), BaggingClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]

name = ['LogisticRegression','KNeighborsClassifier','Perceptron','SGDClassifier','LogisticRegressionCV','PassiveAggressiveClassifier','RidgeClassifierCV','Lasso','Ridge','KNeighborsRegressor','KNeighborsClassifier','BernoulliNB','GaussianNB','LinearSVC','SVC','DecisionTreeClassifier','DecisionTreeRegressor','MLPClassifier','MLPRegressor','AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','RandomForestClassifier']

SCORE= []

TESTING=[]

for ku in model:

    #ku will be replaced with each model like as first one is LogisticRegression()

    algorithm = ku.fit(X_train,Y_train)

    print(ku)

    #now 'algorithm' will be fitted by API with above line and next line will check score with data training and testing

    print('training set accuracy: {:.2f}'.format(algorithm.score(X_train,Y_train)))

    print('test set accuracy: {:.2f}'.format(algorithm.score(X_test,Y_test)))

    print('---'*20)

    #Now we are making a dataframe where by each loop the dataframe is added by SCORE,TESTING

    SCORE.append(algorithm.score(X_train,Y_train))

    TESTING.append(algorithm.score(X_test,Y_test))

models_dataframe=pd.DataFrame({'training score':SCORE,'testing score':TESTING},index=name)

models_dataframe
models_dataframe['training score'].plot.barh(width=0.8)

plt.title('Average training Accuracy')

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.show()
asendingtraining = models_dataframe.sort_values(by='training score', ascending=False)

asendingtraining 
asendingtraining['training score'].plot.barh(width=0.8)

plt.title('Average training Accuracy')

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.show()
ascendingtesting= models_dataframe.sort_values(by='testing score', ascending=True)

ascendingtesting
ascendingtesting['testing score'].plot.barh(width=0.8)

plt.title('Average testing Accuracy')

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.show()
model = LogisticRegression()

model.fit(X_train,Y_train)

prediction=model.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": prediction

    })
submission.head()
model=BernoulliNB().fit(X_train,Y_train)

prediction= model.predict(X_test)

gender_submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": prediction

    })
gender_submission.head()
#don't know how to upload this file to server of kaggle using BigQuery

#gender_submission.to_csv('C:\\Users\\username\\Desktop\\ML Projects\submission\gender_submission.csv',index=False)
os.listdir()
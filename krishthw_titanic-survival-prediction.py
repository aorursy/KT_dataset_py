import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# This helps not to see warnings

import warnings

warnings.filterwarnings("ignore")
df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')
combine=[df_train,df_test] # list containing whole data

print('df_train size', df_train.shape,'-- df_test size', df_test.shape)
df_train.head()
df_train.describe(include="all")
df_test.describe(include='all')
for df in combine:

    df.drop(['Ticket'],axis=1,inplace=True)

    

print('df_train size', df_train.shape,'-- df_test size', df_test.shape)
for df in combine:

    df['Family_Counts']=df['SibSp']+df['Parch']+1 # combine sibsp and Parch, value 1 represent traveled alone passengers

    df['Alone']=0 # Assign 0 to all entries

    df.loc[df['Family_Counts']==1,'Alone']=1 # Assign 1 to passengers travelled alone.

    df.drop(['SibSp','Parch','Family_Counts'],axis=1,inplace=True) # drop 'SibSp','Parch','Family_Counts' in favor of 'Alone'
for df in combine:

    df_train['Embarked'].fillna(df_train['Embarked'].mode()[0],inplace=True) # fill missing values in embarked with it's mode.

    df['Embarked'].replace(to_replace=['S','C','Q'],value=[1,2,3],inplace=True) # assigned categorical variables to both train and test set
df_test['Fare'].fillna(df_test['Fare'].mean(),inplace=True) 
bins1=[-1,8,15,31,513]

labels1=['F1','F2','F3','F4']

for df in combine:

    df['FareGroup']=pd.cut(df["Fare"],bins1,labels=labels1)

    df['FareGroup'].replace(to_replace=['F1','F2','F3','F4'],value=[1,2,3,4],inplace=True)
for df in combine:

    df['RecordedCabin']=df['Cabin'].notnull().astype('int64') # extract notnull values changed type to int64

    df.drop(['Cabin'],axis=1,inplace=True)
for df in combine:

    df['Sex'].replace(to_replace=['male','female'],value=[0,1],inplace=True)
df_train.head()
df_test.head()
# gives number of not NAN values of each columns

print(df_train.info())

print('____________________')

print(df_test.info())
for df in combine:

    df['Title']=df.Name.str.extract(' ([A-Za-z]+)\.', expand=False) 



    df['Title']=df['Title'].replace(['Mlle', 'Ms'],'Miss')

    df['Title']=df['Title'].replace(['Mrs','Mme'],'Mrs')

    df['Title']=df['Title'].replace(['Don','Rev','Dr','Col', 'Capt','Jonkheer','Major'],'Rare')

    df['Title']=df['Title'].replace(['Lady', 'Sir','Countess'],'Royal')



    df['Title'].replace(to_replace=['Master','Miss','Mr','Mrs','Rare','Royal'],value=[1,2,3,4,5,6],inplace=True)

    df.drop(['Name'],axis=1,inplace=True)
df_train.corr()
plt.figure(figsize=(20,15))

for i, col in enumerate(df_train.columns):

    plt.subplot(3,4,i+1)

    sns.boxplot(x=col,y='Age',data=df_train)
group1=df_train[['Pclass','Title','RecordedCabin','Age']].groupby(['Pclass','Title','RecordedCabin'],as_index=False).mean()

group1.head(10)
for df in combine:

    df['Age'].fillna(df.groupby(['Pclass','Title','RecordedCabin']).transform('mean').Age, inplace=True)
bins2=[0,2,10,20,30,60,100]

labels2 = ['baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

for df in combine:

    df['AgeGroup'] = pd.cut(df["Age"], bins2, labels = labels2)

    df['AgeGroup'].replace(to_replace=['baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'],value=[1,2,3,4,5,6],inplace=True)

    df.drop(['Age','Fare','Title'],axis=1,inplace=True)
df_train.head()
plt.figure(figsize=(10,5))

sns.heatmap(df_train.corr(),annot=True,vmax = .9)
fig1,axes=plt.subplots(4,2,figsize=(15,15))

g1=df_train.groupby(['Pclass', 'Survived']).size().unstack().plot.bar(stacked=True,ax=axes[0,0])

g2=df_train.groupby(['Sex', 'Survived']).size().unstack().plot.bar(stacked=True,ax=axes[0,1])

g3=df_train.groupby(['AgeGroup', 'Survived']).size().unstack().plot.bar(stacked=True,ax=axes[1,0])

g4=df_train.groupby(['FareGroup', 'Survived']).size().unstack().plot.bar(stacked=True,ax=axes[1,1])

g5=df_train.groupby(['Embarked', 'Survived']).size().unstack().plot.bar(stacked=True,ax=axes[2,0])

g6=df_train.groupby(['Alone', 'Survived']).size().unstack().plot.bar(stacked=True,ax=axes[2,1])

g7=df_train.groupby(['RecordedCabin', 'Survived']).size().unstack().plot.bar(stacked=True,ax=axes[3,0])

# Training set

X=df_train[['Pclass','Sex','Embarked','Alone','RecordedCabin','AgeGroup','FareGroup']]

Y=df_train['Survived']

print(X.shape,Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)

print ('X_Train set:', X_train.shape,  Y_train.shape)

print ('y_Test set:', X_test.shape,  Y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

K = 10

mean_acc = np.zeros((K-1)) # create a array of zeroes of shape Ks-1 i.e 9



for n in range(1,K): # The range() function returns a sequence of numbers, starting from 0 by default,

                      #.... and increments by 1 (by default), and ends at a specified number.

    #Train Model and Predict  

    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train) #train the model for different k

    Yhat=knn.predict(X_test) # predict

    mean_acc[n-1] = metrics.accuracy_score(Y_test, Yhat)



print(mean_acc)

plt.plot(range(1,K),mean_acc,'r')

plt.ylabel('Accuracy ')

plt.xlabel('Number of Neighbours (K)')

plt.show()

print('The best accuracy is with k =', mean_acc.argmax()+1)
KNN = KNeighborsClassifier(n_neighbors = 4).fit(X_train,Y_train)

KNNhat=KNN.predict(X_test)

KNN_ac=round(metrics.accuracy_score(Y_test, KNNhat)*100,2)

KNN_ac
from sklearn import svm

SVM=svm.SVC(kernel='rbf').fit(X_train,Y_train)

SVMhat=SVM.predict(X_test)

SVM_ac=round(metrics.accuracy_score(Y_test, SVMhat)*100,2)

SVM_ac
from sklearn.tree import DecisionTreeClassifier

DTC=DecisionTreeClassifier(criterion="entropy",max_depth=4,random_state=4).fit(X_train,Y_train)

DTChat=DTC.predict(X_test)

DTC_ac=round(metrics.accuracy_score(Y_test, DTChat)*100,2)

DTC_ac
from sklearn.ensemble import RandomForestClassifier

RF= RandomForestClassifier().fit(X_train,Y_train)

RFhat=RF.predict(X_test)

RF_ac=round(metrics.accuracy_score(Y_test, RFhat)*100,2)

RF_ac
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression().fit(X_train,Y_train)

LRhat=LR.predict(X_test)

LR_ac=round(metrics.accuracy_score(Y_test, LRhat)*100,2)

LR_ac
from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier().fit(X_train,Y_train)

SGDhat=SGD.predict(X_test)

SGD_ac=round(metrics.accuracy_score(Y_test, SGDhat)*100,2)

SGD_ac
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB().fit(X_train,Y_train)

GNBhat=GNB.predict(X_test)

GNB_ac=round(metrics.accuracy_score(Y_test, GNBhat)*100,2)

GNB_ac
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier().fit(X_train,Y_train)

GBChat=GBC.predict(X_test)

GBC_ac=round(metrics.accuracy_score(Y_test, GBChat)*100,2)

GBC_ac
list1=[KNNhat,SVMhat,DTChat,RFhat,LRhat,SGDhat,GBChat,GNBhat]

list2=['KNN','SVM','Decision Tree','RandomForest','LogisticRegression','Stochastic Gradient','Gradient Boosting','Gaussian Naive']

accuracy=[]

# Calculating accuracy for each model

for ii in list1:

    accuracy.append(round(metrics.accuracy_score(Y_test,ii)*100,2))

    

# Results in a data frame

Results=pd.DataFrame(list(zip(list2,accuracy)),columns=['Algorithm','Accuracy'])  

Results
Z=df_test[['Pclass','Sex','Embarked','Alone','RecordedCabin','AgeGroup','FareGroup']]

test_prediction=KNN.predict(Z)

passengerid=df_test['PassengerId']

submission = pd.DataFrame({"PassengerId": passengerid,"Survived": test_prediction})

submission.to_csv('Titanic_Submission.csv', index=False)

submission
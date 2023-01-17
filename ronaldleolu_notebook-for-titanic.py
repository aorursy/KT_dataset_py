import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
#missing values

print(train_df.isnull().sum())

print("----------------------------------------------------")

print(test_df.isnull().sum())
#fill item "Embarked" in train dataframe

print(train_df.Embarked.value_counts())

train_df[train_df['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train_df)
#fillna

train_df['Embarked'] = train_df['Embarked'].fillna('C')
#fill item 'Fare' in test dataframe

print(test_df['Fare'].describe())

test_df[test_df['Fare'].isnull()]
fare_mean = test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')]['Fare'].mean()

fare_median = test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')]['Fare'].median()

print(test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')]['Fare'].describe())

print('Mean:%d'%fare_mean)

print('Median:%d'%fare_median)
#fillna with median

test_df['Fare'] = test_df['Fare'].fillna(fare_median)
import re

def get_title(name):

    pattern=re.compile('^.*, (.*?)\..*$',re.S)

    pear=re.findall(pattern,name)

    return pear[0]



train_df['Title'] = train_df['Name'].apply(get_title)

test_df['Title'] = test_df['Name'].apply(get_title)



print(train_df.Title.value_counts())

print("--------------------------------------------------")

print(test_df.Title.value_counts())
def group_title(title):

    

    group_men=['Mr','Don','Sir','Master']# return 1

    group_younger_women=['Mlle','Miss','Ms']# return 2

    group_elder_women=['Lady','Mrs','Mme','Dona'] # return 3

    group_army=['Major','Col','Capt']# return 4

    # else return 5 

    

    if title in group_men:

        return 1

    elif title in group_younger_women:

        return 2

    elif title in group_elder_women:

        return 3

    elif title in group_army:

        return 4

    else:

        return 5

    

train_df['TitGroup'] = train_df['Title'].apply(group_title)

test_df['TitGroup'] = test_df['Title'].apply(group_title)
datasets = [train_df,test_df]



for dataset in datasets:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

    dataset['Age'] = pd.cut(dataset['Age'],5,labels=[i for i in range(1,6)])
def class_gender(sex):

    if sex=='male':

        return 1

    else:

        return 2

    

def class_embarked(emb):

    if emb=='S':

        return 1

    if emb=='C':

        return 2

    else:

        return 3



for dataset in datasets:

    #create new column family

    dataset['Family'] = dataset['SibSp']+dataset['Parch']

    #standardization z-score

    dataset['StdFare'] = dataset['Fare'].apply(lambda x:(x-dataset['Fare'].mean())/dataset['Fare'].std())

    #label

    dataset['Sex'] = dataset['Sex'].map(class_gender)

    dataset['Embarked'] = dataset['Embarked'].map(class_embarked)    

    

    

    
numeric_feature = ['Age','Family','StdFare']

def make_dataset(dataset):

    dummy_df = pd.get_dummies(dataset['Pclass'],prefix='Pclass')

    for item in ['Sex','Embarked','TitGroup']:

        new_df = pd.get_dummies(dataset[item],prefix=item)

        dummy_df = pd.concat([dummy_df,new_df],axis = 1)

        

    return pd.concat([dataset[numeric_feature],dummy_df],axis = 1)

    

X_train = make_dataset(train_df)

Y_train = train_df['Survived']

X_test = make_dataset(test_df)
from sklearn.cross_validation import train_test_split

#cross validation

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X_train,Y_train,test_size=0.3,random_state=0)
#Logistic Regression

def lr_classify(Xtrain,Xtest,Ytrain,Ytest):

    from sklearn.linear_model import LogisticRegression

    

    clf = LogisticRegression(C=1e5)

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print("LinearRegression score: {0}".format(score))
#knn

def knn_classify(Xtrain,Xtest,Ytrain,Ytest):

    from sklearn.neighbors import KNeighborsClassifier

    

    clf = KNeighborsClassifier(n_neighbors=10)

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print("Knn score: {0}".format(score))
#Random Forest

def rf_classify(Xtrain,Xtest,Ytrain,Ytest):

    from sklearn.ensemble import RandomForestClassifier

    

    clf = RandomForestClassifier(n_estimators=10)

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print("RandomForest score: {0}".format(score))
#NaiveBayes GaussianNB

def nb_classify(Xtrain,Xtest,Ytrain,Ytest):

    from sklearn.naive_bayes import GaussianNB

    

    clf = GaussianNB()

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print("GaussianNB score: {0}".format(score))
#SVM

def svm_classify(Xtrain,Xtest,Ytrain,Ytest):

    from sklearn import svm

    

    clf = svm.SVC()

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print("SVM score: {0}".format(score))
#xgboost

def xgb_classify(Xtrain,Xtest,Ytrain,Ytest):

    import xgboost as xgb

    

    clf = xgb.XGBClassifier()

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print("Xgboost score: {0}".format(score))
#Voting class

def voting_classify(Xtrain,Xtest,Ytrain,Ytest):

    from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier,RandomForestClassifier

    from sklearn.linear_model import LogisticRegression

    from sklearn.naive_bayes import GaussianNB

    

    clf1 = GradientBoostingClassifier(n_estimators=200)

    clf2 = RandomForestClassifier(random_state=0,n_estimators=50)

    clf3 = LogisticRegression(random_state=1)

    clf4 = GaussianNB()

    

    clf = VotingClassifier(estimators=[('gbdt',clf1),('rf',clf2),('lr',clf3),('nb',clf4)],voting='soft')

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print("Voting score: {0}".format(score))
def param_rf_n_estimate(Xtrain,Xtest,Ytrain,Ytest,n):

    from sklearn.ensemble import RandomForestClassifier

    

    clf = RandomForestClassifier(random_state=0,n_estimators=n)

    clf.fit(Xtrain,Ytrain)

    score = clf.score(Xtest,Ytest)

    print('When n is %d'%n)

    print("RandomForest score: {0}".format(score))

    

    

    

    
from sklearn.ensemble import RandomForestClassifier

    

clf = RandomForestClassifier(random_state=0,n_estimators=100)

clf.fit(Xtrain,Ytrain)

predictions = clf.predict(X_test)
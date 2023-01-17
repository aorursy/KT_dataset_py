#importing required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report
#loading the train data

train_df=pd.read_csv("../input/titanic/train.csv")

train_df.head()
train_df.shape
#loading the test data

test_df=pd.read_csv("../input/titanic/test.csv")

test_df.head()
test_df.shape
#concatinating train and test to get titanic dataset

titanic_df=pd.concat([train_df, test_df], sort = False)

titanic_df.shape
#retrieving first five records of titanic dataset

titanic_df.head()
#retrieving column names of the titanic dataset

titanic_df.columns
#information about the dataset

titanic_df.info()
#distribution of Age column values

sns.distplot(titanic_df['Age'])
#treating NULL values of Age column of the dataset

titanic_df['Age']=titanic_df['Age'].fillna(titanic_df['Age'].mean())
#treating NULL values of Embarked column of the dataset

titanic_df['Embarked']=titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
#dropping Cabin, PassengerId, Ticket and Fare columns

titanic_df=titanic_df.drop(columns=['Ticket', 'Fare', 'Cabin'])
#checking for NULL values 

titanic_df.isnull().sum()
#distribution of Age column values after treating missing values

sns.distplot(titanic_df['Age'])
#describing statistics of the numerical columns

titanic_df.describe()
#count of survivors

titanic_df['Survived'].value_counts()
#plotting count of survivors

survived=titanic_df['Survived'].map({0:'Not Survived',1:'Survived'})

sns.countplot(survived)
#plotting categorical features related to Survived target

cat_feats=['Pclass','Sex','Embarked']

plt.figure(figsize=(20,5))



for i in range(len(cat_feats)):

    plt.subplot(1,3,i+1)

    sns.countplot(titanic_df[cat_feats[i]],hue=survived,data=titanic_df)
#plotting survival probability as per Passenger class

plt=titanic_df[['Pclass','Survived']].groupby('Pclass').mean().Survived.plot(kind='bar',color=['r','g','y'])

plt.set_xlabel('Pclass')

plt.set_ylabel('Survival Probability')
#plotting survival probability as per Sex

plt=titanic_df[['Sex','Survived']].groupby('Sex').mean().Survived.plot(kind='bar',color=['r','g'])

plt.set_xlabel('Sex')

plt.set_ylabel('Survival Probability')
#plotting survival probability as per Embarked column

plt=titanic_df[['Embarked','Survived']].groupby('Embarked').mean().Survived.plot(kind='bar',color=['r','g','y'])

plt.set_xlabel('Embarked')

plt.set_ylabel('Survival Probability')
#plotting survival probability as per SibSp

plt=titanic_df[['SibSp','Survived']].groupby('SibSp').mean().Survived.plot(kind='bar',color=['r','g','y','b','c','m'])

plt.set_xlabel('Siblings_Spouse')

plt.set_ylabel('Survival Probability')
#plotting survival probability as per Parch

plt=titanic_df[['Parch','Survived']].groupby('Parch').mean().Survived.plot(kind='bar',color=['r','b','g','y','c'])

plt.set_xlabel('Parent_Children')

plt.set_ylabel('Survival Probability')
#adding new column Familsize and IsAlone

titanic_df['FamilySize']=titanic_df['SibSp']+titanic_df['Parch']+1

titanic_df['IsAlone'] = 0

titanic_df.loc[titanic_df['FamilySize']==1, 'IsAlone'] = 1

titanic_df.columns
#adding new column Title

titanic_df['Title']=titanic_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

titanic_df.columns
#distinct values in Title column

titanic_df['Title'].unique()
#replacing some Title values with others

titanic_df['Title']=titanic_df['Title'].replace(['Don','Dr','Mme','Ms','Major','Lady','Sir','Mlle','Col','Capt',

                                                 'Countess','Jonkheer','Dona'],'Others')
#plotting bar plot for Title column

titanic_df.Title.value_counts().plot(kind='bar')
#dropping Name, SibSp and Parch columns

titanic_df=titanic_df.drop(columns=['Name','SibSp','Parch'])

titanic_df.columns
#categorical values into numerical ones

le=LabelEncoder()

cols=['Sex','Embarked','Title']

for i in cols:

    titanic_df[i]=le.fit_transform(titanic_df[i])
#correlation matrix for columns

sns.heatmap(data=titanic_df.corr(),cmap='Blues',annot=True,linewidths=0.2)
#head values of the dataset reformed

titanic_df.head()
#transforming Age values in the range as other columns

data=[titanic_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

titanic_df.head()
#extracting training set from titanic dataset

train=titanic_df[titanic_df['Survived'].notna()]

train.info()
#extracting testing set from titanic dataset

test=titanic_df[titanic_df['Survived'].isna()]

test.info()
#extracting features and target set from titanic_df

feature_df=train.drop(['PassengerId','Survived'],axis=1)

target_df=train['Survived']
#importing classification model creation packages

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
#classification model creation using different classifiers

model_accuracy=[]

training_accuracy=[]

testing_accuracy=[]

cross_val_scores=[]

def classify(model,x,y):

    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.15)

    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    model_accuracy.append(round(accuracy_score(y_test,y_pred)*100,3))

    training_accuracy.append(round(model.score(x_train, y_train)*100,3))

    testing_accuracy.append(round(model.score(x_test, y_test)*100,3))

    score=cross_val_score(model,x,y,cv=5)

    cross_val_scores.append(np.mean(score)*100)
#fitting different classifier models

classifiers=[

    LogisticRegression(),

    SGDClassifier(max_iter=5,tol=None),

    KNeighborsClassifier(),

    GaussianNB(),

    SVC(kernel='linear'),

    DecisionTreeClassifier(max_depth=4),

    MLPClassifier(activation='logistic',max_iter=1500),

    RandomForestClassifier(n_estimators=100,max_depth=4),

    BaggingClassifier(n_estimators=50,base_estimator=KNeighborsClassifier()),

    AdaBoostClassifier(n_estimators=50,base_estimator=LogisticRegression()),

    GradientBoostingClassifier(learning_rate=0.01),

    XGBClassifier(n_estimators=50)]



for classifier in classifiers:

    classify(classifier,feature_df,target_df)
#results recorded in new dataframe

model_results=pd.DataFrame({

    'Model': ['Logistic Regression', 'SGD Classifier', 'KNN Classifier', 'Naive Bayes Classifier', 

              'Linear SVM Classifier', 'Decision Tree Classifier', 'MLP Classifier', 'Random Forest Classifier',

              'Bagging Classifier', 'Adaboost Classifier', 'Gradientboost Classifier','XGBoost Classifier'],

    'Accuracy': model_accuracy,

    'Training_Accuracy':training_accuracy,

    'Testing_Accuracy':testing_accuracy,

    'Cross_Validation_Score':cross_val_scores

})

result_df=model_results.sort_values(by='Accuracy', ascending=False)
#printing the accuracy result

result_df
#splitting for training and testing

x_train,x_test,y_train,y_test=train_test_split(feature_df, target_df, test_size=0.15)
#implementing RandomForestClassifier model

rfc_model=RandomForestClassifier(n_estimators=100,max_depth=4)

rfc_model.fit(x_train,y_train)
#predicting y for x_test part

pred=rfc_model.predict(x_test)
#classification report for RandomForestClassifier model

print(classification_report(y_test,pred))
#Predicting survival for test data

TestForPred=test.drop(['PassengerId', 'Survived'], axis = 1)
#predicting y values

y_pred=rfc_model.predict(TestForPred).astype(int)
PassengerId=test['PassengerId']
rfc_pred=pd.DataFrame({'PassengerId': PassengerId, 'Survived':y_pred })

rfc_pred.head()
rfc_pred.shape
#exporting results to csv file

rfc_pred.to_csv("Titanic_Submission.csv", index = False)
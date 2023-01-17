# data analysis libraries

import pandas as pd

import numpy as np

# visualization libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# ignore warnings library

import warnings

warnings.filterwarnings("ignore")
Train= pd.read_csv("../input/titanic/train.csv")

Test= pd.read_csv("../input/titanic/test.csv")
print(Train.isnull().sum())



print(Test.isnull().sum())
print("Train shape:", Train.shape)

print("Test shape:", Test.shape)
Train.info()
Test.info()
# printing first five lines of our training dataset

Train.head()
# printing first five lines of our testing dataset

Test.head()
# here we are calculating summary of our train dataset

Train.describe()
# Here we are calculating summary of our test dataset

Test.describe()
# ploting bar plot for sex vs survived

sns.barplot(x="Sex",y="Survived",data=Train)

# printing survival percentage of female

print("Percentage of females who survived:", 

      Train["Survived"][Train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

# printing survival percentage of male

print("Percentage of males who survived:", 

      Train["Survived"][Train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
# ploting bar plot for SibSp vs Survived

sns.barplot(x="SibSp",y="Survived",data= Train)

# printing survival percentage of sibsp

print("Percentage of Sibsp- 0 who survived:", 

      Train["Survived"][Train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of Sibsp- 1 who survived:", 

      Train["Survived"][Train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Sibsp- 2 who survived:", 

      Train["Survived"][Train["SibSp"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Sibsp- 3 who survived:", 

      Train["Survived"][Train["SibSp"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of Sibsp- 4 who survived:", 

      Train["Survived"][Train["SibSp"] == 4].value_counts(normalize = True)[1]*100)
# ploting bar plot for Pclass vs Survived

sns.barplot(x="Pclass",y="Survived",data= Train)

# printing survival percentage of Pclass

print("Percentage of pclass- 1 who survived:", 

      Train["Survived"][Train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass- 2 who survived:", 

      Train["Survived"][Train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass- 3 who survived:", 

      Train["Survived"][Train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
# ploting bar plot for Parch vs Survived

sns.barplot(x="Parch",y="Survived",data= Train)

# printing survival percentage of Parch

print("Percentage of parch- 0 who survived:", 

      Train["Survived"][Train["Parch"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of Parch- 1 who survived:", 

      Train["Survived"][Train["Parch"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Parch- 2 who survived:", 

      Train["Survived"][Train["Parch"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Parch- 3 who survived:", 

      Train["Survived"][Train["Parch"] == 3].value_counts(normalize = True)[1]*100)

#sort the ages into logical categories

Train["Age"] = Train["Age"].fillna(-0.5)

Test["Age"] = Test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

Train['AgeGroup'] = pd.cut(Train["Age"], bins, labels = labels)

Test['AgeGroup'] = pd.cut(Test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=Train)

plt.show()

# ploting bar plot for Embarked vs Survived

sns.barplot(x="Embarked",y="Survived",data= Train)

# printing survival percentage of Embarked

print("Survived :\n",Train[Train['Survived']==1]['Embarked'].value_counts())

print("Dead:\n",Train[Train['Survived']==0]['Embarked'].value_counts())



# We will first drop tha cabin column because there is not need of this in our prediction.

Train = Train.drop(["Cabin"],axis=1)

Test = Test.drop(["Cabin"],axis=1)
# We will also drop the Ticket column because there is not need of this in our prediction

Train = Train.drop(["Ticket"],axis=1)

Test = Test.drop(["Ticket"],axis=1)
#now we will fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")

southampton = Train[Train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = Train[Train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = Train[Train["Embarked"] == "Q"].shape[0]

print(queenstown)
# now we will replace missing value with Southampton

Train = Train.fillna({"Embarked": "S"})
# here we are combining our dataset

train_test_data = [Train,Test]

for dataset in train_test_data:

    dataset["Title"] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
Train["Title"].value_counts()
Test["Title"].value_counts()
# Map each of title groups to numerical values.

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }



for dataset in train_test_data:

    dataset['Title'] = dataset["Title"].map(title_mapping)
Test.head()
Train.head()
sns.barplot(x="Title",y="Survived",data= Train)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for x in range(len(Train["Age"])):

    if Train["Age"][x] == "Unknown":

        Train["Age"][x] = title_mapping[Train["Title"][x]]

        

for x in range(len(Test["Age"])):

    if Test["Age"][x] == "Unknown":

        Test["Age"][x] = title_mapping[Test["Title"][x]]
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

Train['AgeGroup'] = Train['AgeGroup'].map(age_mapping)

Test['AgeGroup'] = Test['AgeGroup'].map(age_mapping)



Train.head()



Train["AgeGroup"].fillna(Train.groupby("Title")["AgeGroup"].transform("median"), inplace= True)

Test["AgeGroup"].fillna(Test.groupby('Title')['AgeGroup'].transform("median"), inplace= True)
# here we are mapping sex feature into numerical values

sex_mapping = {"male": 0, "female": 1}

Train['Sex'] = Train['Sex'].map(sex_mapping)

Test['Sex'] = Test['Sex'].map(sex_mapping)



Train.head()
# here we are removing name column because it is not playing important role in doing prediction

Train = Train.drop(['Name'], axis = 1)

Test = Test.drop(['Name'], axis = 1)
# here we are mapping embarked feature into numerical values

embarked_mapping = {"S":1,"C":2,"Q":3}

Train["Embarked"] = Train["Embarked"].map(embarked_mapping)

Test["Embarked"] = Test["Embarked"].map(embarked_mapping)
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(Test["Fare"])):

    if pd.isnull(Test["Fare"][x]):

        pclass = Test["Pclass"][x] #Pclass = 3

        Test["Fare"][x] = round(Train[Train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

Train['FareBand'] = pd.qcut(Train['Fare'], 4, labels = [1, 2, 3, 4])

Test['FareBand'] = pd.qcut(Test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

Train = Train.drop(['Fare'], axis = 1)

Test = Test.drop(['Fare'], axis = 1)
Train.head()
Test.head()
Train.head()
Train.head()
Train = Train.drop(["Age"],axis=1)

Test = Test.drop(["Age"],axis=1)
Train.head()
Test.head()
x = Train.iloc[:, 2:10].values

y = Train.iloc[:, 1].values
x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train.shape)

print(x_test.shape)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression (solver='liblinear', random_state=0)

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
# here we are calculating accuracy rate of our model

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

acc_lg= accuracy_score(y_test, y_pred)

acc_lg
from sklearn.neighbors import KNeighborsClassifier

classifierr = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,

                     weights='uniform')

classifierr.fit(x_train,y_train)
y_pred = classifierr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

acc_knn = accuracy_score(y_test, y_pred)

acc_knn
from sklearn.svm import SVC

classifier1 = SVC(kernel="linear",random_state=0,C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='scale',

    max_iter=-1, probability=True,shrinking=True, tol=0.001,

    verbose=False)

classifier1.fit(x_train,y_train)

y_pred = classifier1.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

acc_svm = accuracy_score(y_test, y_pred)

acc_svm
from sklearn.svm import SVC

classifier2 = SVC(kernel="rbf",random_state=0,probability=True)

classifier2.fit(x_train,y_train)
y_pred = classifier2.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

print(cm)

acc_kernel = accuracy_score(y_test,y_pred)

acc_kernel
from sklearn.naive_bayes import GaussianNB

classifier3 = GaussianNB()

classifier3.fit(x_train, y_train)
y_pred = classifier3.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

print(cm)

acc_naive = accuracy_score(y_test,y_pred)

acc_naive
from sklearn.tree import DecisionTreeClassifier

classifier4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier4.fit(x_train, y_train)
y_pred = classifier4.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

print(cm)

acc_dt = accuracy_score(y_test,y_pred)

acc_dt
from sklearn.ensemble import RandomForestClassifier

classifier5 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier5.fit(x_train, y_train)
y_pred = classifier5.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

print(cm)

acc_rf = accuracy_score(y_test,y_pred)

acc_rf
models = pd.DataFrame({

    'Model': ["LOGISTIC REGRESSION","K NEAREST NEIGHBORS","SUPPORT VECTOR MACHINE","KERNEL SVM","NAIVE BAYES","DECISION TREE","RANDOM FOREST"],

    'Score': [acc_lg,acc_knn,acc_svm,acc_kernel,acc_naive,acc_dt,acc_rf

              ]})

models.sort_values(by='Score', ascending=False)
# importing libraries for plotting roc_curve

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
classifiers = [LogisticRegression(random_state=0),

               KNeighborsClassifier(),

               SVC(random_state=0,probability=True), 

               GaussianNB(), 

               DecisionTreeClassifier(random_state=0),

               RandomForestClassifier(random_state=0)]

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

for cls in classifiers:

    model = cls.fit(x_train, y_train)

    yproba = model.predict_proba(x_test)[:,1]

    

    fpr, tpr, _ = roc_curve(y_test,  yproba)

    auc = roc_auc_score(y_test, yproba)

    

    result_table = result_table.append({'classifiers':cls.__class__.__name__,

                                        'fpr':fpr, 

                                        'tpr':tpr, 

                                        'auc':auc}, ignore_index=True)

result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))



for i in result_table.index:

    plt.plot(result_table.loc[i]['fpr'], 

             result_table.loc[i]['tpr'], 

             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    

plt.plot([0,1], [0,1], color='orange', linestyle='--')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel("Flase Positive Rate", fontsize=15)



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel("True Positive Rate", fontsize=15)



plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop={'size':13}, loc='lower right')



plt.show()

#set ids as PassengerId and predict survival 

ids = Test['PassengerId']

predictions = classifier5.predict(Test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)

print(output)
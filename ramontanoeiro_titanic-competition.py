import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()
train_data = pd.read_csv("../input/titanic/train.csv")
train_data.head(10)
## PassengerId does not train, Cabin has lot's of missing values,

## Ticket is a string with many different values and Sex is a better feature than name.
train_data.drop(columns=['Name', 'Ticket', 'Cabin','PassengerId'], axis=1, inplace=True)
train_data['Sex'] = train_data['Sex'].map({'male':0,'female':1})
train_data['Age'].isnull().sum() ##Many missing values in the dataset
train_data['Embarked'].value_counts() ##S is more common
train_data['Embarked'].fillna(value='S',axis=0, inplace=True)
train_data['Embarked'] = train_data['Embarked'].map({'S':0,'C':1,'Q':2})
train_data.fillna(value=train_data['Age'].mean(), axis=0, inplace=True) ## let's fill with the mean
train_data['Embarked'].unique()
train_data.describe()
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
train_data[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean()
train_data[['Parch', 'Survived']].groupby('Parch', as_index=False).mean()
train_data['AgeBand'] = pd.cut(train_data['Age'], 9) ## Need to save in another column to discover the ageband, then we can replace in the age feature
train_data.head()
train_data[['AgeBand','Survived']].groupby('AgeBand', as_index=False).mean()
train_data.loc[ train_data['Age'] <= 18, 'Age'] = 0

train_data.loc[(train_data['Age'] > 18) & (train_data['Age'] <= 44), 'Age'] = 1

train_data.loc[(train_data['Age'] > 44) & (train_data['Age'] <= 53), 'Age'] = 2

train_data.loc[(train_data['Age'] > 53) & (train_data['Age'] <= 62), 'Age'] = 3

train_data.loc[ train_data['Age'] > 62, 'Age'] = 4
train_data.drop(labels='AgeBand', axis=1, inplace=True)
train_data['FareBand'] = pd.qcut(train_data['Fare'], 5)
train_data[['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean()
train_data.loc[ train_data['Fare'] <= 10, 'Fare'] = 0

train_data.loc[(train_data['Fare'] > 10) & (train_data['Fare'] <= 40), 'Fare'] = 1

train_data.loc[train_data['Fare'] > 40 , 'Fare'] = 2
train_data.drop(labels='FareBand', axis=1, inplace=True)
fig, corr = plt.subplots(figsize=(20,15))

sns.heatmap(train_data.corr(), annot=True)
test_data = pd.read_csv("../input/titanic/test.csv")
test_data.head(10)
test_data.drop(columns=['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male':0,'female':1})
test_data['Embarked'] = test_data['Embarked'].map({'S':0,'C':1,'Q':2})
test_data.describe() ##Missing one value on fare.
test_data['Fare'].fillna(value=test_data['Fare'].median(),axis=0, inplace=True) ## We replace with the median due to it's high variance on the data
test_data.fillna(value=test_data['Age'].mean(), axis=0, inplace=True)
test_data.loc[ test_data['Age'] <= 18, 'Age'] = 0

test_data.loc[(test_data['Age'] > 18) & (test_data['Age'] <= 44), 'Age'] = 1

test_data.loc[(test_data['Age'] > 44) & (test_data['Age'] <= 53), 'Age'] = 2

test_data.loc[(test_data['Age'] > 53) & (test_data['Age'] <= 62), 'Age'] = 3

test_data.loc[ test_data['Age'] > 62, 'Age'] = 4
test_data.loc[ test_data['Fare'] <= 10, 'Fare'] = 0

test_data.loc[(test_data['Fare'] > 10) & (test_data['Fare'] <= 40), 'Fare'] = 1

test_data.loc[test_data['Fare'] > 40 , 'Fare'] = 2
submission = pd.read_csv('../input/titanic/gender_submission.csv')
features = ["Pclass", "Sex", "Age","Fare","SibSp","Parch","Embarked"]

targets = ["Survived"]

X_train, X_test = train_data[features], test_data[features]

y_train, y_test = train_data[targets],submission[targets]
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
lr = LogisticRegression()

model = lr.fit(X_train,y_train)

predictions_lr = model.predict(X_test)
predictions_lr = model.predict(X_test)
print(pd.crosstab(predictions_lr,submission['Survived'],margins=True,rownames=['Previsto'],colnames=['         Real']))

print(metrics.classification_report(y_test,predictions_lr))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_lr})
output = output.to_csv("My_submission_Logistic.csv", index=False)
X_train.head(),X_test.head()
y_train.head(), y_test.head()
from sklearn.neighbors import KNeighborsClassifier
def KNN_1(neighbors):

    neighbors = range(1,neighbors+1,1)

    print("Para p = 1")

    for i in neighbors:

        KNN = KNeighborsClassifier(n_neighbors=i,p=1)

        model = KNN.fit(X_train,np.ravel(y_train))      

        results_KNN = model.predict(test_data[features])

        print("Para uma quantidade de vizinhos de:", i,"a precisão do modelo foi de ",round((model.score(X_train,y_train))*100,2),"%")
def KNN_2(neighbors):

    neighbors = range(1,neighbors+1,1)

    print("Para p = 2")

    for i in neighbors:

        KNN = KNeighborsClassifier(n_neighbors=i,p=2)

        model = KNN.fit(X_train,np.ravel(y_train))

        results_KNN = model.predict(test_data[features])

        print("Para uma quantidade de vizinhos de:", i,"a precisão do modelo foi de ",round((model.score(X_train,y_train))*100,2),"%")
def choosen_KNN(neighbors,p):

    KNN = KNeighborsClassifier(n_neighbors=neighbors,p=p)

    model = KNN.fit(X_train,np.ravel(y_train))

    results_KNN = model.predict(test_data[features])

    print("Precisão ",round((model.score(X_train,y_train))*100,2),"% \n\n")

    output_KNN = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': results_KNN})

    output_KNN.to_csv('my_submission_KNN.csv', index=False)

    print(pd.crosstab(results_KNN,submission['Survived'],margins=True,rownames=['Previsto'],colnames=['         Real']))
KNN_1(20)
KNN_2(20)
choosen_KNN(7,2)
from sklearn.tree import DecisionTreeClassifier
def DTC(depth):

    depth = range(1,depth+1,1)

    for i in depth:

        DTC = DecisionTreeClassifier(max_depth=i)

        model = DTC.fit(X_train, y_train)

        predictions_DTC = model.predict(X_test)

        print("Para uma profundidade de:", i,"a precisão do modelo foi de ",round((model.score(X_train,y_train))*100,2),"%")
def choosen_DTC(depth):

    

    DTC = DecisionTreeClassifier(max_depth=depth)

    model = DTC.fit(X_train, y_train)

    predictions_DTC = model.predict(X_test)

    

    print (pd.crosstab(predictions_DTC, submission['Survived'], margins=True,rownames=['Previsto'],colnames=['         Real']))

    print (metrics.classification_report(submission['Survived'],predictions_DTC))



    output_DTC = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_DTC})

    output_DTC.to_csv('my_submission_DTC.csv', index=False)
DTC(15)
choosen_DTC(3)
from sklearn.model_selection import cross_val_predict,cross_val_score,cross_validate

from sklearn import metrics
X_test.shape,X_train.shape
y_test.shape,y_train.shape
def DTC_CROSS(depth, folds):

    model_DTC = DecisionTreeClassifier(max_depth=depth)

    model_DTC.fit(X_train, y_train)

    cross_validate(model_DTC,X_train,y_train,cv=folds)

    predictions_DTC_cross = model_DTC.predict(X_test)

    accuracy = metrics.accuracy_score(y_test,predictions_DTC_cross)

    

    print("\n")

    print (pd.crosstab(predictions_DTC_cross, submission['Survived'], margins=True,rownames=['Previsto'],colnames=['         Real']))

    print (metrics.classification_report(submission['Survived'],predictions_DTC_cross))



    

    output_DTC = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_DTC_cross})

    output_DTC.to_csv('my_submission_DTC_cross.csv', index=False)



    return print("accuracy = ",accuracy*100,"%")
DTC_CROSS(4,2) ## We got the same results
from sklearn import svm
features = ['Sex', 'Pclass','Fare','Age','Embarked','SibSp','Parch']

targets = ['Survived']
X_train = train_data[features]

X_test = test_data[features]

X_train.head()
y_train = train_data[targets]

y_test = submission['Survived']

y_train.head()
def choosen_SVM(C, gamma):

    model_SVM = svm.SVC(C=C,gamma=gamma,random_state=0)

    model_SVM.fit(X_train,y_train)

    predictions_SVM = model_SVM.predict(X_test)

    print("Para C = ",C,"e gamma = ", gamma,"   Precisão = ",round((model_SVM.score(X_train,y_train))*100,2),"% ")

    print(pd.crosstab(predictions_SVM,submission['Survived'],margins=True,rownames=['Previsto'],colnames=['         Real']))

    print (metrics.classification_report(submission['Survived'],predictions_SVM))



    output_SVM = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_SVM})

    output_SVM.to_csv('my_submission_SVM.csv', index=False)
choosen_SVM(1000,0.01)
sns.distplot(train_data[targets],kde=False)
train_data['Survived'].value_counts()
from imblearn.under_sampling import NearMiss
X  = train_data[features]

X_test = test_data[features]



y = train_data[targets]

y_test = submission[targets]
nrm = NearMiss()
X, y =  nrm.fit_sample(X,y)
X.shape
sns.distplot(y,kde=False) ##Now we see that the dataset is balanced, let's check the metrics now
lr_us = LogisticRegression()

model = lr_us.fit(X,y)

predictions_lr_us = model.predict(X_test)

print(pd.crosstab(predictions_lr_us,submission['Survived'],margins=True,rownames=['Previsto'],colnames=['         Real']))

print(metrics.classification_report(y_test,predictions_lr_us))
predictions_lr_us.shape
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_lr_us})

output.to_csv("My_submission_Logistic_US.csv", index=False)
from imblearn.over_sampling import SMOTE
features = ['Sex', 'Pclass','Fare','Age','Embarked','SibSp','Parch']

targets = ['Survived']
sns.distplot(train_data['Survived'],kde=False)

train_data['Survived'].value_counts()
print("Percentage of 1's = ",(342/549)*100,"%")
sampling = np.linspace(0.65,1,8)

sampling
def Over_samp(train, test, submisson):

    

    features = ['Sex', 'Pclass','Fare','Age','Embarked','SibSp','Parch']

    targets = ['Survived']

    

    sampling = np.linspace(0.65,1,16)

    

    X  = train[features]

    X_test = test[features]

    

    y = train[targets]

    y_test = submission[targets]

    

    for i in sampling:

        print("For sampling strategy = ",i*100,"%\n")

        smt = SMOTE(sampling_strategy = i)

        X, y = smt.fit_sample(X,y)

        lr_os = LogisticRegression()

        model = lr_os.fit(X,y)

        predictions_lr_os = model.predict(X_test)

        print(pd.crosstab(predictions_lr_os,submission['Survived'],margins=True,rownames=['Previsto'],colnames=['         Real']))

        print(metrics.classification_report(y_test,predictions_lr_os))

        print("\n")
Over_samp(train_data,test_data, submission)
## Let's try 65%
X  = train_data[features]

X_test = test_data[features]

    

y = train_data[targets]

y_test = submission[targets]
smt = SMOTE(sampling_strategy = 0.65)

X, y = smt.fit_sample(X,y)

lr_os = LogisticRegression()

model = lr_os.fit(X,y)

predictions_lr_os = model.predict(X_test)

print(pd.crosstab(predictions_lr_os,submission['Survived'],margins=True,rownames=['Previsto'],colnames=['         Real']))

print(metrics.classification_report(y_test,predictions_lr_os))

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions_lr_os})

output.to_csv("My_submission_Logistic_OS.csv", index=False)
def DTC_US(depth):

    features = ['Sex', 'Pclass','Fare','Age','Embarked','SibSp','Parch']

    targets = ['Survived']  

    

    X_train  = train_data[features]

    X_test = test_data[features]

    

    y_train = train_data[targets]

    y_test = submission[targets]

    

    X,y = nrm.fit_sample(X_train, y_train)    

    sns.distplot(y, kde=False)

    

    DTC = DecisionTreeClassifier(max_depth=depth)

    model = DTC.fit(X, y)

    predictions_DTC_US = model.predict(X_test)

    

    print (pd.crosstab(predictions_DTC_US, submission['Survived'], margins=True,rownames=['Previsto'],colnames=['         Real']))

    print(metrics.classification_report(y_test,predictions_DTC_US))

    output_DTC = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_DTC_US})

    output_DTC.to_csv('my_submission_DTC_US.csv', index=False)
DTC_US(5)
def DTC_OS(depth): 

    features = ['Sex', 'Pclass','Fare','Age','Embarked','SibSp','Parch']

    targets = ['Survived']



    X_train  = train_data[features]

    X_test = test_data[features]



    y_train = train_data[targets]

    y_test = submission[targets]



    smt = SMOTE(sampling_strategy = 0.70)

    X,y = smt.fit_sample(X_train, y_train)

    sns.distplot(y, kde=False)



    DTC = DecisionTreeClassifier(max_depth=depth, random_state=42)

    model = DTC.fit(X, y)

    predictions_DTC_OS = model.predict(X_test)



    print (pd.crosstab(predictions_DTC_OS, submission['Survived'], margins=True,rownames=['Previsto'],colnames=['         Real']))

    output_DTC = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_DTC_OS})

    output_DTC.to_csv('my_submission_DTC_OS.csv', index=False)
DTC_OS(5)
def SVM_US(C, gamma): 

    features = ['Sex', 'Pclass','Fare','Age','Embarked','SibSp','Parch']

    targets = ['Survived']  

    

    X_train  = train_data[features]

    X_test = test_data[features]

    

    y_train = train_data[targets]

    y_test = submission[targets]

    

    X,y = nrm.fit_sample(X_train, y_train)    

    sns.distplot(y, kde=False)

    

    sv = svm.SVC(C=C, gamma=gamma)

    model = sv.fit(X, y)

    predictions_SVM_US = model.predict(X_test)

    

    print (pd.crosstab(predictions_SVM_US, submission['Survived'], margins=True,rownames=['Previsto'],colnames=['         Real']))

    print(metrics.classification_report(y_test,predictions_SVM_US))

    output_DTC = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_SVM_US})

    output_DTC.to_csv('my_submission_SVM_US.csv', index=False)
SVM_US(5,0.01)
def SVM_OS(C,gamma): 

    features = ['Sex', 'Pclass','Fare','Age','Embarked','SibSp','Parch']

    targets = ['Survived']



    X_train  = train_data[features]

    X_test = test_data[features]



    y_train = train_data[targets]

    y_test = submission[targets]



    smt = SMOTE(sampling_strategy = 0.75)

    X,y = smt.fit_sample(X_train, y_train)

    sns.distplot(y, kde=False)



    sv = svm.SVC(C=C, gamma=gamma)

    model = sv.fit(X, y)

    predictions_SVM_OS = model.predict(X_test)

    

    print (pd.crosstab(predictions_SVM_OS, submission['Survived'], margins=True,rownames=['Previsto'],colnames=['         Real']))

    print(metrics.classification_report(y_test,predictions_SVM_OS))

    output_DTC = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_SVM_OS})

    output_DTC.to_csv('my_submission_SVM_OS.csv', index=False)
SVM_OS(100,0.01)
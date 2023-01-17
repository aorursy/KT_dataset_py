# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

sns.set_context('talk')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

print('Loading .....')

print('Data Shape: ', data.shape, ' Test Shape: ', test.shape)
print('Data information')

print('-'*50)

data.info()

print('-'*50)

print(data.isnull().sum())

print('*'*75)

print('Test information')

print('-'*50)

test.info()

print('-'*50)

print(test.isnull().sum())
all_data = [data,test] 

for df in all_data:

    data_corr = df.corr().abs()

    plt.figure(figsize=(12, 6))

    sns.heatmap(data_corr, annot=True,cmap='coolwarm')

    plt.show()
for df in all_data:

    print(df.groupby(['Sex', 'Pclass']).median()['Age'])
for df in all_data:

    df['Initial']=0

    for i in df:

        df['Initial']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

    
pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r')

pd.crosstab(test.Initial,test.Sex).T.style.background_gradient(cmap='summer_r')
#replace the values according the above cross tab

for df in all_data:

    df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],

                          ['Miss','Miss','Miss','Dr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Miss'],inplace=True)
for df in all_data:

    print(df.groupby(['Sex', 'Pclass','Initial']).median()['Age'])
for df in all_data:

#fill the age values based on median valu for Pclass and sex

    df['Age'] = df.groupby(['Sex','Pclass','Initial'])['Age'].apply(lambda x:x.fillna(x.median()))

   
data[data['Embarked'].isnull()]
print(data.groupby(['Embarked','Pclass'])['Pclass'].count())
print(data.groupby(['Embarked','Sex'])['Pclass'].count())
#filling the Embarked features

data['Embarked'] = data['Embarked'].fillna('S')

test[test['Fare'].isnull()]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

med_fare = test.groupby(['Pclass','SibSp','Parch'])['Fare'].median()[3][0][0]

test['Fare'] = test['Fare'].fillna(med_fare)
for df in all_data:

    df.drop(['Cabin'], axis=1, inplace=True)
print('Data information')

print('-'*50)

print(data.isnull().sum())

print('*'*55)

print('Test information')

print('-'*50)

print(test.isnull().sum())
f,ax=plt.subplots(1,2,figsize=(12,6))

data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=data,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
pd.crosstab(data.Sex,data.Survived,margins=True).style.background_gradient(cmap='Set3')
f,ax=plt.subplots(1,2,figsize=(12,6))

data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='Set3')
f,ax = plt.subplots(1,3, figsize=(18,6))

data['Pclass'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True, cmap='Set3')

data['Pclass'].value_counts().plot.bar(cmap='Set3',ax=ax[1])

ax[1].set_title('Number of Passengers by Class')

ax[1].set_ylabel('Count')

ax[1].set_xlabel('Pclass')

sns.countplot('Pclass', hue='Survived',data=data, ax=ax[2], palette='Set3')

ax[2].set_title('Pclass:Survived vs Dead')

pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='Set3')

sns.factorplot('Pclass','Survived',hue='Sex',data=data, palette='Set2')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.swarmplot("Pclass","Age", hue="Survived", data=data,split=True,ax=ax[0],palette='Set2')

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.swarmplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1],palette='Set2')

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
sns.catplot(x="Age", y="Survived",                 

                hue="Sex", row="Pclass",

                data=data,

                orient="h", aspect=2, palette="Set3",

                kind="violin", dodge=True, cut=0, bw=.2

                )
f,ax=plt.subplots(1,2,figsize=(18,6))

data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,cmap='Set3')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

data[data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20, cmap='Pastel1')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)





plt.show()
#lets se this two plot together

plt.figure(figsize=(18,10))

sns.distplot(data[data['Survived']==0].Age,bins=20, kde=False, color='b', label='Died')

sns.distplot(data[data['Survived']==1].Age,bins=20, kde=False, color='r',label='Survived')

plt.legend()

sns.factorplot('Pclass','Survived', col='Initial', data=data, palette='Set2')

plt.show()
pd.crosstab([data.Embarked, data.Pclass],[data.Sex, data.Survived], margins=True).style.background_gradient(cmap='Set3')
sns.factorplot('Embarked', 'Survived', col='Pclass',data=data, palette='Set2')
f, ax = plt.subplots(2,2, figsize=(15,10))

sns.countplot('Embarked', data=data, ax= ax[0,0], palette='Pastel1')

ax[0,0].set_title('Number of Passengers Boarded')

sns.countplot('Embarked', hue='Sex',data=data, ax= ax[0,1], palette='Pastel1')

ax[0,1].set_title('Embarked Splited Female-Male')

sns.countplot('Embarked',hue='Survived', data=data, ax= ax[1,0], palette='Pastel1')

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass',data=data, ax= ax[1,1], palette='Pastel1')

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()

sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data, palette='Set2')

plt.show()
pd.crosstab(data.SibSp, data.Survived, margins=True).style.background_gradient(cmap='Set3')
f, ax = plt.subplots(1,2, figsize=(20,6))

sns.barplot('SibSp','Survived', data=data, ax=ax[0], palette='Set3')

ax[0].set_title('SibSp vs Survived')

sns.factorplot('SibSp','Survived',data=data,ax=ax[1],hue='Pclass', palette='Set2')

ax[1].set_title('SibSp vs Survived')

plt.close(2)

plt.show()
pd.crosstab([data.Parch, data.Survived], data.Pclass, margins=True).style.background_gradient(cmap='Set3')
f, ax = plt.subplots(1,2,figsize=(20,6))

sns.barplot('Parch','Survived', data=data, ax=ax[0], palette='Set3')

ax[0].set_title('Parch vs Survived')

sns.factorplot('Parch','Survived', data=data, ax=ax[1], palatte='Set3', hue='Pclass')

ax[1].set_title('Parch vs Survived')

plt.close(2)

plt.show()
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(data[data['Pclass']==1].Fare, ax=ax[0],kde=False)

ax[0].set_title('Fares in Class 1')

sns.distplot(data[data['Pclass']==2].Fare, ax=ax[1], kde=False)

ax[1].set_title('Fares in Class 2')

sns.distplot(data[data['Pclass']==3].Fare, ax=ax[2], kde=False)

ax[2].set_title('Fares in Class 3')



sns.heatmap(data.corr(),annot=True,cmap='Pastel2',linewidths=0.2, ) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(16,6)

plt.show()
for df in all_data:

    df['Age_bin'] = 0

    df.loc[df['Age']<16,'Age_bin'] = 0

    df.loc[(df['Age']>16) & (df['Age']<=32),'Age_bin'] =1

    df.loc[(df['Age']>32) & (df['Age']<=48),'Age_bin'] =2

    df.loc[(df['Age']>48) & (df['Age']<=64),'Age_bin'] =3

    df.loc[df['Age']>64,'Age_bin'] =4

    

    
f, ax = plt.subplots(1,2,figsize=(20,6))

sns.barplot('Age_bin','Survived', data=data, ax=ax[0], palette='Set3')

ax[0].set_title('Age_bin vs Survived')

sns.factorplot('Age_bin','Survived', data=data, ax=ax[1], palatte='Set3', hue='Pclass')

ax[1].set_title('Age_bin vs Survived')

plt.close(2)

plt.show()
for df in all_data:

    df['Family_size'] = 0

    df['Family_size'] = df['Parch'] + df['SibSp']

    df['Is_Alone'] = 0

    df.loc[df.Family_size == 0, 'Is_Alone'] =1
f, ax = plt.subplots(1,2,figsize=(20,6))

sns.barplot('Family_size','Survived', data=data, ax=ax[0], palette='Set3')

ax[0].set_title('Family_size vs Survived')

sns.factorplot('Family_size','Survived', data=data, ax=ax[1], palatte='Set3', hue='Pclass')

ax[1].set_title('Family_size vs Survived')



plt.close(2)

plt.show()
sns.factorplot('Is_Alone','Survived',data=data,hue='Sex',col='Pclass',palette='Set2')

plt.show()
data['Fare_Range']=pd.qcut(data['Fare'],4)

data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
for df in all_data:

    df['Fare_cat'] = 0

    df.loc[df['Fare']<=7.91,'Fare_cat'] = 0

    df.loc[(df['Fare']>7.91) & (df['Fare']<=14.454), 'Fare_cat'] = 1

    df.loc[(df['Fare']>14.454) & (df['Fare']<=31.0), 'Fare_cat'] = 2

    df.loc[(df['Fare']>31.0) & (df['Fare']<=513), 'Fare_cat'] = 3



sns.factorplot('Fare_cat','Survived',data=data,hue='Sex',palette='Set2')

plt.show()
gender =  {'male': 0,'female': 1} 

embarked = {'S':0, 'C':1,'Q':2}

initial = {'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Dr':4}

for df in all_data:

    df['Sex'] = [gender[item] for item in df.Sex]

    df['Embarked'] = [embarked[item] for item in df.Embarked]

    df['Initial'] = [initial[item] for item in df.Initial]

    
# creating Class Embark and Sex future together

for df in all_data:

    df['Sex_Class_Embark'] = 0 

    df.loc[(df['Sex'] == 1) & ((df['Pclass'] == 1) | (df['Pclass'] == 2) ) & 

           ((df['Embarked'] == 0)  | (df['Embarked'] == 1)  | (df['Embarked'] == 2)),'Sex_Class_Embark'] = 0

    

    df.loc[(df['Sex'] == 1) & (df['Pclass'] == 3) & ((df['Embarked'] == 1)  | (df['Embarked'] == 2)),'Sex_Class_Embark'] = 1



    df.loc[(df['Sex'] == 0) & (df['Pclass'] == 1) & ((df['Embarked'] == 0)  | (df['Embarked'] == 1)),'Sex_Class_Embark'] = 2

    df.loc[(df['Sex'] == 1) & (df['Pclass'] == 3) & (df['Embarked'] == 0),'Sex_Class_Embark'] = 2

    

    df.loc[(df['Sex'] == 0) & ((df['Pclass'] == 2)  | (df['Pclass'] == 3) ) & 

           ((df['Embarked'] == 0)  | (df['Embarked'] == 1)  | (df['Embarked'] == 2)),'Sex_Class_Embark'] = 3

    

    df.loc[(df['Sex'] == 0) & ((df['Pclass'] == 1)  |(df['Pclass'] == 2) ) & (df['Embarked'] == 2),'Sex_Class_Embark'] = 4

    

    
data['Sex_Class_Embark'].unique()

data.head()
data.drop(['Name','Age','Ticket','Fare','Fare_Range','PassengerId'],axis=1,inplace=True)
#before remove the Passengerid

test_copy = test.copy()
test.drop(['Name','Age','Ticket','Fare','PassengerId'],axis=1,inplace=True)
#data have 1 more columns which is survived column, we are going to used this column as target

print(data.shape, test.shape)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(30,8)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix



from sklearn.model_selection import GridSearchCV
y =data.Survived

X = data.drop('Survived', axis=1)
#from sklearn.preprocessing import StandardScaler

#std_scaler = StandardScaler()

#X = std_scaler.fit_transform(X)

#test = std_scaler.transform(test)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)
model_log = LogisticRegression(solver='liblinear')

model_log.fit(X_train, y_train)

prediction_log = model_log.predict(X_valid)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction_log, y_valid))
model_svm_l = svm.SVC(kernel='linear', C=0.1, gamma=0.1)

model_svm_l.fit(X_train, y_train)

prediction_svm_l = model_svm_l.predict(X_valid)

print('The accuracy of the Linear Support Vector Machine is ', metrics.accuracy_score(prediction_svm_l, y_valid))
model_rbf = svm.SVC(kernel='rbf', C=0.1, gamma=0.1)

model_rbf.fit(X_train, y_train)

prediction_rbf = model_rbf.predict(X_valid)

print('The accuracy of the Radical Support Vector Machine is ', metrics.accuracy_score(prediction_rbf, y_valid))
model_tree = DecisionTreeClassifier() 

model_tree.fit(X_train, y_train)

prediction_tree = model_tree.predict(X_valid)

print('The accuracy of the Decision Tree is ', metrics.accuracy_score(prediction_tree, y_valid))
model_knn = KNeighborsClassifier()

model_knn.fit(X_train, y_train)

prediction_knn = model_knn.predict(X_valid)

print('The accuracy of the  K-Nearest Neighbours is ', metrics.accuracy_score(prediction_knn, y_valid))
s =pd.Series()

for i in list(range(1,11)):

    model_knn = KNeighborsClassifier(n_neighbors=i)

    model_knn.fit(X_train, y_train)

    prediction_knn = model_knn.predict(X_valid)

    s = s.append(pd.Series(metrics.accuracy_score(prediction_knn, y_valid)))



plt.plot(list(range(1,11)), s)

plt.xticks([0,1,2,3,4,5,6,7,8,9,10])

plt.title('The Accuracy vs n_neighbors K-Nearest Neighbours')

plt.xlabel('n_neighbors')

plt.ylabel('The Accuracy of the K-Nearest Neighbours')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()

    
model_gaus = GaussianNB()

model_gaus.fit(X_train, y_train)

prediction_gaus = model_gaus.predict(X_valid)

print('The accuracy of the  Gaussian Naive Bayes is ', metrics.accuracy_score(prediction_gaus, y_valid))
list_n_estimators = [50,100,150,200,250,300,350,400,450,500]

random_acc = pd.Series()

for i in list_n_estimators:

    model_random = RandomForestClassifier(n_estimators=i)

    model_random.fit(X_train, y_train)

    predict_random = model_random.predict(X_valid)

    random_acc =random_acc.append(pd.Series(metrics.accuracy_score(predict_random, y_valid)))

#print(random_acc)

plt.plot(list_n_estimators, random_acc)

plt.xticks(list_n_estimators)

plt.title('The Accuracy vs n_estimators Random Forests')

plt.xlabel('n_estimators')

plt.ylabel('The Accuracy of the Random Forests')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
model_random = RandomForestClassifier(n_estimators=300)

model_random.fit(X_train, y_train)

predict_random = model_random.predict(X_valid)

print('The accuracy of the  Random Forest is ', metrics.accuracy_score(predict_random, y_valid))
import sklearn
#load nesseray libraries

from sklearn.model_selection import KFold

from  sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict



from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier





kfold =KFold(n_splits=10, random_state=22)

xyz = []

accuracy = []

std = []



classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN', 'Decision Tree', 'Naive Bayes' , 'Random Forest']

models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(solver='liblinear'), KNeighborsClassifier(n_neighbors=9), 

      DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=100)]



    

    

    

for i in models:

    model = i

    cv_result = cross_val_score(model,X,y, cv=kfold,scoring='accuracy')

    cv_result =cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)



new_models_data_frame = pd.DataFrame({'CV Mean': xyz, 'Std': std}, index=classifiers)

new_models_data_frame

plt.subplots(figsize=(12,6))

plt.xticks(rotation=45)

sns.boxplot(new_models_data_frame.index, accuracy)

f, ax  =plt.subplots(3,3, figsize=(12,10))

y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred), ax=ax[0,0], annot=True,fmt='2.0f')

ax[0,0].set_title('Linear SVM')



y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred), ax=ax[0,1], annot=True,fmt='2.0f')

ax[0,1].set_title('Radical SVM')



y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9) ,X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred), ax=ax[0,2], annot=True,fmt='2.0f')

ax[0,2].set_title('KNN')



y_pred = cross_val_predict(LogisticRegression(solver='liblinear') ,X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred), ax=ax[1,0], annot=True,fmt='2.0f')

ax[1,0].set_title('Logistic Regression')



y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100) ,X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred), ax=ax[1,1], annot=True,fmt='2.0f')

ax[1,1].set_title('Random Forest')



y_pred = cross_val_predict(DecisionTreeClassifier() ,X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred), ax=ax[1,2], annot=True,fmt='2.0f')

ax[1,2].set_title('Decision Tree')



y_pred = cross_val_predict(GaussianNB() ,X,y,cv=10)

sns.heatmap(confusion_matrix(y,y_pred), ax=ax[2,0], annot=True,fmt='2.0f')

ax[2,0].set_title('Naive Bayes')



plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()

from sklearn.model_selection import GridSearchCV

C=[0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel=['rbf','linear']

hyper = {'kernel':kernel, 'C':C, 'gamma':gamma}

gd =GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)





gd.fit(X,y)



print(gd.best_score_)

print(gd.best_estimator_)
model_rbf = svm.SVC(kernel='rbf', C=0.35, gamma=0.1)

model_rbf.fit(X_train, y_train)

prediction_rbf = model_rbf.predict(X_valid)

print('The accuracy of the Radical Support Vector Machine is ', metrics.accuracy_score(prediction_rbf, y_valid))
#n_estimator =range(50, 1000, 50)

#hyper = {'n_estimators': n_estimator}

#gd = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=hyper, verbose=True)

#gd.fit(X,y)

#print(gd.best_score_)

#print(gd.best_estimator_)

from sklearn.ensemble import VotingClassifier



ensemble = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=9)),

                                        ('RBF', svm.SVC(kernel='rbf',probability=True,C=0.4,gamma=0.1)),

                                        ('RFor', RandomForestClassifier(n_estimators=900, random_state=0)),

                                        ('LR', LogisticRegression(C=0.05)),

                                        ('DT', DecisionTreeClassifier(random_state=0)),

                                        ('NB', GaussianNB()),

                                        ('Svm', svm.SVC(kernel='linear',probability=True))],

                           voting='soft').fit(X_train, y_train)



print('The accuracy for ensembled model is:',ensemble.score(X_valid,y_valid))

cross=cross_val_score(ensemble,X,y, cv = 10,scoring = "accuracy")

print('The cross validated score is',cross.mean())
from sklearn.ensemble import BaggingClassifier

model_bag = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), random_state=0, n_estimators=800)

model_bag.fit(X_train, y_train)

prediction_bag = model_bag.predict(X_valid)

print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction_bag,y_valid))

result=cross_val_score(model_bag,X,y,cv=10,scoring='accuracy')

print('The cross validated score for bagged KNN is:',result.mean())

model_bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0, n_estimators=800)

model_bag.fit(X_train, y_train)

prediction_bag = model_bag.predict(X_valid)

print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction_bag,y_valid))

result=cross_val_score(model_bag,X,y,cv=10,scoring='accuracy')

print('The cross validated score for bagged KNN is:',result.mean())
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.05)

result = cross_val_score(ada, X,y, cv=10, scoring='accuracy')

print('The cross validated score for AdaBoost is:',result.mean())
from sklearn.ensemble import GradientBoostingClassifier

grad=GradientBoostingClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

result=cross_val_score(grad,X,y,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boosting is:',result.mean())
import xgboost as xg

xgboost=xg.XGBClassifier(n_estimator=900,learning_rate=0.1)

result=cross_val_score(xgboost,X,y,cv=10,scoring='accuracy')

print('The cross validated score for XGBoost is:',result.mean())
## Hyper-PArameter Tuning for AdaBoost



#n_estimators = list(range(100,1000,100))

#learn_rate = [0.01,0.02,0.03,0.04,0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

#hyper = {'n_estimators':n_estimators, 'learning_rate': learn_rate}

#gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)

#gd.fit(X,y)

#print(gd.best_score_)

#print(gd.best_estimator_)

model_random.fit(X_train, y_train)
f, ax = plt.subplots(2,2, figsize=(15,12))

model=RandomForestClassifier(n_estimators=500,random_state=0)

model.fit(X,y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0], cmap='Set3')

ax[0,0].set_title('Feature Importance in Random Forests')



model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)

model.fit(X,y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='lightcoral')

ax[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)

model.fit(X,y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],color='lightgreen')

ax[1,0].set_title('Feature Importance in Gradient Boosting')

model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

model.fit(X,y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='violet')

ax[1,1].set_title('Feature Importance in XgBoost')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()

X.columns
#Let's redefine the feture for random forest

feature_random = ['Initial','Sex_Class_Embark','Pclass','Fare_cat','Age_bin','Family_size']



X_random = X[feature_random]

X_train, X_valid, y_train, y_valid = train_test_split(X_random,y, test_size=0.3, random_state=0)



#model=RandomForestClassifier(n_estimators=500,random_state=0)

model =AdaBoostClassifier(n_estimators=900, random_state=0, learning_rate=0.01)

model.fit(X_train, y_train)

predict_random = model.predict(X_valid)

print('The accuracy of the  Model is ', metrics.accuracy_score(predict_random, y_valid))
#Let's redefine the feture for random forest

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=0)



model_rbf = svm.SVC(kernel='rbf', C=0.35, gamma=0.1)

model_rbf.fit(X_train, y_train)

prediction_rbf = model_rbf.predict(X_valid)

print('The accuracy of the Radical Support Vector Machine is ', metrics.accuracy_score(prediction_rbf, y_valid))
test_random = test
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]

algorithm = ['auto']

weights = ['uniform', 'distance']

leaf_size = list(range(1,50,5))

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 

               'n_neighbors': n_neighbors}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 

                cv=10, scoring = "roc_auc")

gd.fit(X, y)

print(gd.best_score_)

print(gd.best_estimator_)
gd.best_estimator_.fit(X, y)

pred_test= gd.best_estimator_.predict(test)
#pred_test = model_rbf.predict(test_random)
output = pd.DataFrame({'PassengerId' : test_copy.loc[:,'PassengerId'],

                       'Survived': pred_test})

output.to_csv('submission.csv', index=False)
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
train_data=pd.read_csv("../input/titanic/train.csv")

test_data=pd.read_csv('../input/titanic/test.csv')
data=train_data.drop("Survived",axis=1).append(test_data,sort=False)

data.head()
# Handling Missing Values

data.isnull().sum()
data.drop('Cabin',axis=1,inplace=True)
data.head()
data['Age'].fillna(data['Age'].mean(),inplace=True)

data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

data['Fare'].fillna(data['Fare'].mode()[0],inplace=True)
data.isnull().sum()
sns.countplot(x='Sex',data=data[:891],hue=train_data['Survived'])
sns.countplot(x='Pclass',data=data[:891],hue=train_data['Survived'])
sns.countplot(x='Embarked',data=data[:891],hue=train_data['Survived'])
train_data['Age'].hist(bins=50,figsize=(12,5))
data.loc[data['Age']<=13,'Age']=0

data.loc[(data['Age']>13) & (data['Age']<=40) ,'Age']=1

data.loc[data['Age']>40,'Age']=3
sns.countplot(x='Age',data=data[:891],hue=train_data['Survived'])
train_data['Fare'].hist(bins=50,figsize=(12,5))
data.loc[data['Fare']<=10,'Fare']=0

data.loc[(data['Fare']>10) & (data['Fare']<=90) ,'Fare']=1

data.loc[data['Fare']>90,'Fare']=2
sns.countplot(x='Fare',data=data[:891],hue=train_data['Survived'])
data['Family_Size']=data.SibSp + data.Parch +1

data['Is_Alone']=np.where(data['Family_Size']>1,0,1)
sns.countplot(x='Family_Size',data=data[:891],hue=train_data['Survived'])
sns.countplot(x='Is_Alone',data=data[:891],hue=train_data['Survived'])
data.drop(['Ticket','Name','SibSp','Parch','Family_Size'],axis=1,inplace=True)
data.head(10)
X=data.iloc[:891,1:].values

y=train_data['Survived'].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:,1] = labelencoder_X.fit_transform(X[:, 1])

X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

onehotencoder = OneHotEncoder(categorical_features = [4])

X= onehotencoder.fit_transform(X).toarray()

X=X[:,1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.metrics import confusion_matrix,accuracy_score

def evaluation_metrics():

    cm=confusion_matrix(y_test,y_pred)

    print("Confusion Matrix:")

    print(cm)

    print("Accuracy:",accuracy_score(y_test,y_pred))
from sklearn.model_selection import cross_val_score

def K_Fold():

    accuracies_f=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)

    print("Mean Accuracy:",accuracies_f.mean())

    print("Std:",accuracies_f.std())
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
evaluation_metrics()
K_Fold()
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

classifier=XGBClassifier()

optimization_dict = {'max_depth': [2,4,6,10],

                     'n_estimators': [50,100,200,300,1000,3000],

                    'learning_rate':[0.001,0.01,0.05,0.1],

                    'gamma':[0,0.1,0.5,1]}



gridsearch = GridSearchCV(classifier, optimization_dict, 

                     scoring='accuracy', verbose=1)

gridsearch.fit(X_train,y_train)

print("best_accuracy:",gridsearch.best_score_)

print("best_parameters:",gridsearch.best_params_)
classifier=XGBClassifier(max_depth=4, n_estimators= 50,learning_rate=0.001,gamma=0.5)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
evaluation_metrics()
K_Fold()
#------------------Decision Trees-----------------

from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
evaluation_metrics()
K_Fold()
### ---------------Random Forest---------------

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
evaluation_metrics()
K_Fold()
from sklearn.svm import SVC

classifier=SVC()

from sklearn.model_selection import GridSearchCV

parameters=[{'C':[1,10,100,1000],'kernel':['linear']},

             {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}

             ]

gridsearch=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)

gridsearch.fit(X_train,y_train)

print("best_accuracy:",gridsearch.best_score_)

print("best_parameters:",gridsearch.best_params_)
classifier=SVC(C=10, gamma=0.1, kernel= 'rbf')

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
evaluation_metrics()
K_Fold()
#-------------Naive Bayes----------------------------------------------------

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
evaluation_metrics()
K_Fold()
#--------------K Nearest Neighobors------

from sklearn.neighbors import KNeighborsClassifier



# Choosing number of Neighbors

error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
classifier=KNeighborsClassifier(n_neighbors=6,metric='minkowski',p=2)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
evaluation_metrics()
K_Fold()
test=data.iloc[891:,1:].values
# Encoding categorical data(Test Data)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

test[:,1] = labelencoder_X.fit_transform(test[:, 1])

test[:, 4] = labelencoder_X.fit_transform(test[:, 4])

onehotencoder = OneHotEncoder(categorical_features = [4])

test= onehotencoder.fit_transform(test).toarray()

test=test[:,1:]

classifier=XGBClassifier(max_depth=4, n_estimators= 50,learning_rate=0.001,gamma=0.5)

classifier.fit(X_train,y_train)
pred=classifier.predict(test)
pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':pred}).set_index('PassengerId').to_csv('Final_Submission.csv')
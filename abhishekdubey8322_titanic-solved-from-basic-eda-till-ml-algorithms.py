import numpy as np 

import pandas as pd 

import seaborn as sns                                   

import matplotlib as mpl

%matplotlib inline

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic.shape
test.shape
print('no of rows',titanic.shape[0])

print('no of columns',titanic.shape[1])
titanic.info()
test.info()
titanic.describe()
titanic.describe(include='all')
sns.countplot(x='Pclass',data = titanic)

sns.countplot(x='Survived',data=titanic)
sns.countplot(x='Pclass', hue='Survived',data=titanic)



## hue parameter splits the graph into defined parameter



sns.countplot(x='Sex',hue='Survived',data=titanic)
sns.countplot(x='Embarked',hue='Survived',data=titanic)
sns.violinplot("Pclass",'Sex',data=titanic)

sns.violinplot('Pclass',data=titanic)
sns.violinplot(x='Pclass',y='Sex',hue='Survived',data=titanic)
pd.crosstab(titanic['Sex'],titanic['Survived'])
sns.factorplot('Sex','Survived',data=titanic)
sns.factorplot('Pclass','Survived',data=titanic)
sns.barplot(x='Pclass',y='Fare',data=titanic)
sns.catplot(x='Embarked',y="Fare", data=titanic,kind='swarm')
plt.figure(figsize=(15,6))

corr = titanic.corr()



sns.heatmap(corr,annot=True) 
sns.pairplot(titanic)
titanic.isnull().sum()
titanic.groupby('SibSp')['Age'].describe()



## To see null values in 'SibSp' with relative to 'Age'
titanic.groupby('SibSp')['Age'].median()

titanic['Age'].median()
titanic['Age']=np.where((titanic['SibSp']==8) & (titanic['Age'].isnull()),28.0,titanic['Age'])

titanic['Age']=np.where((titanic['SibSp']==0) & (titanic['Age'].isnull()),29.0,titanic['Age'])

titanic['Age']=np.where((titanic['SibSp']==1) & (titanic['Age'].isnull()),30.0,titanic['Age'])

titanic['Age']=np.where((titanic['SibSp']==2) & (titanic['Age'].isnull()),23.0,titanic['Age'])

titanic['Age']=np.where((titanic['SibSp']==3) & (titanic['Age'].isnull()),10.0,titanic['Age'])
titanic.isnull().sum()

titanic['Embarked'].value_counts()
titanic['Embarked'].fillna('S',inplace=True)
titanic.head(20)
titanic.drop(columns=['Name','Ticket','Cabin'],inplace=True)
titanic.info()
sns.boxplot(titanic['Age'])
sns.scatterplot('Age','Survived',data=titanic)
sns.boxplot(titanic['Fare'])
sns.scatterplot('Fare','Survived',data=titanic)
titanic.sort_values(by='Fare',ascending=False)[:10]
titanic.drop(titanic[titanic['Fare']>500].index, inplace = True)
sns.boxplot(titanic['Fare'])
df_num = titanic.select_dtypes(include=[np.number]).copy()
df_num
df_cat = titanic.select_dtypes(include='object').copy()
df_cat
df_cat = pd.get_dummies(df_cat)
df_cat
df = pd.concat([df_num,df_cat],axis=1)
df
df.drop(columns=['Sex_male','Embarked_C','PassengerId'],inplace=True)
df.shape
X = df.drop(columns=['Survived'])

Y = df['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.30,random_state=0)
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_pred)

print('Accuracy is',"{:.2f}%".format(100*accuracy))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy',max_depth=15,random_state=0,max_leaf_nodes=10)

dt.fit(X_train,Y_train)

Y_pred = dt.predict(X_test)

accuracy = accuracy_score(Y_test,Y_pred)

print('Accuracy of Decision Tree model is',"{:.2f}%".format(100*accuracy))
from sklearn import *

model_params = {

               'max_leaf_nodes':range(10,20),

               'criterion':['gini','entropy'],

               'max_depth':range(1,10),

               'min_impurity_decrease':[0.00005,0.0001,0.0002,0.0005,0.001,0.0015,0.002,0.005,0.01]}



dt_model_improved = DecisionTreeClassifier()



random_search_object = model_selection.RandomizedSearchCV(dt_model_improved,model_params,

                                     n_iter=10,cv=5,random_state=0)





random_search_best_model = random_search_object.fit(X_train,Y_train)



Y_pred = random_search_best_model.predict(X_test)



accuracy= accuracy_score(Y_test,Y_pred)



print('Accuracy of Decision Tree improved Model is:',"{:.2f}%".format(100*accuracy))
random_search_best_model.best_params_
from sklearn import *

rf=ensemble.RandomForestClassifier(n_estimators=150,criterion='entropy',

                                        random_state=0)

rf.fit(X_train,Y_train)



Y_pred = rf.predict(X_test)

accuracy = accuracy_score(Y_test,Y_pred)



print('Accuracy of Random Forest Model is:',"{:.2f}%".format(100*accuracy))
model_params=  {'n_estimators':[140,145,150,155,160],

               'max_leaf_nodes':range(10,30),

               'criterion':['gini','entropy'],

                'max_depth':range(1,10),

               'min_impurity_decrease':[0.00005,0.0001,0.0002,0.0005,0.001,0.0015,0.002,0.005,0.01]}



rf_model_improved = ensemble.RandomForestClassifier(random_state=0)



random_search_object = model_selection.RandomizedSearchCV(rf_model_improved,model_params,

                                     n_iter=10,cv=5,random_state=0)



random_search_best_model = random_search_object.fit(X_train,Y_train)



Y_pred = random_search_best_model.predict(X_test)



accuracy= accuracy_score(Y_test,Y_pred)



print('Accuracy of Random Forest Model is:',"{:.2f}%".format(100*accuracy))
random_search_best_model.best_params_
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3, metric='euclidean')

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

knn.fit(X_train,Y_train)

Y_pred = knn.predict(X_test)

accuracy= accuracy_score(Y_test,Y_pred)

print('Accuracy of KNN algorithm is',"{:.2f}%".format(100*accuracy))
from sklearn import *



model_params = {'leaf_size':range(1,50),

               'n_neighbors':range(1,30),

               'p':[1,2]}



knn_improved = KNeighborsClassifier()



grid_search_object = model_selection.GridSearchCV(knn_improved, model_params,cv=10)



grid_search_best_model = grid_search_object.fit(X_train,Y_train)



Y_pred = grid_search_best_model.predict(X_test)



accuracy= accuracy_score(Y_test,Y_pred)



print('Accuracy of KNN improved Model is:',"{:.2f}%".format(100*accuracy))
grid_search_best_model.best_params_

## Let's Start submission process
test.info()
test.describe()
test['Fare'].mean()
test['Fare'].fillna(35.6271884892086,inplace=True)
test.isnull().sum()
test.groupby('SibSp')['Age'].median()
test.groupby('SibSp')['Age'].describe()
test['Age'].isnull().value_counts()
test['Age']=np.where((test['SibSp']==8) & (test['Age'].isnull()),14.5,test['Age'])

test['Age']=np.where((test['SibSp']==0) & (test['Age'].isnull()),27.0,test['Age'])

test['Age']=np.where((test['SibSp']==1) & (test['Age'].isnull()),30.0,test['Age'])

test['Age']=np.where((test['SibSp']==2) & (test['Age'].isnull()),21.0,test['Age'])

test['Age']=np.where((test['SibSp']==3) & (test['Age'].isnull()),28.5,test['Age'])
test['Age'].isnull().value_counts()
test.drop(columns=['Cabin','Name','Ticket'],inplace=True)
test.info()
test.head()
test_num = test.select_dtypes(exclude='object').copy()

test_dummies = test.select_dtypes(include='object').copy()

test_dummies = pd.get_dummies(test_dummies)
test_dummies.drop(columns=['Sex_male','Embarked_C'],inplace=True)

df_test = pd.concat([test_num,test_dummies],axis=1)
df_test.head()
X_test = df_test.drop('PassengerId',axis=1)

Y_test_pred = random_search_best_model.predict(X_test)
submission = pd.DataFrame(df_test['PassengerId'])

submission['Survived'] = Y_test_pred 
submission.head()

submission.to_csv('titanic_result.csv',index=False)
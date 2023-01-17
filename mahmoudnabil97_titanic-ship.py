import pandas as pd

import numpy as np  

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train.info()
train.describe().T
train.describe(include=['O']).T
# EXplore Missing Values

train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
df_num = train[['Age','SibSp','Parch','Fare']]

df_cat = train[['Survived' , 'Sex' , 'Pclass' , 'Embarked']]
for col in df_num.columns:

  plt.hist(df_num[col])

  plt.title(col)

  plt.show()
for col in df_cat.columns:

  sns.barplot( df_cat[col].value_counts().index,df_cat[col].value_counts() ).set_title(col)

  plt.show()



    
sns.countplot(x="Survived" , hue="Sex" , data=train)
train[['Sex' , 'Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived' , ascending=False)
sns.countplot(x="Survived" , hue = "Pclass" , data=train)
train[['Survived' , 'Pclass']].groupby(['Pclass']).mean().sort_values(by='Survived' , ascending=False)
pd.pivot_table(train,values="Name" , index=["Pclass","Sex"] , columns="Survived",aggfunc="count" )
train[['Survived' , 'Sex' , 'Pclass']].groupby(['Pclass' , 'Sex']).mean().sort_values(by='Survived' , ascending=False)
sns.countplot(x="Survived", hue="Embarked",data=train)
#pd.pivot_table(train,values="Name", index=['Pclass' , 'Embarked'] , columns="Age",aggfunc="count")

sns.countplot(x='Embarked' , hue='Pclass',data=train)
sns.heatmap(train.corr(),annot=True)
train[['Embarked' , 'Age' ]].groupby(['Embarked' ]).mean().sort_values(by='Age' , ascending=False)
train[['Age' , 'Sex']].groupby(['Sex']).mean().sort_values(by='Age' , ascending=False)
train[['SibSp' ,  'Parch','Age']].groupby(['SibSp','Parch']).mean().sort_values(by ='Age' , ascending=False)
train[['Age' , 'Sex' , 'Pclass']].groupby(['Pclass' , 'Sex']).mean().sort_values(by='Age' , ascending=False)
age = pd.cut(train['Age'], [0, 18, 32,48,64,80])

train.pivot_table('Survived', ['Sex', age], 'Pclass').plot.bar()
dataset = pd.concat((train,test) , sort = False).reset_index(drop=True)

dataset = dataset.drop(columns =['Survived'], axis =1 )
dataset.head()
#columns that have missing values :- 

#Cabin ==>  687 , Age ==> 177 , Embarked === > 2

def impute_age(col):

  Age    = col[0]

  Pclass = col[1]

  Sex    = col[2]

  if pd.isnull(Age):

    if Pclass==1:

      if Sex == 'female':

        return 34

      else:

        return 41

    elif Pclass == 2:

      if Sex == 'female':

        return 28

      else:

        return 30

    elif Pclass==3:

      if Sex =='female':

        return 21

      else:

        return 26

  else:

   return Age 



dataset['Age']= dataset[['Age','Pclass' , 'Sex']].apply(impute_age,axis=1)
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
dataset['Cabin_ch'] = dataset.Cabin.str.extract('([A-Za-z])', expand=False)

dataset.drop('Cabin',axis=1 , inplace=True)

dataset.head()

#dataset['Cabin'][0]
# make intiution on Cabin_ch

dataset['Cabin_ch'].value_counts()
#explore correlation between Fare and Cabin_ch

dataset.pivot_table('Fare' , 'Cabin_ch' , aggfunc='median').sort_values(by='Fare' , ascending=False)
dataset.pivot_table(values='Cabin_ch' , index='Embarked' , aggfunc='count').sort_values(by='Cabin_ch',ascending=False)
sns.countplot(y='Embarked' , hue='Cabin_ch',data=dataset)
dataset['Cabin_ch']= dataset['Cabin_ch'].fillna('C')
dataset.head()
dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].median())
dataset.isnull().sum().sort_values(ascending=False)
#check for Duplicate values

duplicate = dataset[dataset.duplicated()]

print(duplicate)
dataset['Title']= dataset.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())
pd.unique(dataset['Title'])
dataset['Title'].value_counts()
#train['Title'].replace(['Dr' , 'Rev','Col',''])
#train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived' , ascending=False)
dataset['Pclass'] = dataset['Pclass'].astype('str')
dataset.drop(['PassengerId','Name' ,'Ticket'] , axis =1 , inplace=True)
dataset.head()
dataset = pd.get_dummies(dataset , drop_first= True).reset_index(drop=True)

dataset.head() 
y=train['Survived']

X = dataset.iloc[:len(y),:]

df_test = dataset.iloc[len(y): , : ]
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()

sc_test = MinMaxScaler()

X=sc_X.fit_transform(X)

df_test=sc_test.fit_transform(df_test)
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

logReg = LogisticRegression()

logReg.fit(X_train,y_train)

logReg_pred = logReg.predict(X_test)

#Evalulate Model

logReg_acc_score= round(accuracy_score(y_test,logReg_pred),2)

print('Logistic Regression Score :  ',logReg_acc_score)

## using Cross validation 

cv_logReg = round(cross_val_score(logReg , X , y, cv=5).mean(),2)

print('cross_val_score LogReg :  ',cv_logReg, '\n\n')
from sklearn.metrics import  confusion_matrix

logReg_accuracy = confusion_matrix(y_test,logReg_pred)

print(logReg_accuracy)
from sklearn.metrics import classification_report

print(classification_report(y_test,logReg_pred))
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()

knc.fit(X_train,y_train)

knc_pred = knc.predict(X_test)

# evaluate accuracy 

knc_accuracy = round(accuracy_score(y_test,knc_pred),2)

print('KNeighbors Classifier :    ',knc_accuracy)

## using cross validation

cv_knc = round(cross_val_score(knc,X,y,cv=5).mean(),2)

print('cross_val_score K_Nearest Neighbors :    ',cv_knc , '\n\n')
from sklearn.svm import SVC

svc = SVC(kernel='rbf' , random_state=0)

svc.fit(X_train , y_train)

svc_pred = svc.predict(X_test)

# evaluate accuracy 

svc_accuracy = round(accuracy_score(y_test,svc_pred),2)

print(svc_accuracy)

## using cross validation

cv_svc = round(cross_val_score(svc,X,y,cv=5).mean(),2)

print('cross_val_score :    ',cv_svc , '\n\n')
from sklearn.naive_bayes import GaussianNB

n_bayes = GaussianNB()

n_bayes.fit(X_train ,y_train)

n_bayes_pred = n_bayes.predict(X_test)

# evaluate accuracy 

n_bayes_acc = round(accuracy_score(y_test,n_bayes_pred),2)

print('Naive Bayes:   ',n_bayes_acc )

## using cross validataion

cv_n_bayes = round(cross_val_score(n_bayes,X,y,cv=5).mean(),2)

print('cross_val_score:   ' , cv_n_bayes,'\n\n')
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion = 'entropy',random_state=0)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

#Evaluate Model

tree_acc = round(accuracy_score(y_test,tree_pred),2)

print('Decision Tree Classifier:', tree_acc)

## using cross validataion

cv_tree =round(cross_val_score(tree,X,y,cv=5).mean(),2)

print('cross_val_score Decision Tree:   ' , cv_tree,'\n\n')
from sklearn.ensemble import RandomForestClassifier

ran_forest = RandomForestClassifier(n_estimators=10 , criterion='entropy' , random_state=0)

ran_forest.fit(X_train,y_train)

ran_forest_pred = ran_forest.predict(X_test)

#Evaluate Model

ran_forest_acc= round(accuracy_score(y_test,ran_forest_pred),2)

print('Random Forest :     ',ran_forest_acc)

## using cross validataion

cv_ran_forest =round(cross_val_score(ran_forest,X,y,cv=5).mean(),2)

print('cross_val_score:   ' , cv_ran_forest,'\n\n')
models = pd.DataFrame({'Model' : ['Logistic Regression' , 'K_Nearest Neighbors', 'SVM' , 'Naive Bayes' , 'Decision Tree','Random Forest'],

                       'Score' : [logReg_acc_score   ,knc_accuracy     ,svc_accuracy, n_bayes_acc  ,  tree_acc       , ran_forest_acc],

                       'Cross Validation': [cv_logReg , cv_knc     ,cv_svc ,cv_n_bayes,cv_tree,cv_ran_forest]})

models.sort_values(by='Score' , ascending=False)
#prediction for test data

y_pred = logReg.predict(df_test)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic_model_v1.csv' , index=False) 
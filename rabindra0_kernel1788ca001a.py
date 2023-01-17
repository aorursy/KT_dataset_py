import numpy as np                                                 # Implemennts milti-dimensional array and matrices
import pandas as pd                                                # For data manipulation and analysis
import pandas_profiling
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics
%matplotlib inline
sns.set()

from subprocess import check_output
titanic_data = pd.read_csv("https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/titanic_train.csv") 
titanic_data.head()
titanic_data.info()
titanic_data =titanic_data.drop(columns= ['Name','Ticket','Cabin'], axis = 1)
titanic_data.head()
titanic_data.isnull().sum()
median_age = titanic_data.Age.median()
Embarked_mode = titanic_data.Embarked.mode()[0]
print(Embarked_mode)
print("the median age {0} and mode eberked is {1} ".format(median_age,Embarked_mode))
titanic_data.Age.fillna(median_age,inplace=True)
titanic_data.Embarked.fillna(Embarked_mode,inplace = True)
titanic_data.isnull().sum()
titanic_data['Gender'] = titanic_data.apply(lambda x: "Child" if x['Age']<15 else x['Sex'], axis = 1)
titanic_data[titanic_data['Age']<15].head()
titanic_data['FamiltSize'] = titanic_data['SibSp'] + titanic_data['Parch'] +1
titanic_data = titanic_data.drop (columns = ['PassengerId','SibSp','Parch'], axis =1)
titanic_data = titanic_data.drop(['Sex'], axis =1)
titanic_data.head()
titanic_data.to_csv('titanic_data_Processed.csv',index = False)
sns.pairplot(titanic_data,vars = ["Fare","Age","Pclass"],hue ='Survived',kind='scatter')

titanic_data['Survived'].value_counts().plot(kind = 'pie',subplots=True, figsize=(6, 3))

titanic_data['Pclass'].value_counts().plot(kind = 'pie',subplots=True, figsize=(6, 3))

titanic_data['Age'].value_counts().plot(kind = 'pie',subplots=True, figsize=(25, 10))
titanic_data = pd.get_dummies(titanic_data,columns = ['Embarked','Gender'],drop_first =True)
titanic_data.head()
titanic_data.to_csv('titanic_data_Processed_dummy.csv')
X = titanic.loc[:,titanic.columns != 'Survived']
X.head()
X = titanic_data.loc[:,titanic_data.columns!='Survived']
y= titanic_data.Survived
X.head(2)
y.head(2)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=1)
from sklearn.tree  import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state = 0)
model_dt.fit(X_train,y_train)
y_pred_train = model_dt.predict(X_train)
y_pred_test = model_dt.predict(X_test)
from sklearn.metrics import accuracy_score

print("model accuracy for train is  {1} and for test is  {0}".format(accuracy_score(y_test,y_pred_test),accuracy_score(y_train,y_pred_train)))
from sklearn.metrics import confusion_matrix
confusion_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred_test))
confusion_matrix.columns = ['Predcited_Died','Predcited_Survived']
confusion_matrix.index = ['Actual_Died','Actual_Survived']
print(confusion_matrix)
def model_score(ytest,ypredtest):
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    Accuracy = accuracy_score(ytest,ypredtest)
    precision = precision_score(ytest,ypredtest)
    recall = recall_score(ytest,ypredtest)
    F1_Score = f1_score(ytest,ypredtest)
    print(" Accuracy is {0} precision is {1}  Recall is {2}  F1_Score is {3}".format(Accuracy,precision,recall,F1_Score))
model_score(y_test,y_pred_test)
from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier()
model_rfc.fit(X_train,y_train)
y_test_pred_rfc = model_rfc.predict(X_test)
y_train_pred_rfc = model_rfc.predict(X_train)
def confusion_matrix(test,pred_test):
    from sklearn.metrics import confusion_matrix
    confusion_matrix = pd.DataFrame(confusion_matrix(test,pred_test))
    confusion_matrix.columns = ['predicted_dead','predictes_survived']
    confusion_matrix.index = ['Actual_dead','Actual_Survived']
    print(confusion_matrix)
confusion_matrix(y_test,y_test_pred_rfc)
model_score(y_test,y_test_pred_rfc)
from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression()
model_logistic.fit(X_train,y_train)
y_test_pred_logic = model_logistic.predict(X_test)
confusion_matrix(y_test,y_test_pred_logic)
model_score(y_test,y_test_pred_logic)
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)

y_test_pred_SVC = clf.predict(X_test)
confusion_matrix(y_test,y_test_pred_SVC)
model_score(y_test,y_test_pred_SVC)
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
Titanic = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

Titanic.columns
Titanic.head()
Titanic.tail()
Titanic.shape
Titanic.info()
Titanic.describe()
Titanic.describe(include=['O'])
Titanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
Titanic[["Sex" , "Survived"]].groupby(["Sex"] , as_index = False).mean().sort_values(by="Survived" , ascending = False)
Titanic[["Parch" , "Survived"]].groupby(["Parch"] , as_index = False) .mean().sort_values(by="Survived" , ascending = False)
Titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
Titanic[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

Titanic.hist(bins=10,figsize=(9,7),grid=False)
print("No of Passengers in original data:  " , str(len(Titanic.index)))
sns.countplot(x="Survived" , data=Titanic , color="orange")
sns.countplot(x="Survived", hue="Parch",data=Titanic)
sns.countplot(x="Survived" ,hue="Sex" , data=Titanic )
sns.countplot(x="Survived" ,hue="Pclass" , data=Titanic )
sns.countplot(x="Survived" ,hue="Embarked" , data=Titanic )
sns.countplot(x="Survived", hue="SibSp",data=Titanic)
Titanic["Parch"].plot.hist( figsize=(5,4))
Titanic["SibSp"].plot.hist()
Titanic.isnull()
Titanic.isnull().sum()
sns.heatmap(Titanic.isnull())
null_var = Titanic.isnull().sum()/Titanic.shape[0] *100
null_var
drop_column = null_var[null_var >20].keys()
drop_column
##null_var = Titanic_datA.isnull().sum()/Titanic_datA.shape[0] *100
#null_var
#drop_column = null_var[null_var >20].keys()
#drop_column
##null_var = Titanic_datA.isnull().sum()/Titanic_datA.shape[0] *100
#null_var
N_Titanic_datA = Titanic.drop(columns = drop_column)
Titanic_copy = Titanic.copy()
Titanic_copy2 = Titanic.copy()
Titanic_Deep = Titanic_copy.copy()
sns.heatmap( N_Titanic_datA.isnull())
N_Titanic_datA.isnull().sum()/Titanic_Deep.shape[0] *100
N_Titanic_datAA = N_Titanic_datA.dropna()
sns.heatmap( N_Titanic_datAA.isnull())
Categorical_Values = N_Titanic_datAA.select_dtypes(include=["object"]).columns
Categorical_Values_test = test.select_dtypes(include=["object"]).columns
Numarical_Values = N_Titanic_datAA.select_dtypes(include=['int64','float64']).columns
Numarical_Values_test = test.select_dtypes(include=['int64','float64']).columns
test.shape
def cat_var_dist(var):
    return pd.concat([Titanic_Deep[var].value_counts()/Titanic_Deep.shape[0] * 100, 
          N_Titanic_datAA[var].value_counts()/N_Titanic_datAA.shape[0] * 100], axis=1,
         keys=[var+'_org', var+'clean'])
    
cat_var_dist("Ticket")
Imputer_mean = SimpleImputer(strategy='mean')

Imputer_mean.fit(Titanic_Deep[Numarical_Values])

Imputer_mean.statistics_
Imputer_mean.transform(Titanic_Deep[Numarical_Values])
Titanic_Deep[Numarical_Values] = Imputer_mean.transform(Titanic_Deep[Numarical_Values])
nnnn = Titanic_Deep[Numarical_Values]
Titanic_Deep[Numarical_Values].isnull().sum()
Imputer_mean = SimpleImputer(strategy='most_frequent')
Titanic_Deep[Categorical_Values] = Imputer_mean.fit_transform(Titanic_Deep[Categorical_Values])
Titanic_Deep[Categorical_Values].isnull().sum()
New_Titanic_datA = pd.concat([Titanic_Deep[Numarical_Values] , Titanic_Deep[Categorical_Values]] , axis=1)

New_Titanic_datA.isnull().sum()
skip_column = null_var[null_var >20].keys()
skip_column

Nn_Titanic_datA = Titanic_copy.drop(columns = skip_column)

Titanic_mean = Nn_Titanic_datA.fillna(Nn_Titanic_datA.mean())
Titanic_mean = Titanic_mean.dropna()

print(Titanic_mean.isnull().sum())
Titanic_median = Nn_Titanic_datA.fillna(Nn_Titanic_datA.median())
test_median =  test.fillna(test.median())
Titanic_median = Titanic_median.dropna()
Titanic_median.isnull().sum()
print("*"*30 , "Data Cleaning Using Different Method" , "*"*30)
print("*"*30 , "Simple Row Delete Mehtod" , "*"*30)
print(N_Titanic_datAA.isnull().sum())
print("*"*30 , "SimpleImputer Method" , "*"*30)
print(New_Titanic_datA.isnull().sum())
print("*"*30 , "Median" , "*"*30)
print(Titanic_median.isnull().sum())
print("*"*30 , "Mean" , "*"*30)
print(Titanic_mean.isnull().sum())
N_Titanic_datAA.tail()
sex = pd.get_dummies(N_Titanic_datAA["Sex"] , drop_first=True)
sexx =  pd.get_dummies(test_median["Sex"] , drop_first=True)
pclass = pd.get_dummies(N_Titanic_datAA["Pclass"] , drop_first=True)
pclasss = pd.get_dummies(test_median["Pclass"] , drop_first=True)
embarked = pd.get_dummies(N_Titanic_datAA["Embarked"] , drop_first=True)
embarkedd = pd.get_dummies(test_median["Embarked"] , drop_first=True)
N_Titanic_datAA_copy = N_Titanic_datAA.copy()
N_Titanic_datAA_copy.drop(['Embarked', 'Pclass' ,"Sex" , "Ticket" , "Name"], axis=1 , inplace=True)
test_median.drop(['Embarked', 'Pclass' ,"Sex" , "Ticket" , "Name"], axis=1 , inplace=True)
N_Titanic_datAA_copy = pd.concat([N_Titanic_datAA_copy ,sex ,pclass ,embarked] ,axis=1)
N_Titanic_datAA_copy.head()
test_median = pd.concat([test_median ,sexx ,pclasss ,embarkedd] ,axis=1)
test_median.head()
test_median.drop(["Cabin"], axis=1 , inplace=True)
test1= test_median.copy()
test_median.head()
X = N_Titanic_datAA_copy.drop("Survived" , axis=1)
y = N_Titanic_datAA_copy["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y.shape
#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train,y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
#LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(logreg.score(X_train, y_train) * 100, 2)
acc_logreg
models = pd.DataFrame({
    'Model': [ 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree'],
    'Score': [ acc_knn, acc_logreg, 
              acc_random_forest, acc_gaussian,
              acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
New_Titanic_datA.head()

embarked = pd.get_dummies(New_Titanic_datA["Embarked"] , drop_first=True)
pclass = pd.get_dummies(New_Titanic_datA["Pclass"] , drop_first=True)
sex = pd.get_dummies(New_Titanic_datA["Sex"] , drop_first=True)

New_Titanic_datA_copy= New_Titanic_datA.copy()
New_Titanic_datA_copy.drop(['Embarked', 'Pclass' ,"Sex" , "Ticket" , "Name"], axis=1 , inplace=True)
New_Titanic_datA_copy = pd.concat([New_Titanic_datA_copy ,sex ,pclass ,embarked] ,axis=1)
New_Titanic_datA_copy.head()
XXX = New_Titanic_datA_copy.drop("Survived" , axis=1)
yyy = New_Titanic_datA_copy["Survived"]
XXX_train, XXX_test, yyy_train, yyy_test = train_test_split(XXX, yyy, test_size=0.2, random_state=42)
#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(XXX_train, yyy_train)
yyy_pred = knn.predict(XXX_test)
acc_knn = round(knn.score(XXX_train, yyy_train) * 100, 2)
acc_knn
#LogisticRegression
logreg = LogisticRegression()
logreg.fit(XXXX_train, yyyy_train)
yyy_pred = logreg.predict(XXXX_test)
acc_logreg = round(logreg.score(XXXX_train, yyyy_train) * 100, 2)
acc_logreg
# Random Forest

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(XXX_train, yyy_train)
yyy_pred = random_forest.predict(XXX_test)
random_forest.score(XXX_train, yyy_train)
acc_random_forest = round(random_forest.score(XXX_train, yyy_train) * 100, 2)
acc_random_forest
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(XXX_train, yyy_train)
yyy_pred = gaussian.predict(XXX_test)
acc_gaussian = round(gaussian.score(XXX_train, yyy_train) * 100, 2)
acc_gaussian
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(XXX_train, yyy_train)
yyy_pred = decision_tree.predict(XXX_test)
acc_decision_tree = round(decision_tree.score(XXX_train, yyy_train) * 100, 2)
acc_decision_tree
models = pd.DataFrame({
    'Model': [ 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree'],
    'Score': [ acc_knn, acc_logreg, 
              acc_random_forest, acc_gaussian,
              acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
Titanic_median.head()
embarked = pd.get_dummies(Titanic_median["Embarked"] , drop_first=True)
pclass = pd.get_dummies(Titanic_median["Pclass"] , drop_first=True)
sex = pd.get_dummies(Titanic_median["Sex"] , drop_first=True)

Titanic_median_copy= Titanic_median.copy()
Titanic_median_copy.drop(['Embarked', 'Pclass' ,"Sex" , "Ticket" , "Name"], axis=1 , inplace=True)


Titanic_median_copy = pd.concat([Titanic_median_copy ,sex ,pclass ,embarked] ,axis=1)
Titanic_median_copy.head()
XX = Titanic_median_copy.drop("Survived" , axis=1)
yy = Titanic_median_copy["Survived"]
XXX_train, XXX_test, yyy_train, yyy_test = train_test_split(XX, yy, test_size=0.2, random_state=42)
XXX_train.shape
#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(XXX_train, yyy_train)
yyy_pred = knn.predict(XXX_test)
acc_knn = round(knn.score(XXX_train, yyy_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(XXX_train, yyy_train)
yyy_pred = gaussian.predict(XXX_test)
acc_gaussian = round(gaussian.score(XXX_train, yyy_train) * 100, 2)
acc_gaussian
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(XXX_train, yyy_train)
yyy_pred = decision_tree.predict(XXX_test)
acc_decision_tree = round(decision_tree.score(XXX_train, yyy_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(XXX_train, yyy_train)
yyy_pred = random_forest.predict(XXX_test)
random_forest.score(XXX_train, yyy_train)
acc_random_forest = round(random_forest.score(XXX_train, yyy_train) * 100, 2)
acc_random_forest
#LogisticRegression
logreg = LogisticRegression()
logreg.fit(XXX_train, yyy_train)
yyy_pred = logreg.predict(XXX_test)
acc_logreg = round(logreg.score(XXX_train, yyy_train) * 100, 2)
acc_logreg
models = pd.DataFrame({
    'Model': [ 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree'],
    'Score': [ acc_knn, acc_logreg, 
              acc_random_forest, acc_gaussian,
              acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
test_median.isnull().sum()
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
from sklearn.svm import SVC

svc = SVC()

# Gradient Boosting Classifier
gbkk = GradientBoostingClassifier()
gbkk.fit(X_train, y_train)
gbk_pred = gbkk.predict(X_test)
acc_gbkk = round(gbkk.score(X_train, y_train) * 100, 2)
predictions = gbkk.predict(test_median)
acc_gbkk
gbk_pred = gbkk.predict(X_test)
gbk_pred
sgd.fit(XXX_train, yyy_train)
yyy_pred = sgd.predict(XXX_test)
acc_gbk = round(sgd.score(XXX_train, yyy_train) * 100, 2)
acc_gbk
perceptron.fit(XXX_train, yyy_train)
yyy_pred = perceptron.predict(XXX_test)
acc_gbk = round(perceptron.score(XXX_train, yyy_train) * 100, 2)
acc_gbk
linear_svc.fit(XXX_train, yyy_train)
yyy_pred = linear_svc.predict(XXX_test)
acc_gbk = round(linear_svc.score(XXX_train, yyy_train) * 100, 2)
acc_gbk
svc.fit(XXX_train, yyy_train)
yyy_pred = svc.predict(XXX_test)
acc_gbk = round(svc.score(XXX_train, yyy_train) * 100, 2)
acc_gbk
#set ids as PassengerId and predict survival 
ids = test1['PassengerId']
predictions = gbkk.predict(test_median)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

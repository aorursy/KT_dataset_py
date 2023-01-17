import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
train=pd.read_csv("../input/titanic/train.csv")
test=pd.read_csv("../input/titanic/test.csv")
test.head()
train.head()
train.describe()
fig,axes= plt.subplots(1,2)
fig.tight_layout()

sns.countplot(x='Survived',hue='Sex',data=train,ax=axes[0])
sns.countplot(x='Survived',hue='Pclass',data=train,ax=axes[1])

plt.show()
fig = plt.subplots(figsize=(10,10)) 
sns.heatmap(train.corr(), annot=True)
columnas_importantes= ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']
train_new=train[columnas_importantes]
train_new.isnull().any()
age_null=train_new["Age"].isnull().sum()
embarked_null=train_new["Embarked"].isnull().sum()

print("La cantidad de nulos en Age es:",age_null)
print("La cantidad de nulos en Embarked es:",embarked_null)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train_new)
def imputar_Edad(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train_new['Age'] = train_new[['Age','Pclass']].apply(imputar_Edad,axis=1)
train_new['Embarked'] = train_new['Embarked'].fillna(train_new["Embarked"].mode()[0])
train_new.isnull().sum()
train_new.head()
train_new = pd.get_dummies(data=train_new,prefix="Embarked_",columns=["Embarked"])
train_new = pd.get_dummies(data=train_new,prefix="Sex_",columns=["Sex"])
train_new.loc[ train_new['Fare'] <= 7.91, 'Fare'] = 0
train_new.loc[(train_new['Fare'] > 7.91) & (train_new['Fare'] <= 14.454), 'Fare'] = 1
train_new.loc[(train_new['Fare'] > 14.454) & (train_new['Fare'] <= 31), 'Fare']   = 2
train_new.loc[ train_new['Fare'] > 31, 'Fare']= 3
train_new['Fare'] = train_new['Fare'].astype(int)
train_new.head()
test.head()
columnas_importantes_test= ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']
test2=test.copy()
test_new= test2[columnas_importantes_test]
test.head()
test_new.isnull().sum()
test_new['Age'] = test_new[['Age','Pclass']].apply(imputar_Edad,axis=1)
test_new['Fare'].fillna(35.6271,inplace=True)
test_new.isnull().sum()
test_new = pd.get_dummies(data=test_new,prefix="Embarked_",columns=["Embarked"])
test_new = pd.get_dummies(data=test_new,prefix="Sex_",columns=["Sex"])
test_new.loc[ test_new['Fare'] <= 7.91, 'Fare'] = 0
test_new.loc[(test_new['Fare'] > 7.91) & (test_new['Fare'] <= 14.454), 'Fare'] = 1
test_new.loc[(test_new['Fare'] > 14.454) & (test_new['Fare'] <= 31), 'Fare']   = 2
test_new.loc[ test_new['Fare'] > 31, 'Fare']= 3
test_new['Fare'] = test_new['Fare'].astype(int)
test_new.head()
test.head()
from sklearn.metrics import accuracy_score
X= train_new.drop(columns = ['Survived',"Name","Ticket"])
y=train_new['Survived']
testeo= test_new.drop(columns=["Name","Ticket"])
from sklearn.preprocessing import RobustScaler
rscl=RobustScaler()
X= rscl.fit_transform(X)
testeo=rscl.fit_transform(testeo)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X,y)
tree_predict= tree.predict(testeo)

tree.score(X,y)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X,y)
Prediction_KNC = knn.predict(testeo)

knn.score(X,y)
MAX_DEPTH_range = range(1, 20)
scores = []
for k in MAX_DEPTH_range:
    titanic_tree = DecisionTreeClassifier(max_depth=k)
    titanic_tree.fit(X,y)
    scores.append(titanic_tree.score(X,y))
plt.figure()
plt.xlabel('max depths')
plt.ylabel('accuracy')
plt.scatter(MAX_DEPTH_range, scores)
plt.xticks([0,5,10,15,20])
k_range = range(1, 20)
scores = []
for k in k_range:
    titanic_knn = KNeighborsClassifier(n_neighbors = k)
    titanic_knn.fit(X,y)
    scores.append(titanic_knn.score(X,y))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])    
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=16)
tree.fit(X,y)
tree_predict= tree.predict(testeo)

tree.score(X,y)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X,y)
Prediction_KNC = knn.predict(testeo)

knn.score(X,y)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": tree_predict
    })
submission.to_csv('Resultados_titanic.csv', index=False)
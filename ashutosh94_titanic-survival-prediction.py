import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../input/titanictrain.csv')
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC

df.head(10)
df.info()
df.drop(["PassengerId","Name","Ticket","Fare","Cabin"],axis=1,inplace=True)

df.info()
df.head(7)
df['Sex'].unique()
gendermap = {'male' : 1 , 'female' : 0 }
df['Sex'] = df['Sex'].map(gendermap)
df.head(4)
df['Embarked'].value_counts()
df['Embarked'].isnull().sum()
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'].isnull().sum()
df['Age'].isnull().sum()
df['Age'].describe()
median_age= df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
df['Age'].isnull().sum()
df['familysize'] = df['SibSp'] + df['Parch'] +1 
df['familysize'].describe()
df.head(5)
embarked_mapping = { 'S' : 1 , 'C' : 2 , 'Q' : 3}
df['emb_cat'] = df['Embarked'].map(embarked_mapping)
df.head(8)
df['Age'] = df['Age'].astype('int64')
    

df.head(5)
def ageband(x) :
    if x in range(0,15):
        return 1 
    elif x in range(15,30):
        return 2
    elif x in range(30,60):
        return 3 
    elif x >= 60 :
        return 4

df['ageband'] = df['Age'].apply(ageband)
df.info()
df.drop(['SibSp','Parch','Embarked'],axis=1,inplace = True)
df.head(5)
corr_matrix = df.corr()
corr_matrix['Survived'].sort_values(ascending = False)
features = ['Pclass','Sex','Age','familysize','emb_cat','ageband']
label = ['Survived']
X = df[features]
y = df[label]
from sklearn import preprocessing,cross_validation
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy*100)
ex = np.array([1,1,38,2,2,3])
ex = ex.reshape(1,-1)
prediction = clf.predict(ex)
print(prediction)
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
accuracy = tree.score(X_test,y_test)
print(accuracy*100)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
accuracy = knn.score(X_test,y_test)
print(accuracy*100)
svc = SVC()
svc.fit(X_train,y_train)
accuracy = svc.score(X_test,y_test)
print(accuracy*100)
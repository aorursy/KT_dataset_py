import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titan = pd.read_csv("../input/titanic/train.csv")
titan.info()
titan.head()
sns.heatmap(titan.isnull())
titan = titan.drop('Cabin',axis = 1)
titan = titan.drop(labels = ['Name','Ticket'],axis = 1)
titan.head()
def clean_gender(x):
    if x == 'male':
        return 1
    else:
        return 0
titan['Is Male'] = titan["Sex"].apply(lambda x : clean_gender(x))
titan = titan.drop('Sex',axis = 1)
titan.head()
titan['SibSp'].unique()
sns.countplot(x = 'Survived',data = titan,hue = 'Pclass')
sns.countplot(x = 'Is Male',data = titan,hue = 'Survived')
g = sns.FacetGrid(titan, col = 'Survived',size = 5)
g.map(sns.distplot,'Fare')
plt.figure(figsize=(12,7))
sns.countplot(x = 'Survived',data = titan,hue = 'Parch')
fig = plt.figure(figsize=(12,7))
sns.boxplot(x = 'Pclass',y = 'Age',data = titan)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age
titan["Age"] = titan[["Age","Pclass"]].apply(impute_age,axis = 1)
sns.heatmap(titan.isnull())
sns.countplot(x = 'Survived',data = titan,hue = 'Embarked')
embark = pd.get_dummies(titan['Embarked'],drop_first=True)
titan = titan.drop('Embarked',axis = 1)
titan = pd.concat([titan,embark],axis = 1)
titan.head()
train = titan
test = pd.read_csv("../input/titanic/test.csv")
test.head()
test = test.drop("Name Cabin Ticket".split(),axis = 1)
test.head()
test["Sex"] = test["Sex"].apply(lambda x : clean_gender(x))
test.rename(columns={'Sex' : 'Is Male'},inplace = True)
test.head()
embarked = pd.get_dummies(test['Embarked'],drop_first=True)

test = test.drop('Embarked',axis = 1)
test = pd.concat([test,embarked],axis = 1)
test.head()
train.head()
sns.heatmap(test.isnull())
fig = plt.figure(figsize=(12,7))
sns.boxplot(x = 'Pclass', y = "Age",data = test)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 25

        else:
            return 23

    else:
        return Age
test["Age"] = test[["Age","Pclass"]].apply(impute_age,axis = 1)
sns.heatmap(test.isnull())
mean_fare = test["Fare"].mean()
def cleanfare(x):
    if pd.isnull(x):
        return mean_fare
    else:
        return x
test["Fare"] = test["Fare"].apply(lambda x : cleanfare(x))
sns.heatmap(test.isnull())
sns.heatmap(train.isnull())
train = train[["PassengerId","Pclass","Is Male","Age","SibSp","Parch","Fare","Q","S","Survived"]]
train.head()
test.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
train[train.columns] = scaled_train
train.head()
scaled_test = scaler.fit_transform(test)
test[test.columns] = scaled_test
X = train.drop('Survived',axis = 1)
Y = train["Survived"]
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X,Y)
predictions = logmodel.predict(test)
test = pd.read_csv("../input/titanic/test.csv")
submission = pd.DataFrame(data = predictions,columns = ["Survived"])
submission["PassengerId"] = test["PassengerId"]
submission = submission[["PassengerId","Survived"]]
submission["Survived"] = submission["Survived"].apply(lambda x : np.int32(x))
submission.head()
submission.to_csv("./kaggle_submission.csv",index = False)
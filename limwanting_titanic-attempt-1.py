# Import necessary packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load data

training_set=pd.read_csv('C:/Users/Wan Ting/Desktop/Kaggle_titanic/train.csv',index_col=False)
test_set=pd.read_csv('C:/Users/Wan Ting/Desktop/Kaggle_titanic/test.csv')
# Data preprocessing step
print(training_set.info())
print(test_set.info())
training_set=training_set[pd.notnull(training_set['Embarked'])]
training_set.info()
# Creating some charts for visualisation
sns.catplot(x="Survived",y="Age",kind='swarm',hue='Pclass',data=training_set,palette='pastel')
print("unique values for Embarked are %s" %str(training_set['Embarked'].unique()))
training_set=pd.concat((training_set,pd.get_dummies(training_set['Embarked'])),axis=1)
training_set.drop('Embarked',axis=1,inplace=True)
training_set.head()
training_set.info()
imputevalues=training_set.groupby(['Pclass', 'Sex'])['Age'].median()
imputevalues=imputevalues.to_dict()
print(imputevalues)
# Let's create a new column so that we can use the keyvalue from the dictionary that was created to impute the values 
training_set["Pclass_Sex"] =(training_set.loc[:,["Pclass","Sex"]]).apply(tuple, axis=1)

# the function below maps the null values in Age column using the Pclass_Sex column and the dictionary we created
training_set.Age = training_set.Age.fillna(training_set.Pclass_Sex.map(imputevalues))
print("unique values for Sex are %s" %str(training_set['Sex'].unique()))
training_set=pd.concat((training_set,pd.get_dummies(training_set['Sex'])),axis=1)
training_set.drop('Sex',axis=1,inplace=True)
training_set=training_set.drop("Cabin",axis=1)

training_set=training_set.drop(["PassengerId","Name","Ticket","Pclass_Sex"],axis=1)
training_set.info()
from sklearn import ensemble
from sklearn import model_selection

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

y=training_set["Survived"]
X=training_set.drop("Survived",axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)

print(training_set.info())
print(test_set.info())
print("unique values for Embarked are %s" %str(test_set['Embarked'].unique()))
test_set=pd.concat((test_set,pd.get_dummies(test_set['Embarked'])),axis=1)
test_set.drop('Embarked',axis=1,inplace=True)
# Let's create a new column so that we can use the keyvalue from the dictionary that was created to impute the values 
test_set["Pclass_Sex"] =(test_set.loc[:,["Pclass","Sex"]]).apply(tuple, axis=1)

# the function below maps the null values in Age column using the Pclass_Sex column and the dictionary we created
test_set.Age = test_set.Age.fillna(test_set.Pclass_Sex.map(imputevalues))
test_set.info()
print("unique values for Sex are %s" %str(test_set['Sex'].unique()))
test_set=pd.concat((test_set,pd.get_dummies(test_set['Sex'])),axis=1)
test_set.drop('Sex',axis=1,inplace=True)
test_set=test_set.drop("Cabin",axis=1)

test_set=test_set.drop(["PassengerId","Name","Ticket","Pclass_Sex"],axis=1)
test_set.info()
test_set.fillna(test_set.mean(),inplace=True)

test_set.info()
y_predict_test = model.predict(test_set)

submission=pd.DataFrame(y_predict_test,columns=['Survived'])
submission_export=pd.concat((pd.DataFrame(list(range(892, 1310))),submission),axis=1)
submission_export.columns=['PassengerId','Survived']
submission_export.to_csv('export.csv',index=False)

# Imports

import numpy as np
import pandas as pd 
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
from sklearn import tree
%matplotlib inline
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import impute
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer

#Files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_table = pd.read_csv("/kaggle/input/titanic/train.csv")
test_table = pd.read_csv("/kaggle/input/titanic/test.csv")

print(train_table.shape)
print(test_table.shape)

plt.figure(figsize= (15,5))
plt.title("Null Values")
sns.barplot(x = train_table.columns,y = train_table.isnull().sum())
sns.barplot(x = test_table.columns,y = test_table.isnull().sum())
test_table.describe()
test_table.dtypes

test_table.head(20)
train_table["Cabin"].unique()
#way too many. Goodbye cabin!
train_table = train_table.drop(columns = ["Cabin","Ticket","Name"])
test_table = test_table.drop(columns = ["Cabin","Ticket","Name"])
X = train_table.drop(columns = ["Survived","PassengerId"])
y = train_table.Survived
sex_converter = {"male":0,"female":1}
print(X)
print(test_table)
X.Sex = X.Sex.map(sex_converter)
test_table.Sex =test_table.Sex.map(sex_converter)
X = pd.concat([X.drop(columns = "Embarked"),pd.get_dummies(X["Embarked"])],axis = 1)
print(X)
test_table = pd.concat([test_table.drop(columns = "Embarked"),pd.get_dummies(test_table["Embarked"])],axis = 1)

print(test_table)
si = SimpleImputer(strategy = "mean")
X = si.fit_transform(X)
X = pd.DataFrame(X)
X.columns = ["Pclass","Sex","Age","SibSp","Parch","Fare","C","Q","S"]
print(X.head(20))
#X.Age = pd.Series(si.fit_transform(X.Age))
test_table_X = test_table.drop(columns = "PassengerId")
test_table_X = si.fit_transform(test_table_X)
test_table_X = pd.DataFrame(test_table_X)
test_table_X.columns = ["Pclass","Sex","Age","SibSp","Parch","Fare","C","Q","S"]
print(test_table_X)
rfg1 = RandomForestRegressor(n_estimators = 50)
rfg2 = RandomForestRegressor(n_estimators = 25)
rfg3 = RandomForestRegressor(n_estimators = 75)
scores1 = -1*cross_val_score(rfg1,X,y,cv = 8, scoring = "neg_mean_absolute_error")
total1 = 0 
for i in scores1:
    total1 = total1 +i
scores2 = -1*cross_val_score(rfg2,X,y,cv = 8, scoring = "neg_mean_absolute_error")
total2 = 0 
for i in scores2:
    total2 = total2 + i 
scores3 = -1*cross_val_score(rfg3,X,y,cv = 8, scoring = "neg_mean_absolute_error")
total3 = 0 
for i in scores3:
    total3 = total3 + i 
print(total1/len(scores1))
print(total2/len(scores2))
print(total3/len(scores3))
clf = tree.DecisionTreeClassifier(max_depth = 8)
clf = clf.fit(X,y)
score_clf = -1*cross_val_score(clf,X,y,cv = 8, scoring = "neg_mean_absolute_error")
total_clf = 0
for i in score_clf:
    total_clf = total_clf + i
    total_clf = total_clf / len(score_clf)
print(total_clf)
#depth 20 is 1.759
#0 is 1.75
#submission = open("submission_file.csv","w")
predictions_final = pd.DataFrame() 
predictions_final = pd.concat([test_table.PassengerId, pd.Series(clf.predict(test_table_X))],axis = 1)
#pd.Series(test_table.PassengerId) + pd.Series(clf.predict(test_table_X))

predictions_final.columns = ["PassengerId","Survived"]
predictions_final.to_csv("submission_file.csv",index=False)

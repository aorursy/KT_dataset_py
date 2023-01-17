import numpy as np
import pylab as py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
my_data = pd.read_csv("../input/titanic/train.csv")
my_data.head(7)
my_test = pd.read_csv("../input/titanic/test.csv")
my_test.head()
my_test = my_test.dropna()
my_data = my_data.dropna()
my_test.columns
my_data.columns
# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
my_data['Sex']= le.fit_transform(my_data['Sex'])
X_train = my_data[['PassengerId','Sex', 'Age','Pclass','SibSp','Parch']]
X_train[0:7]
# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
  
he = LabelEncoder() 
  
my_test['Sex']= he.fit_transform(my_test['Sex'])
X_test = my_data[['PassengerId','Sex', 'Age','Pclass','SibSp','Parch']]
X_test[0:5]
y_train = my_data[['Survived']]
y_train[0:5]
y_test = my_data[['Survived']]
y_test[0:5]
from sklearn.model_selection import train_test_split
print("Train Set:",X_train.shape,y_train.shape)
print("Test Set:",X_test.shape,y_test.shape)
ShipTree=DecisionTreeClassifier(criterion = 'entropy',max_depth = 4)
ShipTree.fit(X_train,y_train)
ShipTree=ChurnTree.predict(X_test)
print(y_test[0:5])
print(ShipTree[0:5])
from sklearn import metrics
import matplotlib.pyplot as plt
print("The Accuracy Score : ",metrics.accuracy_score(y_test,ShipTree))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
men = my_data.loc[my_data.Sex == 1]["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
count=0
for males in ShipTree:
    if males == 0:
        count = count+1
        
pred_rate_male = count/len(ShipTree)
print("% of men who survived (predicted by model):", pred_rate_male)
female = my_data.loc[my_data.Sex == 0]["Survived"]
rate_female = sum(female)/len(female)

print("% of female who survived:", rate_female)
countf=0
for females in ShipTree:
    if females == 1:
        countf = countf+1
        
pred_rate_female = countf/len(ShipTree)
print("% of females who survived (predicted by model):", pred_rate_female)
output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': ShipTree})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

X_test.shape

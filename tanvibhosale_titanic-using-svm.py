import numpy as np
import pandas as pd
train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head(10)
train_data.isnull().sum()
train_data["Age"].fillna(train_data["Age"].mode(),inplace=True)
train_data["Age"].fillna(train_data["Age"][0],inplace=True)
train_data.isnull().sum()
test_data["Age"].fillna(train_data["Age"].mode(),inplace=True)
test_data["Age"].fillna(train_data["Age"][0],inplace=True)
test_data.isnull().sum()
train_data_1 = train_data.copy()
train_data_1 = train_data.drop(['Name','Ticket','Cabin','Embarked','Fare'],axis=1)
train_data_1.head()
test_data_1 = test_data.copy()
test_data_1 = test_data.drop(['Name','Ticket','Cabin','Embarked','Fare'],axis=1)
test_data_1.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data_1['Sex'] = le.fit_transform(train_data_1['Sex'])
train_data_1.head()
le1 = LabelEncoder()
test_data_1['Sex'] = le1.fit_transform(test_data_1['Sex'])
test_data_1.shape
x = train_data_1.drop(['Survived'],axis=1).values
y = train_data_1.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
test_data_x = sc.fit_transform(test_data_1)
y_pred_1 = classifier.predict(test_data_x)
Y = y_pred_1.reshape(len(y_pred_1),1)
#print(np.concatenate(((test_data_1['PassengerId'],1),y_pred_1.reshape(len(y_pred_1),1)),1))
print(Y)
ans = pd.read_csv('../input/titanic/gender_submission.csv')
ans1 =ans.copy()
ans1.head(17)
ans1['Survived'] = Y
ans1.head()
ans1.to_csv('Gender_prediction.csv',index=False)

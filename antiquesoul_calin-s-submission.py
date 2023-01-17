#Am ales modelul Support Vector Machines pentru ca mi se pare cel mai de incredere din punct de vedere matematic

#(calculeaza distantele si creeaza un plan pentru a delimita cele doua categorii, Survived si Not Survived in cazul nostru)

                                                                                                        
#pentru citire date .csv

import pandas as pd 

from pandas import DataFrame



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.svm import SVC

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data_all= pd.read_csv("/kaggle/input/titanic/test.csv")





#vizualizare tabel cu datele cu care antrenez modelul

train_data.head()
#vizualizare datele din test

test_data_all.head()
#Renunt la coloanele pe care nu le folosesc. Cand am incercat sa folosesc pd.get_dummies, a facut coloana separata pentru fiecare nume

#si incarca tabelul degeaba.

train_data=train_data.drop(['Name','Ticket','Cabin'],axis=1)

test_data_all=test_data_all.drop(['Name','Ticket','Cabin'],axis=1)

train_data.head()
test_data_all.head()
train_data = pd.get_dummies(train_data)

test_data_all= pd.get_dummies(test_data_all)

train_data.head()
test_data_all.head()

#Identific variantele null si le inlocuiesc cu media coloanei respective, ca sa nu afecteze modelul

train_data.isnull().sum().sort_values(ascending=False)
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())




test_data_all.isnull().sum().sort_values(ascending=False)

test_data_all['Age'] = test_data_all['Age'].fillna(test_data_all['Age'].mean())

test_data_all['Fare'] = test_data_all['Fare'].fillna(test_data_all['Fare'].mean())
#Input si Output pentru Model

x= train_data.drop(['PassengerId','Survived'],axis=1)

y=train_data['Survived']
x.head()
y.head()
#Clasificatorul invata din datele de train

svc_classifier= SVC(gamma='auto',random_state=1)

svc_classifier.fit(x,y)
test_data=test_data_all.drop(['PassengerId'],axis=1)

test_data.head()

#metoda total chinuita sa pun PassengerId si rezultatul predictiei in acelasi data frame

results=DataFrame(test_data_all.drop(['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'],axis=1))

results.head()
results1=DataFrame(svc_classifier.predict(test_data))

results1.head()



final_result=pd.concat([results,results1],axis=1)

final_result.head()

print(final_result)
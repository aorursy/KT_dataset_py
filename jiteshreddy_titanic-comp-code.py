import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/titanic/train.csv")

submit_data = pd.read_csv("/kaggle/input/titanic/test.csv")

submit_data_copy = pd.read_csv("/kaggle/input/titanic/test.csv")
X = ['Pclass','Sex','Age','Embarked','Parch']

y = ['Survived']





data_X = data[X]

data_y = data[y]
embarked_size_mapping = {'S' : int(1) , 'C' : int(2) , 'Q' : int(3) }



data_X['Embarked'] = data_X['Embarked'].map(embarked_size_mapping)
data_X['Embarked'] = data_X['Embarked'].convert_dtypes(int)

sex_mapping = {'male' : 0 , 'female' : 1}



data_X['Sex'] = data_X['Sex'].map(sex_mapping)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_X = pd.DataFrame(my_imputer.fit_transform(data_X))
data_X.columns = X
from sklearn.model_selection import train_test_split as tts

X_train , X_test , y_train , y_test = tts(data_X, data_y, test_size =0.1)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop = True)
X_train['Age'] = X_train['Age'].round()
X_train= X_train.convert_dtypes(int)
X_test = X_test.convert_dtypes(int)
X_test['Age'] = X_test['Age'].astype(int)
X_train['Embarked'] = X_train['Embarked'].astype(int)

y_train= y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)
from sklearn.ensemble import RandomForestClassifier





model_1 = RandomForestClassifier(n_estimators= 75, max_depth=8, random_state=1)

model_2 = RandomForestClassifier(n_estimators= 150, max_depth=7, random_state=1)

model_3 = RandomForestClassifier(n_estimators= 200, max_depth=5, random_state=1)

model_4 = RandomForestClassifier(n_estimators= 100, max_depth=5, random_state=1)

model_5 = RandomForestClassifier(n_estimators= 300, max_depth=2, random_state=1)

model_6 = RandomForestClassifier(n_estimators= 400, max_depth=4, random_state=1)
model_1.fit(X_train,y_train.values.ravel())

model_2.fit(X_train,y_train.values.ravel())

model_3.fit(X_train,y_train.values.ravel())

model_4.fit(X_train,y_train.values.ravel())

model_5.fit(X_train,y_train.values.ravel())

model_6.fit(X_train,y_train.values.ravel())
model_list = [model_1, model_2, model_3, model_4, model_5, model_6]
from sklearn.metrics import mean_absolute_error as mae

def score(model_list):

    for i in model_list:

        l=i.predict(X_test)

        x = mae(y_test,l)

        print(x)
score(model_list)
final_model = model_6
submit_data = submit_data[X]
submit_data['Embarked'] = submit_data['Embarked'].map(embarked_size_mapping)

submit_data['Sex'] = submit_data['Sex'].map(sex_mapping)
submit_data.head()
submit_data = pd.DataFrame(my_imputer.fit_transform(submit_data))
submit_data.columns = X
submit_data = submit_data.convert_dtypes(int)
submit_data['Age'] = submit_data['Age'].round(0)
submit_data = submit_data.convert_dtypes(int)
predictions = final_model.predict(submit_data)
predictions_csv = pd.DataFrame({'PassengerId': submit_data_copy.PassengerId,'Survived':predictions})
predictions_csv.head()
predictions_csv.to_csv('Jeetes.v2',index = False)
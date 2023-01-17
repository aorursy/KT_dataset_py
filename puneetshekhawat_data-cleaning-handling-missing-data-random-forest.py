import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#Creating copies to keep original data
data = dataset
test_data = test

#
data = data.drop(['Id'], axis = 1)
test_data = test_data.drop(['Id'], axis=1)
#
data = data.drop(['hacapo'], axis = 1)
test_data = test_data.drop(['hacapo'], axis=1)
#
data = data.drop(['v14a'], axis = 1)
test_data = test_data.drop(['v14a'], axis=1)
#
data = data.drop(['refrig'], axis = 1)
test_data = test_data.drop(['refrig'], axis=1)
# merging v18q and v18q1
data = data.drop(['v18q'], axis = 1)
test_data = test_data.drop(['v18q'], axis=1)
data['v18q1'].fillna(0, inplace = True)
test_data['v18q1'].fillna(0, inplace = True)
#droping unnecessrary columns 
colms_to_be_dropped = ['r4h3', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamviv', 'rez_esc', 'tamhog', 'abastaguadentro']
data = data.drop(colms_to_be_dropped, axis = 1)
test_data = test_data.drop(colms_to_be_dropped, axis = 1)
#dropping columns
colms_to_be_dropped = ['public', 'sanitario1', 'energcocinar1', 'elimbasu1']
data = data.drop(colms_to_be_dropped, axis = 1)
test_data = test_data.drop(colms_to_be_dropped, axis = 1)
#
colms_to_be_dropped =  ['epared1', 'eviv1', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'idhogar']    
data = data.drop(colms_to_be_dropped, axis = 1)
test_data = test_data.drop(colms_to_be_dropped, axis = 1)
#
data['dependency'] = data['dependency'].replace({'yes': 1, 'no': 0}, regex=True)
test_data['dependency'] = test_data['dependency'].replace({'yes': 1, 'no': 0}, regex=True)
data['edjefe'] = data['edjefe'].replace({'yes': 1, 'no': 0}, regex=True)
test_data['edjefe'] = test_data['edjefe'].replace({'yes': 1, 'no': 0}, regex=True)
data['edjefa'] = data['edjefa'].replace({'yes': 1, 'no': 0}, regex=True)
test_data['edjefa'] = test_data['edjefa'].replace({'yes': 1, 'no': 0}, regex=True)
#
colms_to_be_dropped = ['meaneduc', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
data = data.drop(colms_to_be_dropped, axis = 1)
test_data = test_data.drop(colms_to_be_dropped, axis = 1)
cols_with_missing = [col for col in data.columns 
                                 if data[col].isnull().any()]
cols_with_missing_test = [col for col in test_data.columns 
                                 if test_data[col].isnull().any()]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(data[cols_with_missing])
data[cols_with_missing] = imputer.transform(data[cols_with_missing])
test_data[cols_with_missing] = imputer.transform(test_data[cols_with_missing])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Target'], axis = 1),
                                                    data['Target'], test_size = 0.2, 
                                                    random_state = 0)

""""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_data = sc.transform(test_data)"""
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier( criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
ans = classifier.predict(test_data)
my_submission = pd.DataFrame({'Id': test.Id, 'Target': ans})
my_submission.to_csv('submission1.csv', index=False)
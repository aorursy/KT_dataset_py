# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head()
#features_all = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
 #      'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
label = ['Survived']
features = [ 'Pclass', 'Sex', 'SibSp',
       'Parch']#,'PassengerId','Age', 'Embarked']
data =data[label+features]
data.dropna(inplace = True)
data.describe()
data.head()
sex = {'male': 1, 'female': 0}
embark = {'S': 1, 'C': 0}
data['Sex'] = data['Sex'].map(sex)
#data['Embarked'] = data['Embarked'].map(embark)
data.head()
feature_data = np.array(data[features])

label_data = np.array(data[label[0]])

from sklearn import preprocessing
feature_data = preprocessing.StandardScaler().fit_transform(feature_data)
feature_data[0]
missing_val_count_by_column = (data.isnull().sum())
missing_val_count_by_column
test = pd.read_csv("/kaggle/input/titanic/test.csv")
missing_val_count_by_column = (test_data.isnull().sum())
print(missing_val_count_by_column)
test_data =test[features]
#test_data.dropna(inplace = True)
test_data['Sex'] = test_data['Sex'].map(sex)
#test_data['Embarked'] = test_data['Embarked'].map(embark)
missing_val_count_by_column = (test_data.isnull().sum())
print(missing_val_count_by_column)
test_data.describe()

d_test = np.array(test_data[features])
d_test = preprocessing.StandardScaler().fit_transform(d_test)
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(feature_data,label_data)
model = decision_tree
sub = pd.DataFrame()
sub['PassengerId'] = test['PassengerId']
sub['Survived'] = model.predict(d_test)
sub['Survived'] = sub['Survived'].apply(lambda x: 1 if x>0.8 else 0)
# sub['Survived'] = sub.apply(lambda r: leaks[int(r['PassengerId'])] if int(r['PassengerId']) in leaks else r['Survived'], axis=1)
sub.to_csv('/kaggle/working/submission.csv', index=False)

sub.head()

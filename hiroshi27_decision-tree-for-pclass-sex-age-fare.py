import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from sklearn.impute import SimpleImputer

# Input data files are available in the "../input/" directory.
## test data
test_data = pd.read_csv('../input/test.csv')
x_data = test_data[['Pclass','Sex','Age','Fare']] #need ticket to improve
x_data['Sex'] = x_data['Sex'].map({'female':0, 'male':1})

x_test = np.array(x_data)

imputer = SimpleImputer()
x_test = imputer.fit_transform(x_test)
print(x_test)
## Train data
data = pd.read_csv('../input/train.csv')
x_data = data[['Pclass','Sex','Age','Fare']] #need ticket to improve
x_data['Sex'] = x_data['Sex'].map({'female':0, 'male':1})

y_train = np.array(data['Survived'])
x_train = np.array(x_data)

imputer = SimpleImputer()
x_train = imputer.fit_transform(x_train)
print(x_train)
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
type(test_data.PassengerId)
my_pred = clf.predict(x_test)
my_submission = pd.DataFrame({'id':test_data.PassengerId,'survive':my_pred})
my_submission.to_csv('/Users/Hiroshi/submission.csv', index=False)

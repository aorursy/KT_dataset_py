import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
test=pd.read_csv("/kaggle/input/datathon19/test.csv")

test.head()
train=pd.read_csv("/kaggle/input/datathon19/train.csv")

train.head()
test.describe()
train.describe()
x_train = train.iloc[:,0:9]

y_train = train['class']

x_train.head()
x_test=test.iloc[:,0:9]

x_test.head()
from sklearn.preprocessing import LabelEncoder

encd = LabelEncoder()
x_train['top_left_square'] = encd.fit_transform(x_train['top_left_square'])

x_train['top_middle_square'] = encd.fit_transform(x_train['top_middle_square'])

x_train['top_right_square'] = encd.fit_transform(x_train['top_right_square'])

x_train['middle_left_square'] = encd.fit_transform(x_train['middle_left_square'])

x_train['middle_middle_square'] = encd.fit_transform(x_train['middle_middle_square'])

x_train['middle_right_square'] =encd.fit_transform(x_train['middle_right_square'])

x_train['bottom_left_square'] = encd.fit_transform(x_train['bottom_left_square'])

x_train['bottom_middle_square'] = encd.fit_transform(x_train['bottom_middle_square'])

x_train['bottom_right_square'] = encd.fit_transform(x_train['bottom_right_square'])
x_test['top_left_square'] = encd.fit_transform(x_test['top_left_square'])

x_test['top_middle_square'] = encd.fit_transform(x_test['top_middle_square'])

x_test['top_right_square'] = encd.fit_transform(x_test['top_right_square'])

x_test['middle_left_square'] = encd.fit_transform(x_test['middle_left_square'])

x_test['middle_middle_square'] = encd.fit_transform(x_test['middle_middle_square'])

x_test['middle_right_square'] = encd.fit_transform(x_test['middle_right_square'])

x_test['bottom_left_square'] = encd.fit_transform(x_test['bottom_left_square'])

x_test['bottom_middle_square'] = encd.fit_transform(x_test['bottom_middle_square'])

x_test['bottom_right_square'] = encd.fit_transform(x_test['bottom_right_square'])
x_train.head()
x_test.head()
x_test_pred = x_test.iloc[:, 0:9]

x_test_pred.head()
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='entropy')

model.fit(x_train, y_train)
y_pred = model.predict(x_test_pred)
y_pred
id =test['Id']
np.random.seed(123)

e = np.random.normal(size=10)  

pred=pd.DataFrame(y_pred, columns=['prediction']) 

print (pred)
pred = np.vstack((id,y_pred)).T

pred
np.savetxt('DatathonResult.csv', pred, delimiter=',', fmt="%i")
import csv

with open('DatathonResult.csv',newline='') as f:

    r = csv.reader(f)

    data = [line for line in r]

with open('DatathonResult.csv','w',newline='') as f:

    w = csv.writer(f)

    w.writerow(['Id','class'])

    w.writerows(data)
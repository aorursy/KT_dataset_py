import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
data_test=pd.read_csv("../input/test.csv")

data_test.head()
data_train=pd.read_csv("../input/train.csv")

data_train.head()
data_test.describe()
data_train.describe()
y_Train = data_train['class']

x_Train = data_train.iloc[:,0:9]

x_Train.head()
x_test=data_test.iloc[:,0:9]

x_test.head()
print(x_Train.shape)

print(y_Train.shape)
from sklearn.preprocessing import LabelEncoder

LEn = LabelEncoder()
x_Train['top_left_square'] = LEn.fit_transform(x_Train['top_left_square'])

x_Train['top_middle_square'] = LEn.fit_transform(x_Train['top_middle_square'])

x_Train['top_right_square'] = LEn.fit_transform(x_Train['top_right_square'])

x_Train['middle_left_square'] = LEn.fit_transform(x_Train['middle_left_square'])

x_Train['middle_middle_square'] = LEn.fit_transform(x_Train['middle_middle_square'])

x_Train['middle_right_square'] = LEn.fit_transform(x_Train['middle_right_square'])

x_Train['bottom_left_square'] = LEn.fit_transform(x_Train['bottom_left_square'])

x_Train['bottom_middle_square'] = LEn.fit_transform(x_Train['bottom_middle_square'])

x_Train['bottom_right_square'] = LEn.fit_transform(x_Train['bottom_right_square'])
x_test['top_left_square'] = LEn.fit_transform(x_test['top_left_square'])

x_test['top_middle_square'] = LEn.fit_transform(x_test['top_middle_square'])

x_test['top_right_square'] = LEn.fit_transform(x_test['top_right_square'])

x_test['middle_left_square'] = LEn.fit_transform(x_test['middle_left_square'])

x_test['middle_middle_square'] = LEn.fit_transform(x_test['middle_middle_square'])

x_test['middle_right_square'] = LEn.fit_transform(x_test['middle_right_square'])

x_test['bottom_left_square'] = LEn.fit_transform(x_test['bottom_left_square'])

x_test['bottom_middle_square'] = LEn.fit_transform(x_test['bottom_middle_square'])

x_test['bottom_right_square'] = LEn.fit_transform(x_test['bottom_right_square'])
x_Train.head()
x_test.head()
x_test_pred = x_test.iloc[:, 0:9]

x_test_pred.head()
from sklearn.tree import DecisionTreeClassifier



model=DecisionTreeClassifier(criterion='entropy')

model.fit(x_Train, y_Train)
y_pred = model.predict(x_test_pred)
y_pred
id = data_test['Id']
np.random.seed(123)

e = np.random.normal(size=10)  

pred=pd.DataFrame(y_pred, columns=['prediction']) 

print (pred)
pred = np.vstack((id,y_pred)).T

pred
np.savetxt('datathon19.csv', pred, delimiter=',', fmt="%i")
import csv

with open('datathon19.csv',newline='') as f:

    r = csv.reader(f)

    data = [line for line in r]

with open('datathon19.csv','w',newline='') as f:

    w = csv.writer(f)

    w.writerow(['Id','class'])

    w.writerows(data)
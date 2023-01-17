import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv("/kaggle/input/datathon19/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/datathon19/test.csv")

test_data.head()
train_data.shape

test_data.shape
x_train = train_data.iloc[:,0:9]

y_train = train_data['class']

x_test = test_data.iloc[:,0:9]

x_test.head()
x_train.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



x_train['top_left_square'] = le.fit_transform(x_train['top_left_square'])

x_train['top_middle_square'] = le.fit_transform(x_train['top_middle_square'])

x_train['top_right_square'] = le.fit_transform(x_train['top_right_square'])

x_train['middle_left_square'] = le.fit_transform(x_train['middle_left_square'])

x_train['middle_middle_square'] = le.fit_transform(x_train['middle_middle_square'])

x_train['middle_right_square'] = le.fit_transform(x_train['middle_right_square'])

x_train['bottom_left_square'] = le.fit_transform(x_train['bottom_left_square'])

x_train['bottom_middle_square'] = le.fit_transform(x_train['bottom_middle_square'])

x_train['bottom_right_square'] = le.fit_transform(x_train['bottom_right_square'])



x_test['top_left_square'] = le.fit_transform(x_test['top_left_square'])

x_test['top_middle_square'] = le.fit_transform(x_test['top_middle_square'])

x_test['top_right_square'] = le.fit_transform(x_test['top_right_square'])

x_test['middle_left_square'] = le.fit_transform(x_test['middle_left_square'])

x_test['middle_middle_square'] = le.fit_transform(x_test['middle_middle_square'])

x_test['middle_right_square'] = le.fit_transform(x_test['middle_right_square'])

x_test['bottom_left_square'] = le.fit_transform(x_test['bottom_left_square'])

x_test['bottom_middle_square'] = le.fit_transform(x_test['bottom_middle_square'])

x_test['bottom_right_square'] = le.fit_transform(x_test['bottom_right_square'])
x_test.head()
classifier = DecisionTreeClassifier()

classifier = classifier.fit(x_train,y_train)

predict = classifier.predict(x_test)
id = test_data['Id']
np.random.seed(123)

result= np.random.normal(size=10)  

prediction=pd.DataFrame(predict, columns=['prediction']) 

print (prediction)
pred = np.vstack((id,predict)).T

pred
np.savetxt('tic_toc_toe_datathon19.csv', pred, delimiter=',', fmt="%i")
import csv

with open('tic_toc_toe_datathon19.csv',newline='') as file:

    r = csv.reader(file)

    data = [line for line in r]

with open('tic_toc_toe_datathon19.csv','w',newline='') as file:

    w = csv.writer(file)

    w.writerow(['Id','class'])

    w.writerows(data)
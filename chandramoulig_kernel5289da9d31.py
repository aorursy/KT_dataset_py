import numpy as np

import pandas as pd
df_train = pd.read_csv("/kaggle/input/datathon19/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/datathon19/test.csv")

df_test.head()
df_train['top_left_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['top_left_square']]

df_train['top_middle_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['top_middle_square']]

df_train['top_right_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['top_right_square']]

df_train['middle_left_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['middle_left_square']]

df_train['middle_middle_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['middle_middle_square']]

df_train['middle_right_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['middle_right_square']]

df_train['bottom_left_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['bottom_left_square']]

df_train['bottom_middle_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['bottom_middle_square']]

df_train['bottom_right_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_train['bottom_right_square']]
df_test['top_left_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['top_left_square']]

df_test['top_middle_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['top_middle_square']]

df_test['top_right_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['top_right_square']]

df_test['middle_left_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['middle_left_square']]

df_test['middle_middle_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['middle_middle_square']]

df_test['middle_right_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['middle_right_square']]

df_test['bottom_left_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['bottom_left_square']]

df_test['bottom_middle_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['bottom_middle_square']]

df_test['bottom_right_square'] = [0 if i=='x' else 1 if i=='o' else 2 for i in df_test['bottom_right_square']]
x_train = df_train.iloc[:,0:9]

y_train = df_train['class']
x_test = df_test.iloc[:,0:9]

y_test = df_test['Id']
x_train.head()
x_test_predict = x_test.iloc[:,0:9]
from sklearn.tree import DecisionTreeClassifier

dtmodel = DecisionTreeClassifier(criterion='gini')
dtmodel.fit(x_train,y_train)
y_predict = dtmodel.predict(x_test_predict)
y_predict
id = df_test['Id']
np.random.seed(123)

e = np.random.normal(size=10)  

pred = pd.DataFrame(y_predict, columns=['prediction']) 

print (pred)
pred = np.vstack((id,y_predict)).T

pred
np.savetxt('datathon2019.csv', pred, delimiter=',', fmt="%i")
import csv

with open('datathon2019.csv',newline='') as f:

    r = csv.reader(f)

    data = [line for line in r]

with open('datathon2019.csv','w',newline='') as f:

    w = csv.writer(f)

    w.writerow(['Id','class'])

    w.writerows(data)
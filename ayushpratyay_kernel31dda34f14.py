import numpy as np

import pandas as pd
df=pd.read_csv('../input/datathon19/train.csv')

df.head()
df.describe()
df1=pd.read_csv('../input/datathon19/test.csv')

df1.head()
df1.describe()
df.shape
df1.shape
x_train=df.iloc[:,:9]

y_train=df['class']

x_train
x_test=df1.iloc[:,:9]

x_test
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
x_train['top_left_square'] = encoder.fit_transform(x_train['top_left_square'])

x_train['top_middle_square'] = encoder.fit_transform(x_train['top_middle_square'])

x_train['top_right_square'] = encoder.fit_transform(x_train['top_right_square'])

x_train['middle_left_square'] = encoder.fit_transform(x_train['middle_left_square'])

x_train['middle_middle_square'] = encoder.fit_transform(x_train['middle_middle_square'])

x_train['middle_right_square'] = encoder.fit_transform(x_train['middle_right_square'])

x_train['bottom_left_square'] = encoder.fit_transform(x_train['bottom_left_square'])

x_train['bottom_middle_square'] = encoder.fit_transform(x_train['bottom_middle_square'])

x_train['bottom_right_square'] = encoder.fit_transform(x_train['bottom_right_square'])



x_test['top_left_square'] = encoder.fit_transform(x_test['top_left_square'])

x_test['top_middle_square'] = encoder.fit_transform(x_test['top_middle_square'])

x_test['top_right_square'] = encoder.fit_transform(x_test['top_right_square'])

x_test['middle_left_square'] = encoder.fit_transform(x_test['middle_left_square'])

x_test['middle_middle_square'] = encoder.fit_transform(x_test['middle_middle_square'])

x_test['middle_right_square'] = encoder.fit_transform(x_test['middle_right_square'])

x_test['bottom_left_square'] = encoder.fit_transform(x_test['bottom_left_square'])

x_test['bottom_middle_square'] = encoder.fit_transform(x_test['bottom_middle_square'])

x_test['bottom_right_square'] = encoder.fit_transform(x_test['bottom_right_square'])
x_test.head()
x_train.head()
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model=model.fit(x_train,y_train)

y_pred=model.predict(x_test)

y_pred
id=df1['Id']
np.random.seed(123)

pred=pd.DataFrame(y_pred,columns=['predictions'])

pred
pred = np.vstack((id,y_pred)).T

pred
np.savetxt('Ayush-Pratyay-datathon19.csv', pred, delimiter=',', fmt="%i")
import csv

with open('Ayush-Pratyay-datathon19.csv',newline='') as f:

    r = csv.reader(f)

    data = [line for line in r]

with open('Ayush-Pratyay-datathon19.csv','w',newline='') as f:

    w = csv.writer(f)

    w.writerow(['Id','class'])

    w.writerows(data)
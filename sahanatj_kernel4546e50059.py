import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
df_train = pd.read_csv("C:/Users/sahana/Downloads/train.csv")

df_train.head()
df_test = pd.read_csv("C:/Users/sahana/Downloads/test.csv")

df_test.head()
df_train.describe()
df_test.describe()
df_train.shape
df_test.shape
x_train = df_train.iloc[:,0:9]

y_train = df_train['class']

x_test = df_test.iloc[:,0:9]

x_test.head()
x_train.head()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()



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
clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)

y_prediction = clf.predict(x_test)
y_prediction
id = df_test['Id']

id
np.random.seed(123)

e = np.random.normal(size=10)  

prediction=pd.DataFrame(y_prediction, columns=['prediction']) 

print (prediction)
prediction = np.vstack((id,y_prediction)).T

prediction
np.savetxt('datathon19.csv', prediction, delimiter=',', fmt="%i")
import csv

with open('datathon19.csv',newline='') as f:

    r = csv.reader(f)

    data = [line for line in r]

with open('datathon19.csv','w',newline='') as f:

    w = csv.writer(f)

    w.writerow(['Id','class'])

    w.writerows(data)
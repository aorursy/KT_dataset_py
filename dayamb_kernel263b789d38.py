import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv("C:/Users/Dayanand Baligar/train.csv")

train_df.head()
test_df = pd.read_csv("C:/Users/Dayanand Baligar/test.csv")

test_df.head()
train_df.describe()
test_df.describe()
train_df.shape
test_df.shape
x_train = train_df.iloc[:,0:9]

y_train = train_df['class']

x_test = test_df.iloc[:,0:9]

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
classifier = DecisionTreeClassifier()

classifier = classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
y_pred
id = test_df['Id']
np.random.seed(123)

e = np.random.normal(size=10)  

pred=pd.DataFrame(y_pred, columns=['prediction']) 

print (pred)
pred = np.vstack((id,y_pred)).T

pred
np.savetxt('daya_datathon.csv', pred, delimiter=',', fmt="%i")
import csv

with open('daya_datathon.csv',newline='') as f:

    r = csv.reader(f)

    data = [line for line in r]

with open('daya_datathon.csv','w',newline='') as f:

    w = csv.writer(f)

    w.writerow(['Id','class'])

    w.writerows(data)
import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/datathon19/train.csv')

df.head()
x_train = df.iloc[:, 0:9]

y_train = df['class']

x_train.head()
x_test = pd.read_csv('../input/datathon19/test.csv')

x_test.head()
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
x_train.head()
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=20000)

model.fit(x_train,y_train)
x_test_pred = x_test.iloc[:, 0:9]

x_test_pred.head()
y_pred = model.predict(x_test_pred)
y_pred
y_pred.shape
y_test_id = x_test['Id']

y_test_id.shape
combined = np.vstack((y_test_id, y_pred)).T

combined.shape
np.savetxt('datathon19.csv', combined, delimiter=',', fmt="%i")
asdf =  pd.read_csv('datathon19.csv')

asdf.head()
import csv

with open('datathon19.csv',newline='') as f:

    r = csv.reader(f)

    data = [line for line in r]

with open('datathon19.csv','w',newline='') as f:

    w = csv.writer(f)

    w.writerow(['Id','class'])

    w.writerows(data)
asdf =  pd.read_csv('datathon19.csv')

asdf.head()
dfOrigin = pd.read_csv('../input/datathon19/tic_tac_toe_dataset.csv')
x = dfOrigin.iloc[:, 0:9]

y = dfOrigin['class']

x.head()
from sklearn.model_selection import train_test_split



x_train_origin, x_test_origin, y_train_origin, y_test_origin = train_test_split(x, y, random_state = 47, test_size = 0.25)
x_train_origin.head()

y_train_origin.head()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20000)

# model.fit(x_train_origin, y_train_origin)
x_train_origin['top_left_square'] = encoder.fit_transform(x_train_origin['top_left_square'])

x_train_origin['top_middle_square'] = encoder.fit_transform(x_train_origin['top_middle_square'])

x_train_origin['top_right_square'] = encoder.fit_transform(x_train_origin['top_right_square'])

x_train_origin['middle_left_square'] = encoder.fit_transform(x_train_origin['middle_left_square'])

x_train_origin['middle_middle_square'] = encoder.fit_transform(x_train_origin['middle_middle_square'])

x_train_origin['middle_right_square'] = encoder.fit_transform(x_train_origin['middle_right_square'])

x_train_origin['bottom_left_square'] = encoder.fit_transform(x_train_origin['bottom_left_square'])

x_train_origin['bottom_middle_square'] = encoder.fit_transform(x_train_origin['bottom_middle_square'])

x_train_origin['bottom_right_square'] = encoder.fit_transform(x_train_origin['bottom_right_square'])



x_test_origin['top_left_square'] = encoder.fit_transform(x_test_origin['top_left_square'])

x_test_origin['top_middle_square'] = encoder.fit_transform(x_test_origin['top_middle_square'])

x_test_origin['top_right_square'] = encoder.fit_transform(x_test_origin['top_right_square'])

x_test_origin['middle_left_square'] = encoder.fit_transform(x_test_origin['middle_left_square'])

x_test_origin['middle_middle_square'] = encoder.fit_transform(x_test_origin['middle_middle_square'])

x_test_origin['middle_right_square'] = encoder.fit_transform(x_test_origin['middle_right_square'])

x_test_origin['bottom_left_square'] = encoder.fit_transform(x_test_origin['bottom_left_square'])

x_test_origin['bottom_middle_square'] = encoder.fit_transform(x_test_origin['bottom_middle_square'])

x_test_origin['bottom_right_square'] = encoder.fit_transform(x_test_origin['bottom_right_square'])
x_test_origin.head()
x_test_origin.head()
y_train_origin =  encoder.fit_transform(y_train_origin)

y_test_origin =  encoder.fit_transform(y_test_origin)
y_test_origin[0:5]
model.fit(x_train_origin, y_train_origin)
y_pred_origin = model.predict(x_test_origin)
y_pred_origin
from sklearn.metrics import accuracy_score

print('Accuracy Score on train data: ', accuracy_score(y_true=y_train_origin, y_pred=model.predict(x_train_origin)))

print('Accuracy Score on test data: ', accuracy_score(y_true=y_test_origin, y_pred=y_pred_origin))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
from sklearn.metrics import accuracy_score
import csv

df = pd.read_csv('../input/train.csv')
df1 = pd.read_csv('../input/test.csv')
def cabin_num(c):
  for i in range(len(c)):
    if c[i] and type(c[i]) == str and len(str(c[i]).split()) > 0:
      c[i] = str(c[i]).split()[0]
      alp = c[i][0]
      num = c[i][1:]
      alp = (ord(alp.lower())-96)
      c[i] = int(str(alp)+num)
  return c
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
from sklearn.metrics import accuracy_score
import csv

df = pd.read_csv('../input/train.csv')
df1 = pd.read_csv('../input/test.csv')

def cabin_num(c):
  for i in range(len(c)):
    if c[i] and type(c[i]) == str and len(str(c[i]).split()) > 0:
      c[i] = str(c[i]).split()[0]
      alp = c[i][0]
      num = c[i][1:]
      alp = (ord(alp.lower())-96)
      c[i] = int(str(alp)+num)
  return c

raw_data = df.drop(labels=['Name', 'Ticket'], axis=1)
raw_data_1 = df1.drop(labels=['Name', 'Ticket'], axis=1)
sex_mapping = {'male': 0, 'female': 1}
embark_mapping =  {'S': 0, 'C': 1, 'Q': 2}
raw_data.replace({'Sex': sex_mapping, 'Embarked': embark_mapping}, inplace=True)
raw_data_1.replace({'Sex': sex_mapping, 'Embarked': embark_mapping}, inplace=True)
c = np.array(raw_data['Cabin'])
c1 = np.array(raw_data_1['Cabin'])
raw_data['Cabin'] = cabin_num(c)
raw_data_1['Cabin'] = cabin_num(c1)
raw_data.fillna(raw_data.median(), inplace=True)
raw_data_1.fillna(raw_data.median(), inplace=True)
test = raw_data_1
train = raw_data
train_y = train['Survived']
train.drop(labels=['Survived'], axis=1, inplace=True)
x = train
clf = SVC(gamma='auto', kernel='linear')
t0 = time()
clf.fit(x, train_y)
print('Training time:', round(time() - t0, 3), 's')
pred = clf.predict(test)
pred = pred.tolist()
passenger = test['PassengerId']
passenger = np.array(passenger).tolist()

with open('output.csv','w', newline='') as file:
  w = csv.writer(file)
  w.writerow(['PassengerId', 'Survived'])
  w.writerows(zip(passenger, pred))
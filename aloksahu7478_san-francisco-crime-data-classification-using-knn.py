import csv

import os

import numpy as np

from sklearn.neighbors import NearestNeighbors

import pandas as pd

import zipfile

from sklearn.neighbors import NearestNeighbors

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from datetime import datetime

le = preprocessing.LabelEncoder()
#../input/sf-crime/train.csv.zip

z = zipfile.ZipFile('../input/sf-crime/train.csv.zip')

train = pd.read_csv(z.open('train.csv'))

z = zipfile.ZipFile('../input/sf-crime/test.csv.zip')

test = pd.read_csv(z.open('test.csv'))



l = train.Category.unique()

train_cf = ['Category','Resolution','DayOfWeek','PdDistrict','Descript','Address']

test_cf = ['DayOfWeek','PdDistrict','Address']

test_id = test['Id']

cat_train = train['Category']
s1 = list()

for i in train['Dates']:

    dt = datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S')

    s1.append(dt.strftime("%H%M"))

train['Dates']  = pd.DataFrame(s1,columns=['Dates'])



s1 = list()

for i in test['Dates']:

    dt = datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S')

    s1.append(dt.strftime("%H%M"))

test['Dates']  = pd.DataFrame(s1,columns=['Dates'])

    

for i in train_cf:

    train[i] = le.fit_transform(train[i])

    

for i in test_cf:

    test[i] = le.fit_transform(test[i])

train_pred = train['Category']



    

#'2015-05-13 23:53:00'    

#datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
test
train
training = train.drop(['Category','Descript','Resolution'], axis = 1)

testing = test.drop(['Id'], axis = 1)
result = pd.concat([pd.DataFrame(cat_train),pd.DataFrame(train_pred)], axis=1)

result = result.drop_duplicates() 

result.reset_index(drop=True, inplace=True)

#result.columns.name = None

result.columns = ['Id', 'Index']

category = dict(zip(result.Index, result.Id))

result = result.sort_values(by=['Id'])

print(result)
#reg = LinearRegression()

#reg.fit(training,train_pred)

#model = LogisticRegression(solver='lbfgs')

#model.fit(training,train_pred)

#clf = RandomForestClassifier(n_estimators = 1000, random_state = 9)

#clf.fit(training,train_pred).TPUClusterResolver()

model = KNeighborsClassifier(n_neighbors=len(l))

model.fit(training,train_pred)
predicted= model.predict(testing)
p = predicted.tolist()
output = pd.DataFrame(p,columns=['Output'])

output = output['Output'].map(category)

output
output = output.to_dict()

len(output)
df = pd.DataFrame(columns=result['Id'])

zero_mat = np.zeros((len(output), len(result)), dtype=int)

final = pd.DataFrame(data= zero_mat, columns=result['Id'])    

for i in range(0, len(output)):

    c = final.columns.get_loc(output[i])

    final.iloc[i,c] = 1
final.index.name = 'ID'

final.to_csv('./Submission.csv', index = True)
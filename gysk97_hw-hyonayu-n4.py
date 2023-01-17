# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import cv2



dataset_train = "/kaggle/input/2019-fall-pr-project/train/train/"

data = []

label = []

raw_d = []

for d in os.listdir(dataset_train):

  raw_d = cv2.imread(dataset_train + d )

  raw_d = cv2.resize(raw_d, (32,32))

  raw_d = raw_d.flatten()

  data.append(raw_d)

  if d.split('.')[0] == 'cat':

    label.append('cat')

  else:

    label.append('dog')

  if len(data)>=2000:break





trainD, testD, trainL, testL = train_test_split(data, label, test_size = 0.25, random_state = 42)

from sklearn import svm





clf = svm.SVC(gamma = 'scale' , kernel = 'rbf' ,random_state = 42)

clf.fit(trainD, trainL)



y = clf.predict(testD)











dataset_train = "/kaggle/input/2019-fall-pr-project/test1/test1/"





data = []

label = []

raw_d = []

for d in os.listdir(dataset_test):

  raw_d = cv2.imread(dataset_test + d )

  raw_d = cv2.resize(raw_d, (32,32))

  raw_d = raw_d.flatten()

  data.append(raw_d)

  if d.split('.')[0] == 'cat':

    label.append('cat')

  else:

    label.append('dog')

 











y_pred = clf.predict(data)

print(y_pred)













id = []

for i in range(1,5001):

  id.append(i)

y_pred=  list(y_pred)

dic = {'id': id}

dic2 = { 'label': y_pred}

print(dic)

print(id)
dic = {'id' : id, 'label':y_pred}



# numpy 를 Pandas 이용하여 결과 파일로 저장



result = dic

import pandas as pd



#print(result.shape)

df = pd.DataFrame(result)

df = df.replace('dog',1)

df = df.replace('cat',0)

print(df)

df.to_csv('results-yk-v2.csv',index=False, header=True)
pd.read_csv('results-yk-v2.csv')
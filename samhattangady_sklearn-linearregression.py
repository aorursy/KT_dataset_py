# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df =  pd.read_csv('../input/train.csv')
df.values[0::,1::]
labels = df.label
labels
dataset = df.values[0::,1::]
from sklearn.linear_model import LogisticRegression

regr = LogisticRegression()
data = []

for image in dataset:

    data.append([(pix-128)/255 for pix in image])

data = np.array(data)
print(data[0].mean())

print(np.median(data[0]))
train_data = data

train_labels = np.array(labels)
regr.fit(train_data, train_labels)
test = pd.read_csv('../input/test.csv')

test_data1 = test.values[0::,0::]



test_data=[]

for image in test_data1:

    test_data.append([(pix-128)/255 for pix in image])

test_data = np.array(test_data)
ans = regr.predict(test_data)
ans
len(ans)
csv_ans = []

csv_ans.append(['ImageId', 'Label'])
for i, a in enumerate(ans):

    csv_ans.append([i+1, a])
csv_ans[-1]
import csv

with open('ans.csv', 'w') as a:

    writer = csv.writer(a)

    for row in csv_ans:

        writer.writerow(row)
print(check_output(["ls"]).decode("utf8"))
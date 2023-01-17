# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.DataFrame.from_csv('../input/train.csv')
data.info()
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
Survived = data['Survived']

del data['Survived']
def cleanData(data):

    mapping = {'male':0,'female':1}

    data['Sex']=[mapping[x] for x in data['Sex']]

    embarked_mapping = {}

    data.Embarked.fillna(value=data.Embarked.mode())

    for id,x in  enumerate(data.Embarked.unique()):

        embarked_mapping[x]=id

    data.Embarked = [embarked_mapping[x] for x in data.Embarked]

    del data['Name']

    del data['Ticket']

    del data['Cabin']

    for key in data.columns:

        try:

            assert data[key].notnull().all()

        except(AssertionError):

            data[key]=data[key].fillna(value=data[key].mean())
cleanData(data)
clf.fit(X=data,y=Survived)
test = pd.DataFrame.from_csv('../input/test.csv')
cleanData(test)
result = clf.predict(X=test)
from matplotlib import pyplot as plt
aim = [Survived]
roundedResult = [int(round(x)) for x in result]
import csv
roundedResult
f = open('result.csv','w')
writer = csv.writer(f)
for i in roundedResult:

    writer.writerow([i])
f.close()
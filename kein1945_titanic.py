# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
# Find null records

# data_test[data_test.isnull().T.any().T]

def prepare_data(data):

    # drop unnecessary columns

    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # transform categorical variables

    sex = pd.Series([0,1],index=['male','female'])    

    data['sex']=data.Sex.map(sex)

    

    data.Embarked = data.Embarked.fillna('Q')

    embarked=pd.Series([1,2,3],index=data.Embarked.unique())

    data['embarked']=data.Embarked.map(embarked)

    

    data = data.drop(['Sex'], axis=1)

    data = data.drop(['Embarked'], axis=1)

    

    data.Age = data.Age.fillna(data.Age.mean())

    data.Fare = data.Fare.fillna(data.Fare.mean())

    return data
data = prepare_data(data)

data_test = prepare_data(data_test)
clsf = RandomForestClassifier()
clsf.fit(X=data.drop(['Survived', 'PassengerId'], axis=1), y=data.Survived)
predict = clsf.predict(data_test.drop(['PassengerId'], axis=1))
submission = pd.concat([data_test.PassengerId,\

                        pd.Series(predict, name='Survived')], axis=1)

submission.head()
#submission.to_csv("output.csv", header=True, index_label="PassengerId")
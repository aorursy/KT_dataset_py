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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
train_x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

train_x.fillna(0, inplace=True)

train_y = df['Survived']
train_x.head()

sex_to_cat = {'male':0,

              'female':1}

embarked_to_cat = {'S': 0,

                   'C':1,

                   'Q':2,

                   0: 3}



train_x['Sex'] = train_x['Sex'].apply(lambda x: sex_to_cat[x])

train_x['Embarked'] = train_x['Embarked'].apply(lambda x: embarked_to_cat[x])

train_x.head()



from sklearn.svm import SVC

from sklearn.metrics import classification_report



cutoff = int(.8 * len(train_x))

test_x, test_y = train_x[cutoff:], train_y[cutoff:]

train_x, train_y = train_x[:cutoff], train_y[:cutoff]



print(test_x.shape, train_x.shape)



model = SVC()

model.fit(train_x, train_y)

pred = model.predict(test_x)

print(classification_report(test_y, pred))

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report





model = GradientBoostingClassifier()

model.fit(train_x, train_y)

pred = model.predict(test_x)

print(classification_report(test_y, pred))

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
imputer = Imputer()

# Preprocessing

train = pd.read_csv('../input/train.csv')

train_Y = train['Survived']

train_X = train.drop(['Name','Survived','Ticket','Cabin'],axis=1)

onehot_train_X = imputer.fit_transform(pd.get_dummies(train_X))

onehot_train_X, onehot_test_X, train_Y, test_Y = train_test_split(onehot_train_X, train_Y)
#The actual model

tree = RandomForestClassifier(min_samples_split=4,max_depth=5)

tree.fit(onehot_train_X,train_Y)

print("Train score:{}".format(tree.score(onehot_train_X,train_Y)))

print("Test score:{}".format(tree.score(onehot_test_X,test_Y)))
# Compute the final output, and write out

test = pd.read_csv('../input/test.csv')

test_X = test.drop(['Name', 'Ticket','Cabin'],axis=1)

onehot_test_X = imputer.fit_transform(pd.get_dummies(test_X))

output_df = pd.DataFrame(columns=['PassengerId','Survived'])

output_df['PassengerId'] = test_X['PassengerId']

output_df['Survived'] = tree.predict(onehot_test_X)

output_df.to_csv(path_or_buf=open('./output.csv','w'),index=False)
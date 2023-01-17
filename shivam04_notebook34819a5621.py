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
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

for k in ('Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'):

    del data_train[k]

    del data_test[k]

features_test = data_test

labels_test = pd.read_csv('../input/genderclassmodel.csv')

del labels_test['PassengerId']

#print (label_test.head())



    

#print(data_train.head())

import matplotlib.pyplot as plt

plt.plot(data_train['PassengerId'], data_train['Survived'], 'ro')

plt.axis([0, 5,0 ,1])

plt.show()

labels_train = data_train['Survived']

del data_train['Survived']

features_train = data_train

features_train['Sex'] =  np.where(features_train['Sex']=='male',1,0)

features_test['Sex'] = np.where(features_test['Sex']=='male',1,0)



features_train = features_train.replace(['C','Q','S'],[0,1,2])

features_test = features_test.replace(['C','Q','S'],[0,1,2])

print(features_train.head())

mu = features_train['Age'].mean()

features_train['Age'] = features_train['Age'].fillna(mu)

mu = features_train['Fare'].mean()

features_train['Fare'] = features_train['Fare'].fillna(mu)

features_train['Embarked'] = features_train['Embarked'].fillna(0)

mu = features_test['Age'].mean()



features_test['Age'] = features_test['Age'].fillna(mu)

mu = features_test['Fare'].mean()

features_test['Fare'] = features_test['Fare'].fillna(mu)

features_test['Embarked'] = features_test['Embarked'].fillna(0)

#features_train = features_train.dropna()

#features_test = features_test.dropna()

from sklearn import tree

from sklearn.metrics import accuracy_score

acc = list(tuple())

for s in range(2,100):

    clf = tree.DecisionTreeClassifier(min_samples_split=s)

    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)

    #print(len(pred))

    

    acc.append((accuracy_score(pred, labels_test),s))

print(max(acc))
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
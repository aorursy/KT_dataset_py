import numpy as np

import pandas as pd

from sklearn import svm



train_data = pd.read_csv('../input/train.csv', sep=',')

test_data = pd.read_csv('../input/test.csv', sep=',')
# Functions definitions

def printPassengerData(my_data, passengerId):

    adaptPassengerId = passengerId - 1

    

    for data in my_data:

        print(my_data[data][adaptPassengerId])
# Choose passengerId

passengerId = 1



printPassengerData(test_data, passengerId)
"""

X = [[0, 0], [1, 1]]

y = [0, 1]

clf = svm.SVC(random_state=True)

clf.fit(X, y)  



clf.predict([[2., 2.]])

"""



from sklearn import datasets

iris = datasets.load_iris()

digits = datasets.load_digits()



#print(digits.data[800])

#print(digits.target[800])



# Create a classifier

clf = svm.SVC(gamma=0.001, C=100., kernel='linear')



# Teach the algorithm

#clf.fit(digits.data[:-1], digits.target[:-1])



dtrain = []

for num, data in enumerate(train_data.Survived):

    test = []

    test.append(train_data.Pclass[num])

    test.append(train_data.SibSp[num])

    dtrain.append(test)

    

clf.fit(dtrain, train_data.Survived)



# Validation

#clf.predict(digits.data[-1:])



pclass = []

for num, data in enumerate(test_data.Pclass):

    pclass.append(data)



sibsp = []

for num, data in enumerate(test_data.SibSp):

    sibsp.append(data)



test = []

for num, data in enumerate(pclass):

    test.append([sibsp[num], pclass[num]])

    print(clf.predict([[sibsp[num], pclass[num]]]))
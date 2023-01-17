import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split 

from sklearn import datasets 

import numpy as np

train = pd.read_csv("../input/digit-recognizer/train.csv")

print(train.values)
(trainData, valData, trainLabels, valLabels) = train_test_split(np.array(train.values), train.label, test_size=0.1, random_state=42)

print("training data points: {}".format(len(trainLabels))) 

print("validation data points: {}".format(len(valLabels))) 
from sklearn.neighbors  import KNeighborsClassifier

kVals = range(1, 30, 2) 

accuracies = []

for k in range(1, 30, 2):    

    model = KNeighborsClassifier(n_neighbors=k)   

    model.fit(trainData, trainLabels)

    score = model.score(valData, valLabels) 

    print("k=%d, accuracy=%.2f%%" % (k, score * 100)) 

    accuracies.append(score)

i = np.argmax(accuracies) 

print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))


from sklearn.metrics import classification_report 

model = KNeighborsClassifier(n_neighbors=kVals[i])

model.fit(trainData, trainLabels)

predictions = model.predict(valData)

print('EVALUATION ON VAL DATA')

print(classification_report(valLabels, predictions))
from sklearn.decomposition import PCA

train = pd.read_csv("../input/digit-recognizer/train.csv")

train.shape 

test  = pd.read_csv("../input/digit-recognizer/test.csv")

test.shape

train.head()

test.head()

train_x = train.values[:,1:]

train_y = train.iloc[:,0]

test_x = test.values

p = PCA(n_components=0.8)

train_x = p.fit_transform(train_x)

test_x = p.transform(test_x)

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(train_x, train_y)

test_y = neigh.predict(test_x)

pd.DataFrame({"ImageId": range(1,len(test_y)+1), "Label": test_y}).to_csv('output.csv', index=False, header=True)
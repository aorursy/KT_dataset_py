radius = 1/30


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read some arff data

def read_train_data():
    f = open('../input/kiwhs-comp-1-complete/train.arff')
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

# Training Data
train = read_train_data()

x_train = []
y_train = []
for i in train:
    x_train.append([i[0],i[1]])
    if (i[2] == -1):
        y_train.append(-1)
    else:
        y_train.append((1))

print(train[1])
print(train[200])
#Test Data

def read_test_data():
    f = open('../input/kiwhs-comp-1-complete/test.csv')
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            newcont = [content[1],content[2]]
            if len(newcont) == 2:
                data.append(newcont)
        else:
            if l.startswith('Id,X,Y'):
                data_line = True
    return data

test = read_test_data()
from sklearn import tree

clf = tree.DecisionTreeClassifier()
for i in range(100):
    clf = clf.fit(x_train, y_train)


print(clf.predict([[-1.73522, -2.359194]]))
print(clf.predict([[1.603484, 0.529976]]))

print(clf.predict(test))
#red & blue trennen (train)
red = [[i[0],i[1]] for i in train if i[2] == 1]
blue = [[i[0],i[1]] for i in train if i[2] == -1]

def average_point(trainset):
    avrg = np.array([0.0,0.0])
    for i in trainset:
        avrg += i
    avrg /= len(trainset)
    return avrg

red_hotspot = average_point(red)
blue_hotspot = average_point(blue)

def predict(point):
    if np.linalg.norm(point - red_hotspot) < np.linalg.norm(point - blue_hotspot):
        return 1
    else:
        return -1

def isInMiddleRadius(point):
    global radius
    length = np.linalg.norm(red_hotspot - blue_hotspot)
    radiusall = radius * length
    r = np.linalg.norm(point - red_hotspot)
    b = np.linalg.norm(point - blue_hotspot)
    
    if (r < b):
        return (b-r) < radiusall
    else:
        return (r-b) < radiusall
predictions = []
for x,y in test:
    tree = clf.predict([[x,y]])
    hot = predict([x,y])
    #print(tree[0], hot)
    if (tree == hot):
        predictions.append(tree[0])
    else:
        #print(tree[0], hot)
        if (isInMiddleRadius([x,y])):
            predictions.append(tree[0])
            print("tree")
        else:
            predictions.append(hot)

print(predictions)
import csv
counter = 0
#Submission
with open('submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id (String)', 'Category (String)'])
    for i in predictions:
        writer.writerow([counter, i])
        counter += 1
from random import shuffle
shuffle(train)

from sklearn import tree
###Train und Testdaten generieren###
trainlength = len(train)
trainanteil = 3/4
test_train = []
test_test = []

for i in range(0,300):
    test_train.append(train[i])
#print(len(test_train))

for i in range(300, 400):
    test_test.append(train[i])
#print(len(test_test))

x_train = []
y_train = []
for i in test_train:
    x_train.append([i[0],i[1]])
    if (i[2] == -1):
        y_train.append(-1)
    else:
        y_train.append((1))


###Baum erstellen###
clf = tree.DecisionTreeClassifier()
for i in range(100):
    clf = clf.fit(x_train, y_train)

###Hotspot erstellen###
#red & blue trennen (train)
red = [[i[0],i[1]] for i in test_train if i[2] == 1]
blue = [[i[0],i[1]] for i in test_train if i[2] == -1]

def average_point(trainset):
    avrg = np.array([0.0,0.0])
    for i in trainset:
        avrg += i
    avrg /= len(trainset)
    return avrg

red_hotspot = average_point(red)
blue_hotspot = average_point(blue)

def predict(point):
    if np.linalg.norm(point - red_hotspot) < np.linalg.norm(point - blue_hotspot):
        return 1
    else:
        return -1

    
###############
testradius = 1/12

###############

def isInMiddleRadius(point):
    length = np.linalg.norm(red_hotspot - blue_hotspot)
    radiusall = testradius * length
    r = np.linalg.norm(point - red_hotspot)
    b = np.linalg.norm(point - blue_hotspot)
    
    if (r < b):
        return (b-r) < radiusall
    else:
        return (r-b) < radiusall

    
###Testen###
predictionsTest = []

for x,y,lo in test_test:
    tree = clf.predict([[x,y]])
    hot = predict([x,y])
    #print(tree[0], hot)
    if (tree == hot):
        predictionsTest.append(tree[0])
    else:
        #print(tree[0], hot)
        if (isInMiddleRadius([x,y])):
            predictionsTest.append(tree[0])
            #print("jo")
            #print("tree")
        else:
            predictionsTest.append(hot)

correct = 0.
incorrect = 0.
c = 0
for entry in test_test:
    if predictionsTest[c] == entry[2]:
        correct += 1
    else:
        incorrect += 1
    c += 1
print("Fehlerquote: " + str(100 * incorrect / correct) + "%")
import csv
counter = 0
#Submission
with open('submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id (String)', 'Category (String)'])
    for i in predictions:
        writer.writerow([counter, i])
        counter += 1
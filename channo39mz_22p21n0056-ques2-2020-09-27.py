# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
data = pd.read_csv("/kaggle/input/titanic/train.csv", index_col='PassengerId') 
head = data #[:10] #จำนวนแถว
head # แสดงค่า expression
print("len(data) = ", len(data)) #จำนวนแถว
print("data.shape = ", data.shape) #จำนวนคอลั่ม
from sklearn import metrics, tree
head.describe()
#จากตางรางด้านบนAgeหายจึงใช้ค่า Pclass,Fare แทน
tree1 = tree.DecisionTreeClassifier(random_state=1318)
tree1.fit(head[['Pclass','Fare']].values, head['Survived'].values)
tree.plot_tree(tree1)
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt
clf = CategoricalNB()
clf.fit(head[['Pclass','Fare']].values, head['Survived'].values)
metrics.plot_confusion_matrix(clf, head[['Pclass','Fare']].values, head['Survived'].values)
plt.show()
pred = clf.predict(head[['Pclass','Fare']].values)
accuracy = (180+455) / (455+94+162+180)
precision = 180 / (94+180)
recall = 180 / (180+164)
f1 = ((precision * recall) / (precision + recall))*2
print(accuracy, precision, recall, f1)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
clf2 = MLPClassifier(random_state=1, max_iter=300).fit(head[['Pclass','Fare']].values, head['Survived'].values)

metrics.plot_confusion_matrix(clf2, head[['Pclass','Fare']].values, head['Survived'].values)
plt.show()
pred = clf2.predict(head[['Pclass','Fare']].values)
accuracy = (138+470) / (470+79+204+138)
precision = 138 / (79+138)
recall = 138 / (138+204)
f1 = ((precision * recall) / (precision + recall))*2
print(accuracy, precision, recall, f1)
from statistics import mean
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn import feature_selection
accuracyA =[] ; precisionA = []; recallA = []; f1A = []
kf = model_selection.KFold(shuffle=True, random_state=1234)
for train_index, test_index in kf.split(head[['Pclass','Fare']].values, head['Survived'].values):
    x_train2, x_test2 = head[['Pclass','Fare']].values[train_index], head[['Pclass','Fare']].values[test_index]
    y_train2, y_test2 = head['Survived'].values[train_index], head['Survived'].values[test_index]
    clf.fit(x_train2,y_train2)
    cop = clf.predict(x_test2) #ค่าy predict
    
    #metrics.plot_confusion_matrix(clf, x_test2, y_test2)
    #plt.show()
    con = confusion_matrix(y_test2, cop)
    #print(con[0,0])
    accuracy = (con[1,1]+con[0,0]) / (con[0,0]+con[0,1]+con[1,0]+con[1,1])
    precision = con[1,1] / (con[1,1]+con[0,1])
    recall = con[1,1] / (con[1,1]+con[1,0])
    f1 = ((precision * recall) / (precision + recall))*2
    accuracyA.append(accuracy)
    precisionA.append(precision)
    recallA.append(recall)
    f1A.append(f1)
    
    
print('accuracy ->',mean(accuracyA), 'precision ->', mean(precisionA), 'recall ->',mean(recallA), 'F-Measure ->',mean(f1A))
accuracyB =[] ; precisionB = []; recallB = []; f1B = []
kf = model_selection.KFold(shuffle=True, random_state=1234)
for train_index, test_index in kf.split(head[['Pclass','Fare']].values, head['Survived'].values):
    x_train2, x_test2 = head[['Pclass','Fare']].values[train_index], head[['Pclass','Fare']].values[test_index]
    y_train2, y_test2 = head['Survived'].values[train_index], head['Survived'].values[test_index]
    clf2.fit(x_train2,y_train2)
    cop = clf2.predict(x_test2) #ค่าy predict
    
    #metrics.plot_confusion_matrix(clf, x_test2, y_test2)
    #plt.show()
    con = confusion_matrix(y_test2, cop)
    #print(con[0,0])
    accuracy = (con[1,1]+con[0,0]) / (con[0,0]+con[0,1]+con[1,0]+con[1,1])
    precision = con[1,1] / (con[1,1]+con[0,1])
    recall = con[1,1] / (con[1,1]+con[1,0])
    f1 = ((precision * recall) / (precision + recall))*2
    accuracyB.append(accuracy)
    precisionB.append(precision)
    recallB.append(recall)
    f1B.append(f1)
    
    
print('accuracy ->',mean(accuracyB), 'precision ->', mean(precisionB), 'recall ->',mean(recallB), 'F-Measure ->',mean(f1B))
accuracyC =[] ; precisionC = []; recallC = []; f1C = []
kf = model_selection.KFold(shuffle=True, random_state=1234)
for train_index, test_index in kf.split(head[['Pclass','Fare']].values, head['Survived'].values):
    x_train2, x_test2 = head[['Pclass','Fare']].values[train_index], head[['Pclass','Fare']].values[test_index]
    y_train2, y_test2 = head['Survived'].values[train_index], head['Survived'].values[test_index]
    tree1.fit(x_train2,y_train2)
    cop = tree1.predict(x_test2) #ค่าy predict
    
    #metrics.plot_confusion_matrix(clf, x_test2, y_test2)
    #plt.show()
    con = confusion_matrix(y_test2, cop)
    #print(con[0,0])
    accuracy = (con[1,1]+con[0,0]) / (con[0,0]+con[0,1]+con[1,0]+con[1,1])
    precision = con[1,1] / (con[1,1]+con[0,1])
    recall = con[1,1] / (con[1,1]+con[1,0])
    f1 = ((precision * recall) / (precision + recall))*2
    accuracyC.append(accuracy)
    precisionC.append(precision)
    recallC.append(recall)
    f1C.append(f1)
    
    
print('accuracy ->',mean(accuracyC), 'precision ->', mean(precisionC), 'recall ->',mean(recallC), 'F-Measure ->',mean(f1C))
print("Average F-Measure",(mean(f1C)+mean(f1B)+mean(f1A))/3)
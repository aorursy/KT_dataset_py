import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(os.listdir("../input"))

#Als erstes lesen wir die .arff-Datei ein, die unsere Trainingsdaten enthält
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

dataset = read_data("../input/kiwhs-comp-1-complete/train.arff")
#print(dataset) # 3-elem-array in array (400)
# Pandas Dataframes mit Ausgabe der ersten fünf Zeilen durch .head()
data_pd = pd.DataFrame(dataset)
data_pd.columns = ['x', 'y', 'Kategorie']
data_pd.head()

# Aufsplittung der Trainings- und Testdaten durch .values werden Dataframes zu Numpyarrays
features = data_pd[["x", "y"]].values 
labels = data_pd["Kategorie"].values

x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=0, test_size = 0.2)

print(len(x_train))
# Darstellung in Graphen.

#scaler = StandardScaler()
#scaler.fit(x_train)
print(features[:,0])
print(features[:,1])
print(data_pd["Kategorie"])
#x_train_sc = scaler.transform(x_train)
#x_test_sc = scaler.transform(x_test)
colors = {-1:'red',1:'blue'}

plt.scatter(features[:,0],features[:,1],c=data_pd["Kategorie"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()

from sklearn.tree import DecisionTreeClassifier
# Decission Tree 
clf = DecisionTreeClassifier().fit(x_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.4f}'
     .format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.4f}'
     .format(clf.score(x_test, y_test)))
from sklearn.neighbors import KNeighborsClassifier
# kNN
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(knn.score(x_test, y_test)))
from sklearn.linear_model import LogisticRegression
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.4f}'
     .format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.4f}'
     .format(logreg.score(x_test, y_test)))
# SVM
clf = svm.SVC()
clf.fit(x_train, y_train)  
print('Accuracy of SVN classifier on training set: {:.4f}'
     .format(clf.score(x_train, y_train)))
print('Accuracy of SVN classifier on test set: {:.4f}'
     .format(clf.score(x_test, y_test)))
# Vorhersage von KNN
pred_pd = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")

x_pred = pred_pd[["X","Y"]].values
knn.predict(x_pred)

# Vorhersage speichern
prediction = pd.DataFrame()
id = []
for i in range(len(x_pred)):
    id.append(i)
    i = i + 1

# Struktur der Ausgabe
prediction["Id (String)"] = id 
prediction["Category (String)"] = knn.predict(x_pred).astype(int)
print(prediction)
prediction.to_csv("predict.csv", index=False)

import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")
dataset=Iris

dataset
X=dataset[["SepalLengthCm",'SepalWidthCm','PetalLengthCm','PetalWidthCm']]
X
Y=dataset[['Species']]
Y
from sklearn.model_selection import train_test_split

Xbig, X1, Ybig, Y1 = train_test_split(X, Y, test_size = 0.2, random_state = 400)
X1
Y1
from sklearn.model_selection import train_test_split

Xbig1, Xbig2, Ybig1, Ybig2 = train_test_split(Xbig, Ybig, test_size = 0.5, random_state = 400)
#from sklearn.model_selection import train_test_split

X2, X3, Y2, Y3 = train_test_split(Xbig1, Ybig1, test_size = 0.5, random_state = 400)
X4, X5, Y4, Y5 = train_test_split(Xbig2, Ybig2, test_size = 0.5, random_state = 400)
Xtrain1=X2.append(X3).append(X4).append(X5)

Ytrain1=Y2.append(Y3).append(Y4).append(Y5)
Xtrain2=X1.append(X3).append(X4).append(X5)

Ytrain2=Y1.append(Y3).append(Y4).append(Y5)
Xtrain3=X1.append(X2).append(X4).append(X5)

Ytrain3=Y1.append(Y2).append(Y4).append(Y5)
Xtrain4=X1.append(X2).append(X3).append(X5)

Ytrain4=Y1.append(Y2).append(Y3).append(Y5)
Xtrain5=X1.append(X2).append(X3).append(X4)

Ytrain5=Y1.append(Y2).append(Y3).append(Y4)
K=5 #Number of nearest neighbors, {1,2,5,9,12,15,20} were used
# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier1 = KNeighborsClassifier(n_neighbors = K, metric = 'minkowski', p = 2)

classifier1.fit(Xtrain1, Ytrain1)
classifier2 = KNeighborsClassifier(n_neighbors = K, metric = 'minkowski', p = 2)

classifier2.fit(Xtrain2, Ytrain2)
classifier3 = KNeighborsClassifier(n_neighbors = K, metric = 'minkowski', p = 2)

classifier3.fit(Xtrain3, Ytrain3)
classifier4 = KNeighborsClassifier(n_neighbors = K, metric = 'minkowski', p = 2)

classifier4.fit(Xtrain4, Ytrain4)
classifier5 = KNeighborsClassifier(n_neighbors = K, metric = 'minkowski', p = 2)

classifier5.fit(Xtrain5, Ytrain5)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
# Predicting the Test set results

Ypred1 = classifier1.predict(X1)

a1=accuracy_score(Y1, Ypred1)

a1

#f1=f1_score(Y1, Ypred1,average=None)
Ypred2 = classifier2.predict(X2)

a2=accuracy_score(Y2, Ypred2)

a2
Ypred3 = classifier3.predict(X3)

a3=accuracy_score(Y3, Ypred3)

a3
Ypred4 = classifier4.predict(X4)

a4=accuracy_score(Y4, Ypred4)

a4
Ypred5 = classifier5.predict(X5)

a5=accuracy_score(Y5, Ypred5)

a5
aggregated_accuracy=(a1+a2+a3+a4+a5)/5
aggregated_accuracy
#K is number of nearest neighbors

#Accuracy is overall accuracy

histdata=pd.DataFrame({

    'K':[1,2,5,9,12,15,20],

    'Accuracy':[0.953,0.946,0.967,0.966,0.95,0.944,0.94]

})
histdata.plot.bar(x='K',y='Accuracy',rot=0)
#It is seen that K=5 is slightly better than other K values
#Now we will implement DT
K='entropy' #Here K represents type of gain information used
# Fitting K-NN to the Training set

from sklearn import tree

classifier1 = tree.DecisionTreeClassifier(random_state=400,criterion=K)

classifier1.fit(Xtrain1, Ytrain1)
classifier2 = tree.DecisionTreeClassifier(random_state=400,criterion=K)

classifier2.fit(Xtrain2, Ytrain2)
classifier3 = tree.DecisionTreeClassifier(random_state=400,criterion=K)

classifier3.fit(Xtrain3, Ytrain3)
classifier4 = tree.DecisionTreeClassifier(random_state=400,criterion=K)

classifier4.fit(Xtrain4, Ytrain4)
classifier5 = tree.DecisionTreeClassifier(random_state=400,criterion=K)

classifier5.fit(Xtrain5, Ytrain5)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
# Predicting the Test set results

Ypred1 = classifier1.predict(X1)

a1=accuracy_score(Y1, Ypred1)
Ypred2 = classifier2.predict(X2)

a2=accuracy_score(Y2, Ypred2)
Ypred3 = classifier3.predict(X3)

a3=accuracy_score(Y3, Ypred3)
Ypred4 = classifier4.predict(X4)

a4=accuracy_score(Y4, Ypred4)
Ypred5 = classifier5.predict(X5)

a5=accuracy_score(Y5, Ypred5)
overall_accuracy=(a1+a2+a3+a4+a5)/5
overall_accuracy
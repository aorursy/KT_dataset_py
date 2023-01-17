from sklearn.datasets import load_digits

import numpy as np

from matplotlib import style

import matplotlib.pyplot as plt

style.use('fivethirtyeight')

mnist=load_digits()
x=np.array(mnist.images)

y=np.array(mnist.target)

#n=int(input("enter the number btw 1 to 1500>>"))

some_digit=np.array(x[25])

plt.imshow(some_digit,cmap=plt.cm.gray_r,interpolation='nearest')

plt.axis("off")

plt.show

print(f"expected output {y[25]}")
nsamples=len(x)

x=x.reshape((nsamples,-1))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score

x_train,y_train,x_test,y_test=x[:1300],y[:1300],x[1300:],y[1300:]

shuffle_index=np.random.permutation(1300)

x_train,y_train=x_train[shuffle_index],y_train[shuffle_index]
def nva():

    nvb=MultinomialNB()

    nvb.fit(x_train,y_train)

    y_pred=nvb.predict(x_test)

    print(confusion_matrix(y_test,y_pred))

    print(classification_report(y_test,y_pred))

    print(cross_val_score(nvb,x_train,y_train,cv=13).mean())

    print("predicted output >>",end=" ")

    print(nvb.predict(some_digit.reshape((1,-1))))
def kna():

    knn=KNeighborsClassifier(n_neighbors=3)

    knn.fit(x_train,y_train)

    y_pred=knn.predict(x_test)

    print(confusion_matrix(y_test,y_pred))

    print(classification_report(y_test,y_pred))

    print(cross_val_score(knn,x_train,y_train,cv=13).mean())

    print("predicted output >>",end=" ")

    print(knn.predict(some_digit.reshape((1,-1))))
def error():

    error_rate=[]

    for i in range(1,10):

         knn=KNeighborsClassifier(n_neighbors=i)

         knn.fit(x_train,y_train)

         y_pred=knn.predict(x_test)

         error_rate.append(np.mean(y_pred!=y_test))

    plt.plot(range(1,10),

             error_rate,color='blue',

             linestyle='dashed',

             marker='o',

             markersize=10,

             markerfacecolor='red')

    plt.xlabel("k value")

    plt.ylabel(" error rate")

    plt.show()
nva()

kna()
error()
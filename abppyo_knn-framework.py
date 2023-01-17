import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
data_dir = "../input/"



def load_data(data_dir, train_row):

    train = pd.read_csv(data_dir + "train.csv")

    print(train.shape)

    

    X_train = train.values[0:train_row,1:]

    Y_train = train.values[0:train_row,0]

    

    Pred_test = pd.read_csv(data_dir + "test.csv").values

    

    return X_train, Y_train, Pred_test



train_row = 10000

Origin_X_train, Origin_Y_train, Origin_X_test = load_data(data_dir, train_row)



import matplotlib 

import matplotlib.pyplot as plt

row = 3

print(Origin_Y_train[row])



plt.imshow(Origin_X_train[row].reshape((28,28)))

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_vali, Y_train, Y_vali = train_test_split(Origin_X_train, Origin_Y_train, test_size = 0.2, random_state = 0)



print(X_train.shape, X_vali.shape, Y_train.shape, Y_vali.shape)

import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



ans_k = 0



k_range = range(1, 8)

scores = []



for k in k_range:

    print("k =" + str(k) + "begin ")

    start = time.time()

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_vali)

    accuracy = accuracy_score(Y_vali, Y_pred)

    scores.append(accuracy)

    end = time.time()

    print(classification_report(Y_vali,Y_pred))

    print(confusion_matrix(Y_vali,Y_pred))

    print("complete time:" + str(end - start) + "secs.")

    
print(scores)

plt.plot(k_range, scores)

plt.xlabel('Value of K')

plt.ylabel('Testing Accuracy')

plt.show()

k = 3



knn = KNeighborsClassifier(n_neighbors = k)

knn.fit(Origin_X_train, Origin_Y_train)

Y_pred = knn.predict(Origin_X_test[:600])



print(Y_pred[200])

plt.imshow(Origin_X_test[200].reshape((28,28)))

plt.show()

print(len(Y_pred))



pd.DataFrame({"ImageId":list(range(1,len(Y_pred)+1)),"Label":Y_pred}).to_csv('digit.csv',index = False, header = True)
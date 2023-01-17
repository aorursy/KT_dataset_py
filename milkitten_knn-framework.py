import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
data_folder = "../input/"



def load_data(data_dir):

    train = pd.read_csv(data_dir + "train.csv")

    test = pd.read_csv(data_dir + "test.csv")

    X_train = train.values[:,1:]

    Y_train = train.values[:,0]

    X_test = test.values

    print(X_train.shape)

    print(Y_train.shape)

    print(X_test.shape)

    return X_train, Y_train, X_test



Org_X_train, Org_Y_train, Org_X_test = load_data(data_folder)

def print_graph(r):

    print(Org_Y_train[r])

    plt.imshow(Org_X_train[r].reshape((28,28)))

    plt.show()

    return 

row = 0

print_graph(row)
def print_rand_graph(n):

    for y in range(10):

        idxs = np.nonzero([y == i for i in Org_Y_train])

        idxs = np.random.choice(idxs[0], row_num)

        for i , idx in enumerate(idxs):

            plt_idx = i * 10 + y + 1

            plt.subplot(row_num, 10, plt_idx)

            plt.imshow(Org_X_train[idx].reshape((28, 28)))

            plt.axis("off")

    plt.show()      

    return

row_num = 3

print_rand_graph(row_num)
from sklearn.model_selection import train_test_split 



x_train, x_vali, y_train, y_vali = train_test_split(Org_X_train, Org_Y_train, test_size = 0.2, random_state = 0)



print(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



k_range = range(1,10)

best_k, best_score = 0, 0

scores = []

flag_print = False



for i in k_range:

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_vali)

    accuScore = accuracy_score(y_vali, y_pred)

    scores.append(accuScore)

    if flag_print:

        print("classification_report")

        print(classification_report(y_vali, y_pred))

        print("confision_matrix")

        print(confusion_matrix(y_vali, y_pred))

    if best_score < accuScore:

        best_k, best_score = i, accuScore

print(scores)

plt.plot(k_range, scores)

plt.xlabel("K")

plt.ylabel("scores")

plt.show()
knn = KNeighborsClassifier(n_neighbors = best_k)

knn.fit(Org_X_train, Org_Y_train)

pred = knn.predict(Org_X_test)
for i in range(20, 30):

    idx = i-19

    plt.subplot(1, 10, idx)

    plt.imshow(Org_X_test[i].reshape((28,28)))

    plt.title(pred[i])

    plt.axis("off")

plt.show()    
print(len(pred))

print(type(pred))

from pandas import DataFrame

image_id = list(range(1,len(pred)+1))

DataFrame({'ImageId' : image_id, 'PredLabel' : pred}).to_csv('Digit_Recogniser_Result.csv', index = False, header = True)



import os

os.path.exists("Digit_Recogniser_Result.csv")
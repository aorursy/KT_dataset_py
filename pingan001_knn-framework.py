import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
data_dir = "../input/"



def load_data(data_dir, train_row):

    train = pd.read_csv(data_dir + "train.csv")

    print(train.shape)

    

    X_train = train.values[0:train_row, 1:]

    y_train = train.values[0:train_row, 0]

    

    Pred_test = pd.read_csv(data_dir + "test.csv").values

    return X_train, y_train, Pred_test



train_row = 5000

Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)
import matplotlib

import matplotlib.pyplot as plt



row = 3



print(Origin_X_train[row])



plt.imshow(Origin_X_train[row].reshape(28, 28))

plt.show()
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# classes = ["3"]

rows = 4



print(classes)

print(Origin_y_train)



for i, cls in enumerate(classes):

#     print(i)

    idxs = np.nonzero([i == y for y in Origin_y_train])

    idxs = np.random.choice(idxs[0], rows)

#     print(idxs[0])

    for j , idx in enumerate(idxs):

            plt_idx = j * len(classes) + i + 1

            plt.subplot(rows, len(classes), plt_idx)

            plt.imshow(Origin_X_train[idx].reshape((28, 28)))

            plt.axis("off")

            if j == 0:

                plt.title(cls)

    

plt.show()
# import sklearn.model_selection as train_test_split

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(Origin_X_train, Origin_y_train, test_size = 0.2, random_state = 0)



print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



ans_k = 0



k_range = range(1, 8)

scores = []



# 这个地方通过枚举所有的k值来取找到最好的k值预测数据

for k in k_range:

    print("k={}".format(k))

    start = time.time()

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)

    scores.append(accuracy)

    end = time.time()

    

    print(classification_report(y_valid, y_pred))

    print(confusion_matrix(y_valid, y_pred))

    

    print("Completion time: {} seconds".format(end - start))
print(scores)

plt.plot(k_range, scores) 

plt.xlabel("Value of K")

plt.ylabel("Testing Accuracy")

plt.show()
# print(Origin_X_test[0])



k = 3



knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(Origin_X_train, Origin_y_train)

y_pred = knn.predict(Origin_X_test[:300])
row = 100

print (y_pred[row])

plt.imshow(Origin_X_test[row].reshape((28, 28)))

plt.show()
print(len(y_pred))



# save submission to csv

pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    X_train = train.values[:train_row, 1:]
    y_train = train.values[:train_row, 0]
    Pred_test = pd.read_csv(data_dir + "test.csv")
    return X_train, y_train, Pred_test

Origin_X_train, Origin_y_train, Origin_test = load_data("../input/", 5000)
row = 5
print(Origin_y_train[row])
plt.imshow(Origin_X_train[row].reshape((28, 28)))
plt.show()
rows = 4
class_num = 10
for num in range(class_num):
    idxs = np.nonzero([i == num for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i, idx in enumerate(idxs):
        plt_idx = i * class_num + num + 1
        plt.subplot(rows, class_num, plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(num)
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)
print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 10)
scores = []

for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_vali)
    acc = accuracy_score(y_vali, y_pred)
    scores.append(acc)
    end = time.time()
    print(classification_report(y_vali, y_pred))
    print(confusion_matrix(y_vali, y_pred))
    print("Complete time:" + str(start - end) + " Secs.")
print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_test[:300])
print (y_pred[200])
plt.imshow(Origin_test.values[200].reshape((28, 28)))
plt.show()
print(len(y_pred))

pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
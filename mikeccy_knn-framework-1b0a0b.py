import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
x_train = train.values[:,1:]
y_train = train.values[:,0]
x_test = test.values[:]
plt.imshow(x_train[1].reshape((28,28)))
plt.show()
from sklearn.model_selection import train_test_split
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 8)
scores = []

for k in k_range:
    print("k = %d start" % k)
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_t, y_t)
    y_p = knn.predict(x_v)
    accuracy = accuracy_score(y_v, y_p)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_v, y_p))
    print(confusion_matrix(y_v, y_p))
    print("complete time: %d sec" % (end - start))
print(scores)
plt.plot(k_range, scores)
plt.xlabel('value of k')
plt.ylabel('testing accu')
plt.show()
k = k_range[np.argmax(scores)]

knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
print(y_predict[200])
plt.imshow(x_test[200].reshape(28,28))
plt.show()
print(len(y_predict))
pd.DataFrame({"ImageId": list(range(1, len(y_predict) + 1)), "Label": y_predict}).to_csv('Digit_Recogniser_Result.csv', index = False, header = True)
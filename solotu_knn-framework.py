import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'

def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + 'train.csv')
#     print(train.head())
#     print(train.columns.values)
    X_train = train.values[0:train_row, 1:]
    y_train = train.values[0:train_row, 0]
#     print(type(X_train))
#     print(X_train.shape)
#     print(X_train[0])
    Pred_test = pd.read_csv(data_dir + 'test.csv')
    return X_train, y_train, Pred_test

train_row = 5000
X_train, y_train, X_test = load_data(data_dir, train_row)
import matplotlib.pyplot as plt

row = 2
print(y_train[row])
print(type(X_train[row]))
plt.imshow(X_train[row].reshape((28, 28)))
plt.show()
from sklearn.model_selection import train_test_split

X_train_set, X_validation_set, y_train_set, y_validation_set = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
print(X_train_set.shape, X_validation_set.shape, y_train_set.shape, y_validation_set.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0
k_range = range(1,8)
scores = []

for k in k_range:
    print(f'k = {k} start')
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_set, y_train_set)
    y_pred = knn.predict(X_validation_set)
    accuracy = accuracy_score(y_validation_set, y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_validation_set, y_pred))
    print(confusion_matrix(y_validation_set, y_pred))
    print(f'complete time: {end - start} secs')
print(scores)
plt.plot(k_range, scores)
plt.xlabel('value of k')
plt.ylabel('testing accuracy')
plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test[:300])
import matplotlib.pyplot as plt
print(y_pred[200])
print(X_test.shape)
print(type(X_test))
plt.imshow(X_test.values[200].reshape((28, 28)))
plt.show()
print(len(y_pred))
pd.DataFrame({'ImageId': list(range(1, len(y_pred) + 1))}).to_csv('Digit_Recognizer_Result.csv', index=False, header=True)
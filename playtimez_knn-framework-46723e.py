import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
def load_data(path, train_row):
    train_data = pd.read_csv(path + "train.csv")
    print(train_data.shape)
    pred_test = pd.read_csv(path + "test.csv")
    print(pred_test.shape)
    x_train = train_data.values[0:train_row, 1:]
    y_train = train_data.values[0:train_row, 0]
    return x_train, y_train, pred_test.values

path = "../input/"
train_row = 500
Origin_x_train, Origin_y_train, pred_test = load_data(path,train_row)
print(Origin_x_train.shape, Origin_y_train.shape, pred_test.shape)
print(Origin_x_train)
import matplotlib.pyplot as plt
row = 3
print(Origin_y_train[row])
plt.imshow(Origin_x_train[row].reshape((28, 28))) # 28 * 28 = 784
plt.show()
from sklearn.model_selection import train_test_split
x_train, x_vali, y_train, y_vali = train_test_split(Origin_x_train, Origin_y_train, test_size = 0.2, random_state = 0);
# what is random_state? 
print(x_train, x_vali, y_train, y_vali)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0
k_range = range(1,8)
scores = []

for k in k_range:
    print("k= " + str(k) + " begin ")
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_vali)
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    end_time = time.time()
    print(classification_report(y_vali, y_pred))
    print(confusion_matrix(y_vali, y_pred))
    
    print("Completion time: " + str(end_time-start_time) + "Secs.")
print(scores)
plt.plot(k_range, scores)
plt.xlabel("K")
plt.ylabel("Testing Accuracy")
plt.show()
k = 6

knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(Origin_x_train, Origin_y_train)
y_pred = knn.predict(pred_test[:300])

print(y_pred[200])
plt.imshow(pred_test[200].reshape(28,28))
plt.show()
print(len(y_pred))
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+ 1)), "Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index = False, header = True)

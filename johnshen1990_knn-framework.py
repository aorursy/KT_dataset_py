import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
def load_data(data_dir, row_num):
    train = pd.read_csv(data_dir + "train.csv")  
    X_train = train.values[0:row_num, 1:] # 第2列到第785列：image pixels
    y_train = train.values[0:row_num, 0] # 第1列：label
    
    test = pd.read_csv(data_dir + "test.csv")
    X_test = test.values[:row_num] # 取test.csv中的row_num行作为testing set
    return X_train, y_train, X_test

data_dir = "../input/"
row_num = 10000

origin_X_train, origin_y_train, origin_X_test = load_data(data_dir, row_num)
print(origin_X_train.shape, origin_y_train.shape, origin_X_test.shape)
print(origin_y_train[100])
plt.imshow(origin_X_train[100].reshape(28, 28))
from sklearn.model_selection import train_test_split
X_train, X_vali, y_train, y_vali = train_test_split(origin_X_train, origin_y_train, test_size = 0.2, random_state = 0)

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 8)
scores = []
for k in k_range:
    print('k = ', k, 'begin...')
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors = k)
    # magical step自动建立的KNN的模型
    knn.fit(X_train, y_train)
    # 通过knn模型，根据验证集的题目，得到验证集的预测
    y_pred = knn.predict(X_vali)
    # 通过验证集的答案和验证集的预测算出accuracy
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    end = time.time()
    print("Completion time cost:", end - start, "seconds.")
plt.plot(k_range, scores)
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.show()
k = 3
knn = KNeighborsClassifier(n_neighbors = k)
# 使用全部的原始数据作为训练集，此时无需验证集
knn.fit(origin_X_train, origin_y_train)
# 对原始测试集的前300条数据进行预测
y_pred = knn.predict(origin_X_test[:300])
print(y_pred[252])
plt.imshow(origin_X_test[252].reshape(28, 28))
plt.show()
pd.DataFrame({"ImageId": list(range(1, len(y_pred) + 1)), "Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index = False, header = True)
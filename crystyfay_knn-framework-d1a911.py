import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'
train_row = 42000

def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + 'train.csv')
    print(train.shape)
    pixel_train = train.values[0:train_row, 1:]
    label_train = train.values[0:train_row, 0]
    
    test = pd.read_csv(data_dir + 'test.csv').values
    
    return pixel_train, label_train, test
        
Origin_X_train, Origin_Y_train, Origin_X_test = load_data(data_dir, train_row)
    
import matplotlib
import matplotlib.pyplot as plt

row = 5
# print(Origin_X_train[row].reshape(28,28))
plt.imshow(Origin_X_train[row].reshape(28,28))
plt.show()

# 多画几个图
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4

for y, cls in enumerate(classes):
    # y 是 index
    idxs = np.nonzero([i == y for i in Origin_Y_train]) # 把标记为y(0-9)的拿出来, 是一个tuple, tuple[0]是array
    idxs = np.random.choice(idxs[0], rows) # 随机选四个
    # 画图
    for i, idx in enumerate(idxs):
        # 图的位置
        plt_idx = i * len(classes) + y + 1
        # nrows, ncols的矩阵里第几个
        plt.subplot(rows, len(classes), plt_idx) 
        plt.imshow(Origin_X_train[idx].reshape(28,28))
        plt.axis("off")
        if i == 0:
            plt.title('cls')
            
plt.show()

                 
    
from sklearn.model_selection import train_test_split
# random_state = 0, 随机种子每次一样
X_train, X_valid, Y_train, Y_valid = train_test_split(Origin_X_train, Origin_Y_train,
                                                     test_size = 0.2, random_state = 0)
print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#要选的best k
besk_k = 0
k_range = range(1, 8)
scores = []

for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    # train
    knn.fit(X_train, Y_train)
    # predict
    y_pred = knn.predict(X_valid)
    accuracy = accuracy_score(Y_valid, y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(Y_valid, y_pred))
    print(confusion_matrix(Y_valid, y_pred))

    print("Complete time: " + str(end-start) + " Secs.")
plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Value of accuracy')
plt.show()
# 选最好的k
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train, Origin_Y_train)
y_pred = knn.predict(Origin_X_test[:42000])

print(y_pred[260])
plt.imshow(Origin_X_test[260].reshape(28,28))
plt.show()
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
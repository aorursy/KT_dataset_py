import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库

data_dir = "../input/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
#     print(train.head())
#     print(train.shape)
#     print(train.head())
#     print(train[0:])
    X_train = train.values[0:train_row,1:] # 取下标为1-784的列（pixel0 - pixel783）
    y_train = train.values[0:train_row,0] # 取下标为0的列 (label)
    
    
    Pred_test = pd.read_csv(data_dir + "test.csv").values  # 解释 value
#     print(Pred_test.shape)
#     print(pd.read_csv(data_dir + "test.csv").head())
    return X_train, y_train, Pred_test

train_row = 1000 # 如果想取全部数据，设置为最大值 42000
Origin_X_train, Origin_y_train, Origin_y_test = load_data(data_dir, train_row)


print(Origin_X_train.shape, Origin_y_train.shape, Origin_y_test.shape)
print(Origin_X_train)
import matplotlib
import matplotlib.pyplot as plt
row = 3
# 展示第i个图
# print (X_train[row].reshape((28, 28)))

print (Origin_y_train[row])

plt.imshow(Origin_X_train[row].reshape((28, 28)))
plt.show()
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4

print(classes)
for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i , idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
        

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)
# 解释一下random 那个随机函数随机种子是什么。 随机种子一样，那么结果是一样的。

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.shape
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

ans_k = 0
# KNeighborsClassifier predict train

class knn():
    def __init__(self):
        pass
#     fit 就是train的过程
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
#     预测每一个测试集里面的元素对应的label是哪一个， X是一个测试样本
    def predict(self, X, k = 3):
        dataSetSize = self.X_train.shape[0]
        
        #1. 计算x 和所有train data 距离
        testMat = np.tile(X,(dataSetSize, 1))
#          a. 求差
        diffMat = testMat - self.X_train
#          b. 求每一行每一个元素差的平方
        sqDiffMat = diffMat**2
#          c. 求每一行的和
        sumDiffMat = sqDiffMat.sum(axis = 1)
#          d. 开一个根号
        distances = sumDiffMat**0.5
        #2. 把所有的距离排序
        sortedDistances = distances.argsort()
        
        
        classCount = {}
        #3. 找到里x点最近的k个点
        for i in range(k):        
            #4. 输出这k个点里面出现的label数目最多的类别
            id = sortedDistances[i]
            label = self.y_train[id]
            classCount[label] = classCount.get(label, 0 )  + 1
        
        max = 0
        ans = 0
        for k,v in classCount.items():
#             print ("label: " + str(k) + " frequency"+str(v))
            if (v > max):
                ans = k
                max = v
                
        return(ans)
        

k = 10
for k in range(1, 20):
    classifier = knn()        

    classifier.fit(X_train, y_train)

    prediction = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
    #     print (X_test[i])
        pred = classifier.predict(X_test[i], k)
        prediction[i] = pred

    print (accuracy_score(y_test , prediction))


# print (prediction)

np.zeros(3)

from sklearn.metrics import accuracy_score, confusion_matrix
# accuracy_score(y_test , prediction)
confusion_matrix(y_test, prediction)
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0

k_range = range(1,8)
scores = []

# 这个地方通过枚举所有的k值来取找到最好的k值预测数据
for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_test, y_pred))  
    print(confusion_matrix(y_test, y_pred))  
    

    print("Complete time: " + str(end-start) + " Secs.")

x = [8,4,56,3,1]

y = np.tile(x,2)
print (y)

# y.sum(axis = 1)
print((-y).argsort())
print(y.argsort())

print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_y_test)
print (y_pred[200])
plt.imshow(Origin_y_test[200].reshape((28, 28)))
plt.show()
print(len(y_pred))

# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)

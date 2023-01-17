import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
from tqdm import tqdm

data_dir = "../input/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
    X_train = train.values[0:train_row,1:] # 取下标为1-784的列（pixel0 - pixel783）
    y_train = train.values[0:train_row,0] # 取下标为0的列 (label)
    
    
    Pred_test = pd.read_csv(data_dir + "test.csv").values
    return X_train, y_train, Pred_test

train_row = 5000 # 如果想取全部数据，设置为最大值 42000
Origin_X_train, Origin_y_train, Origin_y_test = load_data(data_dir, train_row)


print(Origin_X_train.shape, Origin_y_train.shape, Origin_y_test.shape)
import matplotlib
import matplotlib.pyplot as plt
row = 6
# 展示第i个图
# print (X_train[row].reshape((28, 28)))

print (Origin_y_train[row])
# imshow 把 矩阵转换成图像
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

X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)
# 解释一下random 那个随机函数随机种子是什么。 随机种子一样，那么结果是一样的。

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
class knn():
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    # if no k is passed, default k  is 3
    def predict(self, X, num, k=3):
        dataSet = X_train
        labels = y_train
        
        # dataSetSize: train data rows
        dataSetSize = dataSet.shape[0]
        
        # np.tile: 重复数组若干次
        # a = np.array([0, 1, 2])
        #
        # np.tile(a, (2, 1))
        # array([[0, 1, 2],
        #       [0, 1, 2]])
        #
        # np.tile(a, (2, 2))
        # array([[0, 1, 2, 0, 1, 2],
        #        [0, 1, 2, 0, 1, 2]])

        # 将带预测的一行数据，重复n词， n 就是train data set 行数
        diffMat = np.tile(X,(dataSetSize,1)) - dataSet
        # existing dataset
        #  [1 2 5 6]
        #  [5 4 2 6]
        #  X [0 1 2]
        #  np.tile(x, (2, 1))
        #  [0 1 2]
        #  [0 1 2]
        sqDiffMat = diffMat**2
        sumDiffMat = sqDiffMat.sum(axis=1)
        # 开根号
        distances = sumDiffMat**0.5
        # distances is numpy.ndarray type
        # 带预测的数据X， 和train dataset里每一行都会产生一个distance
        # 由于argsort()的原因，所以sortedDistances 存储的排序后的train set 中 行 对应的index
        sortedDistances = distances.argsort()
        
        # np.argsort: return idx of elements after sorting in ascending order
        # x = np.array([3, 1, 2])
        # np.argsort(x)
        # array([1, 2, 0])
        
        # dict： {label: count}
        classCount = {}
        # 由于是k 近邻，所以找出最近的k个的label究竟是什么，然后label count最大的那个就是预测结果
        for i in range(k):
            vote = labels[sortedDistances[i]]
            classCount[vote] = classCount.get(vote,0) + 1
        # {chicken: 3 duck:4, goose:3}
        max = 0
        ans = 0
        for k,v in classCount.items():
            if(v>max):
                ans = k
                max = v
#         print("test #"+ str(num+1) + " prediction is " + str(ans)
        return(ans)
from sklearn.metrics import accuracy_score

classifier = knn()
classifier.train(X_train, y_train)

max = 0
ans_k = 0

for k in range(1, 4):
    print ('when k = ' + str(k) + ', start training')
    # 每个k都输出 predictions list, 由于是train phase, 该输出list 会与validate set label 比较计算accuracy
    predictions = np.zeros(len(y_vali))
    for i in range(X_vali.shape[0]):
        if i % 500 == 0:
            print("Computing  " + str(i+1) + "/" + str(int(len(X_vali))) + "...")
        output = classifier.predict(X_vali[i], i, k)
        predictions[i] = output
    
#     print(k, predictions)
#     predictions.shape
    accuracy = accuracy_score(y_vali, predictions)
    print ('k = '+ str(k) , ' accuracy =' + str(accuracy))
    if max < accuracy:
        ans_k = k
        max = accuracy
from sklearn.metrics import accuracy_score

classifier = knn()
classifier.train(X_train, y_train)

max = 0
ans_k = 0

for k in range(1, 4):
    print ('when k = ' + str(k) + ', start training')
    # 每个k都输出 predictions list, 由于是train phase, 该输出list 会与validate set label 比较计算accuracy
    predictions = np.zeros(len(y_vali))
    for i in tqdm(range(X_vali.shape[0])):
        output = classifier.predict(X_vali[i], i, k)
        predictions[i] = output
    
    accuracy = accuracy_score(y_vali, predictions)
    print ('k = ',k , ' accuracy =', accuracy);
    if max < accuracy:
        ans_k = k
        max = accuracy
print(y_vali)
print(predictions)
# 通过上面的train phase, 我们得到 k =3 是最优的结果，接下来用k=3来predict
k = 3
Origin_y_test = Origin_y_test[:300] # remove this line for full test
predictions = np.zeros(Origin_y_test.shape[0])
for i in range(Origin_y_test.shape[0]):
    if i % 100 ==0:
        print("Computing  " + str(i+1) + "/" + str(int(len(Origin_y_test))) + "...")
    predictions[i] = classifier.predict(Origin_y_test[i], i, k)

# 验证一下结果
print (predictions[105])
plt.imshow(Origin_y_test[105].reshape((28, 28)))
print(len(predictions))
out_file = open("predictions.csv", "w")
out_file.write("ImageId,Label\n")
for i in range(len(predictions)):
    out_file.write(str(i+1) + "," + str(int(predictions[i])) + "\n")
out_file.close()

for i in tqdm (range(10000)):
    for j in range(10000):
        sum = 0
        sum =1
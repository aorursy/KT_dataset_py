import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库

data_dir = "../input/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    # pandas read_csv return DataFrame
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
#     print(train.head())
#     print(train.shape)
#     print(train.head())
#     print(train[0:])
    # 每个图片样本是 28 * 28 -> 1 * 784
    X_train = train.values[0:train_row,1:] # 取下标为1-784的列（pixel0 - pixel783）
    y_train = train.values[0:train_row,0] # 取下标为0的列 (label)
    
    # testing data
    Pred_test_dataframe = pd.read_csv(data_dir + "test.csv")
    print(Pred_test_dataframe.shape)
    print(Pred_test_dataframe.head())
    
    # dataframe.values  return numpy.ndarray: The values of the DataFrame.
    Pred_test = Pred_test_dataframe.values  # 解释 value
    print(Pred_test.shape)
    
    return X_train, y_train, Pred_test

train_row = 5000 # 如果想取全部数据，设置为最大值 42000
Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)


print(Origin_X_test[:2])
print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_test.shape)
print(Origin_X_train)
print(Origin_y_train)
import matplotlib
import matplotlib.pyplot as plt
row = 4
# 展示第training dataset里面第 3 行的数字
print(Origin_X_train[row].shape)
# resize to 28 to 28
# print (Origin_X_train[row].reshape((28, 28)))

print (Origin_y_train[row])
# imshow 把矩阵画出图像
plt.imshow(Origin_X_train[row].reshape((28, 28)))
plt.show()
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4

print(classes)
# cls 表示classess list里面每一个元素
# y 表示 classess 元素对应的index, 由于list里面实际的item就是数字， 也可以就用index 来表示其中的item
for y, cls in enumerate(classes):
    # Origin_y_train 就是training data里面 每一个样本对应的label, 也就是实际的数字， 0， 1， 2 ...9
    
    # 这里相当于inner loop, e.g. 第一次循环就是针对0，  loop Origin_y_train lable, 看看那些行是表示0的
    # [i == y for i in Origin_y_train] return  boolean array [false,true.....]...然后np.nonzero 找出那些为true的index
    idxs = np.nonzero([i == y for i in Origin_y_train])
    # 由于idxs is tuple type. e.g. 对于第一次大的循环，idxs= (array([   1,    4,    5,..... 4993]),) 表示1， 4， 5  行的lable 表示0
    # 由于是tuple, 所以要用idxs[0] extract出array
    # 从上面这么多行中，随机选出4 个0的label
    idxs = np.random.choice(idxs[0], rows)
    # 经过上面的np.random.choice, 这里idxs变成numpy.ndarray type 了
    # 下面这个二维表，实际是一列，一列的打印的
    for i , idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        #print(plt_idx)
        # define 二维图的 行，列
        # 针对 subplot行数，plt_idx 表示的是第几个图，所以上面 要加1，e.g: plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        # 如果没有下面的两行， 第一行的colunn name 打印不出来
        if i == 0:
            plt.title(cls)
        

plt.show()
from sklearn.model_selection import train_test_split
# test_size = 0.2 validation size: 1/5, train: 4/5
# random_state = 0 表示每次split 一样的划分，i.e.  第一行一直是train, 第二行一直是validation set
X_train,X_vali, y_train, y_vali = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)
# 解释一下random 那个随机函数随机种子是什么。 随机种子一样，那么结果是一样的。

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
# random_state = 0， 使得你每次运行下面的np.sum结果一样，如果没有random_state=0, 则每次都ubuyiyang
np.sum(X_train)
# 可以看一下有多少个9
np.sum(Origin_y_train == 9)
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0

k_range = range(1, 8)
scores = []

# 这个地方通过枚举所有的k值来取找到最好的k值预测数据
for k in k_range:
    print("k = " + str(k) + " begin ")
    # track time
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_vali)
    accuracy = accuracy_score(y_vali,y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred))  
    print(confusion_matrix(y_vali, y_pred))  
    
    print("Complete time: " + str(end-start) + " Secs.")

scores
print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_X_test[:300])
# change this line to y_pred = knn.predict(Origin_X_test) for full test
print (y_pred[200])
plt.imshow(Origin_X_test[200].reshape((28, 28)))
plt.show()
print(len(y_pred))
result = pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred})

print(result.head())

# save submission to csv
result.to_csv('Digit_Recogniser_Result.csv', index=False,header=True)


import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
train = pd.read_csv("../input/train.csv")

x_train = train.values[0:5000,1:]

y_train = train.values[0:5000,0]



x_test = pd.read_csv("../input/test.csv").values
x_train.shape, y_train.shape, test.shape, x_train

import matplotlib

import matplotlib.pyplot as plt



row = 9

print (y_train[row])



plt.imshow(x_train[row].reshape((28, 28)))

plt.show()
from sklearn.model_selection import train_test_split



x_train,x_vali, y_train, y_vali = train_test_split(x_train,

                                                   y_train,

                                                   test_size = 0.2,

                                                   random_state = 0)

# 解释一下random 那个随机函数随机种子是什么。 随机种子一样，那么结果是一样的。



(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)
import time

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



ans_k = 0



k_range = range(1, 8)

scores = []



# 这个地方通过枚举所有的k值来取找到最好的k值预测数据

for k in k_range:

    print("k = " + str(k) + " begin ")

    start = time.time()

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    y_pred = knn.predict(x_vali)

    end = time.time()

    scores.append(accuracy_score(y_vali,y_pred))

    print(classification_report(y_vali, y_pred))  

    print(confusion_matrix(y_vali, y_pred))  

    

    print("Complete time: " + str(end-start) + " Secs.")
print (scores)

plt.plot(k_range,scores)

plt.xlabel('Value of K')

plt.ylabel('Testing accuracy')

plt.show()
k = 3



knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test[:300])

# change this line to y_pred = knn.predict(Origin_X_test) for full test
row = 230

print (y_pred[row])

plt.imshow(x_test[row].reshape((28, 28)))

plt.show()
print(len(y_pred))



# save submission to csv

pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
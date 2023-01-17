import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
dir = "../input/"



train = pd.read_csv("../input/train.csv")

print (train)
print (train.values)



row = 3000  # 为了上课处理方便。 只截取前面3000行

y_train = train.values[0:row,0]

X_train = train.values[0:row,1:]
# print(X_train)

X_train.shape

print (y_train)
test = pd.read_csv(dir + "test.csv")
X_test = test.values

print  (test)
row = 10



print (X_train[row].shape)



plt.imshow(X_train[row].reshape((28,28)))

plt.show()



print (y_train[row])

print (X_test[row].shape)



plt.imshow(X_test[row].reshape((28,28)))

plt.show()
from sklearn.model_selection import train_test_split



X_train_split, X_test_split, y_train_split, y_test_split =  train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

# feature，label，



print (X_train_split.shape, X_test_split.shape)



#训练20% ， 测试 80
print (X_train_split.shape, X_test_split.shape, y_train_split.shape, y_test_split.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



k_range = range(1,20)

    

k = 3 # 设置K

scores = []

for k in k_range:

    print (k)

    knn = KNeighborsClassifier(n_neighbors = k) # 选择模型

    knn.fit(X_train_split, y_train_split) # feature + label  训练X_train_split 就是训练feature， y_train_split就是训练的label

    y_pred = knn.predict(X_test_split) # 预测

    accuracy = accuracy_score(y_test_split, y_pred)

    

    print (accuracy)

    scores.append(accuracy)

    print (confusion_matrix(y_test_split, y_pred))

    print (classification_report(y_test_split, y_pred))



# print (y_pred)



# average-precision



plt.imshow(X_test_split[3].reshape((28,28)))

plt.show()

print(scores)



import matplotlib.pyplot as plt



plt.plot(k_range, scores)

plt.xlabel('k values')

plt.ylabel('Testing accuracy')

plt.show()
k = 3

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors= k)

knn.fit(X_train, y_train)



y_prediction = knn.predict(test)
print (test)
print (y_prediction[201])



# print (test.values[201])

plt.imshow(test.values[201].reshape((28,28)) )

plt.show()
import pandas as pd # 读入csv常用库





print (y_prediction)

output = pd.DataFrame({"Imageid" : list(range(1, len(y_prediction) + 1)), "Label": y_prediction})

print (output) 



output.to_csv("output.csv", index=False, header = True)
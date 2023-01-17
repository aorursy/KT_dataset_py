import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'
def load_data(data_dir):
    train_data = pd.read_csv(data_dir + 'train.csv')
    print(train_data.shape)
    origin_train_x = train_data.values[:,1:]
    origin_train_y = train_data.values[:,0]
    print(origin_train_x.shape, origin_train_y.shape)
    
    test_data = pd.read_csv(data_dir + 'test.csv')
    test_x = test_data.values[:]
    print(test_x.shape)
    
    return origin_train_x, origin_train_y, test_x
    
Origin_train_x, Origin_train_y, Test_x = load_data(data_dir) 
print(Origin_train_x.shape, Origin_train_y.shape, Test_x.shape)
print(Origin_train_y[208])
plt.imshow(Origin_train_x[208].reshape(28,28))
plt.show()
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(Origin_train_x, Origin_train_y, test_size = 0.2, random_state = 0)
print(Origin_train_x.shape, Origin_train_y.shape)
print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
scores = []
for k in range(1,7):
    print('k= ' + str(k) + ' begin')
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_x, train_y) 
    pred_y = neigh.predict(val_x)
    accuracy = accuracy_score(val_y, pred_y)
    scores.append(accuracy)
    print(accuracy)
    print('k= ' + str(k) + ' end')
    print(' ')
    
print(scores)
krange = range(1,7)
plt.plot(krange, scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
k = 0
mostAccuracy = scores[0]
for idx in range(0,6):
    if scores[idx] > mostAccuracy:
        k = idx

k = k + 1
print(k)
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(Origin_train_x, Origin_train_y) 
pred_test_y = neigh.predict(Test_x)
print(pred_test_y.shape)
test_val = 65
print(pred_test_y[test_val])
plt.imshow(Test_x[test_val].reshape(28,28))
plt.show()
res = pd.DataFrame({'ImageId':list(range(1, len(pred_test_y) + 1)), 'Label': pred_test_y})
res.to_csv('Digit_Recogniser_Result.csv', index = False, header = True)
import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'
# data_dir = './'
train_row = 5000  # did not run all data due to time limit

def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + 'train.csv')
    test = pd.read_csv(data_dir + 'test.csv')
    X_train = train.values[: train_row, 1:]
    y_train = train.values[: train_row, 0]
    X_test = test.values
    return X_train, y_train, X_test

Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)
print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_test.shape)

row = 400
print(Origin_y_train[row])
plt.imshow(Origin_X_train[row].reshape((28,28)))
plt.show()
# I need to revisit this part

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
X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train, Origin_y_train, test_size = 0.2, random_state = 1)
print(X_train.shape, y_train.shape, X_vali.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 8)
scores = []

for k in k_range:
    print('k = ', k, 'begin:')
    start = time.time()
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_vali)
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred))
    print(confusion_matrix(y_vali, y_pred))
    print('completion time = ', end-start, 'seconds')
    
plt.plot(k_range, scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
k = 3

knn = KNeighborsClassifier(k)
knn.fit(X_train, y_train)
y_pred = knn.predict(Origin_X_test[:300])


i = 100
print(y_pred[i])
plt.imshow(Origin_X_test[i].reshape((28,28)))
plt.show()
print(len(y_pred))

#pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred}).to_csv('Digit_Recognizer_result.csv')
import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = "../input/"
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    test = pd.read_csv(data_dir + "test.csv")
    #data
    X_train = train.values[0:train_row, 1:]
    #labels
    y_train = train.values[0:train_row, 0]
    
    return X_train, y_train, test.values

train_row = 5000
org_X_train, org_y_train, org_X_test = load_data(data_dir, train_row)
print(org_X_train.shape, org_y_train.shape, org_X_test.shape)
import matplotlib.pyplot as plt

row = 6
print(org_y_train[row])
plt.imshow(org_X_train[row].reshape(28,28))
plt.show()


classes = list(map(str, range(0,10)))
print(classes)

rows = 4

for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in org_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(org_X_train[idx].reshape(28,28))
        plt.axis("off")
        if i == 0:
            plt.title(cls)

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_vali, y_train, y_vali = train_test_split(org_X_train, org_y_train, test_size = 0.2, random_state = 0)
print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN

ans_k = 0
k_range = range(1,8)
scores = []

for k in k_range:
    print("k = {} begin...".format(str(k)))
    start = time.time()
    knn = KNN(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_vali)
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred))
    print("Complete time: {} secs.".format(end - start))
print(scores)
plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()
k = 3
knn = KNN(n_neighbors = k)
knn.fit(org_X_train, org_y_train)
y_pred = knn.predict(org_X_test[:300])
print(y_pred[200])
plt.imshow(org_X_test[200].reshape(28,28))
plt.show()
print(len(y_pred))
output = pd.DataFrame(pd.Series(y_pred), columns=['Label']) 
output.index = output.index + 1
output.insert(0, "ImageId", output.index)
output.to_csv("Digit_Recogniser_Result.csv", index=False, header=True)

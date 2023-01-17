import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
data_dir = "../input/"



def load_data(data_dir, train_row):

    train = pd.read_csv(data_dir + "train.csv")

    X_train = train.values[:train_row, 1:]  

    y_train = train.values[:train_row, 0]

    

    Pred_test = pd.read_csv(data_dir + "test.csv").values

    

    return X_train, y_train, Pred_test  # All are numpy arrays



train_row = 6000

Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)
print(Origin_X_train.shape, Origin_y_train.shape, Origin_X_train.shape)

print(Origin_X_train[:5])
row = 3

print(Origin_y_train[row])

plt.imshow(Origin_X_train[row].reshape(28, 28))

plt.show()
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

rows = 4



for y, cls in enumerate(classes):

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

rows = 4



for y, cls in enumerate(classes):

    # Get the indices where Origin_y_train equals to specific digit

    idxs = np.nonzero([i == y for i in Origin_y_train]) # Return a tuple of the indice where the value != 0 in each dimension

    # Randomly choose rows elements from the indices above

    idxs = np.random.choice(idxs[0], rows) # Use idxs[0], rather than idxs, because idxs is a tuple

    for i, idx in enumerate(idxs):

        plt_idx = i * len(classes) + y + 1

        plt.subplot(rows, len(classes), plt_idx)

        plt.imshow(Origin_X_train[idx].reshape(28, 28))

        plt.axis("off")

        if i == 0:

            plt.title(cls)

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train, Origin_y_train, test_size = 0.2, random_state = 0)

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



k_range = range(1, 8)

scores = []



for k in k_range:

    

    print("k = " + str(k) + " begin")

    start = time.time()

    

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_vali)

    

    accuracy = accuracy_score(y_vali, y_pred)

    scores.append(accuracy)

    end = time.time()

    

    print(classification_report(y_vali, y_pred))

    print(confusion_matrix(y_vali, y_pred))

    print("Complete time: " + str(end - start) + " Seconds.")
plt.plot(k_range, scores)

plt.xlabel('Value of K')

plt.ylabel('Testing accuracy')

plt.show()
k = 6

knn = KNeighborsClassifier(n_neighbors = 6)

knn.fit(Origin_X_train, Origin_y_train)

y_pred = knn.predict(Origin_X_test[:500])

print(type(y_pred))
row = 300

print(y_pred[row])

plt.imshow(Origin_X_test[row].reshape(28, 28))

plt.show()
result = pd.DataFrame({"ImageId": list(range(1, len(y_pred) + 1)), "Label": y_pred})

result.to_csv("Digit_Recogniser_Result.csv", index = False, header = True)
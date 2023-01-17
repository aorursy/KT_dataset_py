import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = "../input/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
    X_train = train.values[0:train_row, 1:] # pixels
    y_train = train.values[0:train_row, 0] # labels
    Pred_test = pd.read_csv(data_dir + "test.csv").values # data as a matrix, i.e. list of lists; data contains pixels withotu labels
    
    return X_train, y_train, Pred_test

train_row = 5000
Origin_X_train, Origin_y_train, Origin_X_test = load_data(data_dir, train_row)
row = 3
print(Origin_y_train[row]) # 4
# print("-----")
# print(Origin_X_train[row])# pixels of 4
# print("-----")
# print(Origin_X_train[row].reshape(28, 28))# pixels of 4 in 28*28 matrix
plt.imshow(Origin_X_train[row].reshape(28, 28)) # returns an image object, imshow() generates an image without showing it
plt.show() # print the image
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4
print(Origin_y_train)

for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in Origin_y_train]) # record index of each class in Origin_Y_train
    idxs = np.random.choice(idxs[0], rows) # idxs is an array with a list of index
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1 # plt_idx starts at 1, increments across rows first and has a maximum of nrows * ncols.
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape(28, 28))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train,
                                                    Origin_y_train,
                                                    test_size = 0.2,
                                                    random_state = 0) # random_state is the seed used by the random number generator (default None)
print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0
k_range = range(1, 8)
scores = []
for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train) # training
    y_pred = knn.predict(X_vali) # test
    accuracy = accuracy_score(y_vali, y_pred) # accuracy
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred)) # precision, recall, f1-score, support; support means # of each class
    print(confusion_matrix(y_vali, y_pred))
    print("Complete time: " + str(end - start) + "Secs.")
print(scores)
plt.plot(k_range, scores)
plt.xlabel("Value of K")
plt.ylabel("Testing accuracy")
plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train, Origin_y_train)
y_pred = knn.predict(Origin_X_test[:300]) # change to Origin_X_test for full test
print(y_pred[200])
plt.imshow(Origin_X_test[200].reshape(28, 28))
plt.show()
print(len(y_pred))

# save submission to csv
pd.DataFrame({"ImageId": list(range(1, len(y_pred) + 1)),
            "Label": y_pred}).to_csv("Digit_Recogniser_Result.csv", index=False, header=True)

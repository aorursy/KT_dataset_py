import numpy as np # process ndarray
import pandas as pd # csv
import matplotlib.pyplot as plt # visulization

data_dir = "../input/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
#     print(train.head())
#     print(train.shape)
#     print(train.head())
#     print(train[0:])
    X_train = train.values[0:train_row,1:] # pixel0 - pixel783
    y_train = train.values[0:train_row,0] # label
    
    
    Pred_test = pd.read_csv(data_dir + "test.csv").values
#     print(Pred_test.shape)
#     print(pd.read_csv(data_dir + "test.csv").head())
    return X_train, y_train, Pred_test

train_row = 5000
Origin_X_train, Origin_y_train, Origin_y_test = load_data(data_dir, train_row)


print(Origin_X_train.shape, Origin_y_train.shape, Origin_y_test.shape)
print(Origin_X_train)
import matplotlib
import matplotlib.pyplot as plt
row = 3


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


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.shape
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

ans_k = 0
# KNeighborsClassifier predict train

class knn():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=3):
        rows = len(self.X_train)
        # 1. Calculate distances between X and all training data
        testX = np.tile(X, (rows, 1)) # tile testing data to match the shape of training data, so that we can use matrix to simplify the calculation
            # a. difference
        testDiff = testX - self.X_train
            # b. square
        sqDiff = np.square(testDiff)
            # c. sum
        sumDiff = np.sum(sqDiff, axis=1)
            # d. sqrt
        distances = np.sqrt(sumDiff)
        # 2. Sort index by the distance (ascending)
        sortedDistance = np.argsort(distances)
        
        class_cnt = {}
        for i in range(k):
            idx = sortedDistance[k]
            label = self.y_train[idx]
            class_cnt[label] = class_cnt.get(label, 0) + 1
        
        max_number = 0
        ans = 0
        for label, cnt in class_cnt.items():
            if cnt > max_number:
                max_number = cnt
                ans = label
        return (ans)

scores = []
for k in range(1, 10):
    print('k = ' + str(k) + ':')
    start = time.time()
    classifier = knn()        

    classifier.fit(X_train, y_train)

    prediction = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
    #     print (X_test[i])
        pred = classifier.predict(X_test[i], k)
        prediction[i] = pred
    score = accuracy_score(y_test, prediction)
    print ('Accuracy: ', score)
    scores.append(score)
    end = time.time()
    print('Time consuming: ' + str(end - start) + ' s.')
print(scores)
plt.plot(range(1,10), scores)
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.title('Relationship between K and Accuracy')
plt.show()

k = 1

classifier = knn()
classifier.fit(Origin_X_train, Origin_y_train)

prediction = np.zeros(Origin_y_test.shape[0])
for i in range(Origin_y_test.shape[0]):
#     if i % 100 == 0:
#         print(i)
    y_pred = classifier.predict(Origin_y_test[i], k)
    prediction[i] = y_pred
print (prediction[200])
plt.imshow(Origin_y_test[200].reshape((28, 28)))
plt.show()
trows = 8
print(classes)
for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in prediction])
    idxs = np.random.choice(idxs[0], trows)
    for i , idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(trows, len(classes), plt_idx)
        plt.imshow(Origin_y_test[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
        

plt.show()
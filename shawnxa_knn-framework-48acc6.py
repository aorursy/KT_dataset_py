import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
def load_data(data_dir, nrows):
    train = pd.read_csv(data_dir + 'train.csv') #returns a pd dataframe
    test = pd.read_csv(data_dir + 'test.csv').values
    X_train = train.values[:nrows, 1:]
    y_train = train.values[:nrows, 0]
    return X_train, y_train, test

Og_X_train, Og_y_train, Og_X_test = load_data('../input/', 5000)
print(Og_X_train.shape, Og_y_train.shape, Og_X_test.shape)
    
# plot a 7 by 10 subplot of random images in training data
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
rows = 7
for di, d in enumerate(digits):
    # grab all the indices in x train that match di
    matched_indices = np.nonzero([di == label for label in Og_y_train])[0] # np.nonzero returns a tuple, choose the first element
    # select 7 random ones from the above indices
    matched_indices = np.random.choice(matched_indices, rows)
    for i, v in enumerate(matched_indices):
        plt_index = i * len(digits) + di + 1
        plt.subplot(rows, len(digits), plt_index)
        plt.imshow(Og_X_train[v].reshape(28, 28))
        plt.axis('off')
        plt.title(d)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_vali, y_train, y_vali = train_test_split(Og_X_train, Og_y_train, test_size = 0.2)

import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 9)
scores = []

for k in k_range:
    print('k= ' + str(k) + ' began')
    start = time.time()
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_vali)
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred))
    print(confusion_matrix(y_vali, y_pred))
    print('time elapsed: ' + str(end - start) + ' seconds')
print(scores)
plt.plot(k_range, scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
k = 5

model = KNeighborsClassifier(n_neighbors = k)
model.fit(Og_X_train, Og_y_train)
y_pred = model.predict(Og_X_test[:300])
print(y_pred[123])
plt.imshow(Og_X_test[123].reshape(28, 28))
plt.show()
pd.DataFrame({'ImageId': list(range(1, len(y_pred) + 1)), 'Label': y_pred}).to_csv('Digit_Recogniser_Result.csv', index = False, header = True)
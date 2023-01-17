import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import os

os.getcwd()

data_dir = '../input/'



def load_data(data_dir, train_row):

    data_train = pd.read_csv(data_dir + 'train.csv')    

    image_train = data_train.values[0:train_row, 1:]

    label_train = data_train.values[0:train_row, 0]

    image_test = pd.read_csv(data_dir + 'test.csv').values

    

    return image_train, label_train, image_test



train_row = 5000

origin_image_train, origin_label_train, origin_image_test = load_data(data_dir, train_row)



print(origin_image_train.shape)

    

    
print(origin_image_train.shape, origin_label_train.shape, origin_image_test.shape)
row = 678

print(origin_label_train[row])

plt.imshow(origin_image_train[row].reshape((28, 28)))

plt.show()
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

rows = 5

print(classes)



for y, cur_class in enumerate(classes):

    # get the index of the labels which are equal to y in label_train

    idxs = np.nonzero([i == y for i in origin_label_train])

    idxs = np.random.choice(idxs[0], rows)

    

    for x, idx in enumerate(idxs):

        plt_idx = x * len(classes) + y + 1

        plt.subplot(rows, len(classes), plt_idx)

        plt.imshow(origin_image_train[idx].reshape((28, 28)))

        plt.axis('off')

        if x == 0:

            plt.title(y)



plt.show()
from sklearn.model_selection import train_test_split



image_train, image_vali, label_train, label_vali = train_test_split(origin_image_train,

                                                                    origin_label_train,

                                                                    test_size=0.2,

                                                                    random_state=0

                                                                    )

print(image_train.shape, image_vali.shape, label_train.shape, label_vali.shape)

print(type(image_train))
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



ans_k = 0

k_range = range(1, 5)

scores = []



for k in k_range:

    print('k = ' + str(k) + ' begins')

    start = time.time()

    

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=6)

    knn.fit(image_train, label_train)

    label_pred = knn.predict(image_vali)

    

    accuracy = accuracy_score(label_vali, label_pred)

    scores.append(accuracy)

    end = time.time()

    

    print(classification_report(label_vali, label_pred))

    print(confusion_matrix(label_vali, label_pred))

    print("Complete time: " + str(end - start) + 'Secs.\n\n')



print(scores)

plt.plot(k_range, scores)

plt.xlabel('Value of K')

plt.ylabel('Testing Accuracy')

plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

knn.fit(origin_image_train, origin_label_train)

label_pred = knn.predict(origin_image_test)
row = 1250

print(label_pred[row])

plt.imshow(origin_image_test[row].reshape((28, 28)))

plt.show()
print(len(label_pred))

result = pd.DataFrame({'ImageId': list(range(1, len(label_pred) + 1)), 'Label': label_pred})

result.to_csv('Digit_Recogniser_Result.csv', index=False, header=True)
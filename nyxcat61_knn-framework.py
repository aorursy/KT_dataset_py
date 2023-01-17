import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'



def load_data(data_dir):

    train_df = pd.read_csv(data_dir + 'train.csv')

    test_df = pd.read_csv(data_dir + 'test.csv')

    print('Train data shape: %s \n Test data shape: %s' % (train_df.shape, test_df.shape))

    

    # dataframe to np.arrays

    train_X = train_df.values[:, 1:]

    train_y = train_df.values[:, 0]

    test_X = test_df.values

    

    return train_X, train_y, test_X



train_X_origin, train_y_origin, test_X = load_data(data_dir)

    
idx = 10

print('Label: %s' % (train_y_origin[idx]))

plt.imshow(train_X_origin[idx].reshape(28, 28))

plt.show()
nrows = 4

classes = range(0, 10)



for i in classes:

    i_idx = np.nonzero(train_y_origin == i)

    rand_idx = np.random.choice(i_idx[0], nrows)

    for j in range(nrows):

        plt_idx = j * len(classes) + i + 1

        plt.subplot(nrows, len(classes), plt_idx)

        plt.imshow(train_X_origin[rand_idx[j]].reshape(28, 28))

        plt.axis('off')

        if j == 0:

            plt.title(i)

plt.show()
from sklearn.model_selection import train_test_split



train_X, valid_X, train_y, valid_y = train_test_split(train_X_origin, train_y_origin, \

                                                      test_size=0.2, random_state=0)
train_y.shape, valid_y.shape
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier



k = range(1, 10)

accuracy = []



for i in k:

    print('k = %s begin,' % (i))

    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(train_X, train_y)

    pred_y = model.predict(valid_X)

    accuracy.append(accuracy_score(valid_y, pred_y))

    print(classification_report(valid_y, pred_y))

    print(confusion_matrix(valid_y, pred_y))    

    
plt.plot(k, accuracy)

plt.xlabel('Value of k')

plt.ylabel('Accuracy Score')
k = 1

model = KNeighborsClassifier(n_neighbors=k)

model.fit(train_X_origin, train_y_origin)

pred_y = model.predict(test_X)
idx = 161

print('Predicted label: %s' % (pred_y[idx]))

plt.imshow(test_X[idx].reshape(28, 28))

plt.show()
pd.DataFrame({'ImageId': list(range(1, len(pred_y)+1)), 'Label': pred_y}).to_csv(

    'Digit_Recogniser_Result.csv', index=False, header=True)
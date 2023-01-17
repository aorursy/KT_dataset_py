import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data_dir = '../input'



def load_data(data_dir, train_rows):

    data = pd.read_csv(data_dir + '/train.csv')

    print(data.shape)

    X_train = data.iloc[:train_rows, 1:].values

    y_train = data.iloc[:train_rows, 0].values

    

    X_test = pd.read_csv(data_dir + '/test.csv').values

    

    return X_train, y_train, X_test



train_rows = 5000

X_train, y_train, X_test = load_data(data_dir, train_rows)
idx = 10

print('label: ' + str(y_train[idx]))

plt.imshow(X_train[idx].reshape(28,28))

plt.show()
rows = 4

classes = [i for i in range(10)]



for c in classes:

    c_ids = np.where(y_train == c)[0]

    c_ids = np.random.choice(c_ids, rows)

    for i in range(rows):

        plt.subplot(rows, len(classes), c + 1 + i*len(classes))

        plt.imshow(X_train[c_ids[i]].reshape(28,28))

        plt.axis('off')

        if i == 0:

            plt.title(c)

plt.show()
class KNN(object):

    def __init__(self):

        pass

    

    def train(self, X, y):

        self.X = X

        self.y = y

        

    def predict(self, test_x, k=1):

        test_X = np.tile(test_x, (len(self.X), 1))

        diff_X = self.X - test_X

        euclidist = np.sqrt(np.sum(diff_X ** 2, axis=1))

        # print(test_X.shape, diff_X.shape, euclidist.shape)

        sorted_idx = np.argsort(euclidist)

        knn_idx = sorted_idx[:k]

        preds = self.y[knn_idx]

        

        count_class = dict()

        for i in preds:

            count_class[i] = count_class.get(i, 0) + 1

            

        # get most voted

        most_voted, most_count = 0, 0

        for k, v in count_class.items():

            if v > most_count:

                most_count = v

                most_voted = k

        return most_voted
from sklearn.model_selection import train_test_split

# train test split

xtrain, xvalid, ytrain, yvalid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
from sklearn.metrics import accuracy_score



model = KNN()

model.train(xtrain, ytrain)



# choose best k

k_range = range(1, 9)

best_score = 0

best_k = 0



for k in k_range:

    print('Start training, k = %s' % (k))

    predictions = []

    for i in range(len(xvalid)):

        if i % 1000 == 0:

            print('Computing %s/%s...' % (i, len(xvalid)))

        pred = model.predict(xvalid[i], k=k)

        predictions.append(pred)

    accuracy = accuracy_score(yvalid, np.array(predictions))

    print('Finish training, accuracy score = %.5f' % (accuracy))

    # find k that max the accuracy score

    if accuracy > best_score:

        best_score = accuracy

        best_k = k

print('Best k = %s, largest accuracy score = %.5f' % (best_k, best_score))
# best score happen with k=3, train with all data, and test the first 500 test data

model = KNN()

model.train(X_train, y_train)

pred_test = np.zeros(500)



for i in range(500):

    if i % 100 == 0:

        print('Computing %s/%s...' % (i, 500))

    pred_test[i] = model.predict(X_test[i], k=3)
# show the train results

idx = 157

print('label: ' + str(int(pred_test[idx])))

plt.imshow(X_test[idx].reshape(28,28))

plt.show()
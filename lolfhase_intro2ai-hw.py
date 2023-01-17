%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# prepare data
def load_data(data_dir, row_no):
    data = pd.read_csv(data_dir+"train.csv")
    print(data.shape)
    X = data.values[:row_no,1:]
    Y = data.values[:row_no,0]
    test_x = pd.read_csv(data_dir+"test.csv").values[:]
    return X, Y, test_x

# read data
data_dir = '../input/'
train_x, train_y, test_x = load_data(data_dir,10000)
print(train_x.shape, train_y.shape, test_x.shape)
plt.imshow(train_x[233].reshape(28,28))
plt.show()

# split the train dataset
from sklearn.model_selection import train_test_split
x_train, x_vali, y_train, y_vali = train_test_split(train_x, train_y, test_size = 0.3, random_state = 0)
print(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)


# define KNN Classifier
class KNNClassifier(object):
    def configure(self, K, disfun):
        """define the KNN Classifier with K and distance function"""
        self.K = K
        self.disfun = disfun

    def train(self, X, Y):
        """load train data """
        self.Xtr = X
        self.Ytr = Y

    def predict(self, X):
        """predict the result of input data X"""
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.Ytr.dtype)

        for i in range(num_test):
            distances = self.disfun(self.Xtr, X[i])
            k_nearest = np.argsort(distances)[:self.K]
            k_vote = []
            for u in k_nearest:
                k_vote.append(self.Ytr[u])
            k_vote = np.bincount(np.array(k_vote))
            # print(np.argmax(k_vote))
            y_pred[i] = np.argmax(k_vote)

        return y_pred


def L1dis(X1, X2):
    return np.sum(np.abs(X1 - X2), axis=1)


def L2dis(X1, X2):
    return np.sqrt(np.sum(np.square(X1 - X2), axis=1))



# initialize training
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
scores = []
times = []
KNN = KNNClassifier()

# training and select a best K as hyper parameter
for k in range(1, 9):
    start=time.time()
    KNN.configure(k, L2dis)
    KNN.train(x_train, y_train)
    res = KNN.predict(x_vali)
    accuracy = accuracy_score(y_vali, res)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, res))
    print("Complete time: " + str(end - start) + " Secs.")
    times.append(end - start)

# show train results
plt.figure(figsize=(10,10),dpi=80)
print(scores)
plt.subplot(211)
plt.plot(range(1,9),scores)
plt.scatter(range(1,9),scores)
plt.title('Validation Accuracy with Different Ks')
plt.xlabel('Value of K')
plt.subplot(212)
plt.plot(range(1,9),times)
plt.scatter(range(1,9),times)
plt.title('Time Cost with Different Ks')
plt.xlabel('Value of K')
plt.show()

# get the final KNN model
k_res = np.argmax(scores)+1
KNN.configure(k_res,L2dis)
y_pred = KNN.predict(test_x)
print(len(y_pred))
# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('result_10k.csv', index=False,header=True)
# check model performance in test set
def testimg(x):
    plt.imshow(test_x[x].reshape(28, 28))
    plt.show()
    print('Testing Image #%d'%(x))
    print('Testing result: %d'%(y_pred[x]))
togo = np.random.randint(0,28000,5)
for i in togo:
    testimg(i)

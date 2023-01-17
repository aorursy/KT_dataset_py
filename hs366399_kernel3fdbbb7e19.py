import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import os

from tqdm import tqdm_notebook
path = '../input/level_4b_train/level_4b'

dirs = ['background', 'en', 'hi', 'ta']

train_y = []

train_x = []

test_x = []



for dir in dirs:

    label = int((dir != 'background'))

    for images in tqdm_notebook(os.listdir(path + '/' + dir)):

        img = cv2.imread(path + '/' + dir + '/' + images, 0)

        train_y.append(np.asarray([label]).reshape(1,1))

        train_x.append(img.reshape(img.shape[0] * img.shape[1], 1))    



train_X = np.concatenate(train_x, axis=1)

train_Y = np.concatenate(train_y, axis=1)

m = train_X.shape[1]

permutations = list(np.random.permutation(m))

train_X = train_X[:, permutations]

train_Y = train_Y[:, permutations]



print("Training set size : ", train_X.shape)

print("Label size : ", train_Y.shape)

class PerceptronWithSigmoid:

    

    def __init__(self, train_X, train_Y):

        

        self.train_X = train_X/255

        self.train_Y = train_Y

        self.m = self.train_X.shape[1]

        self.lr = None

        self.W = np.random.randn(1, self.train_X.shape[0]) * 0.01

        self.b = np.ones([1,1])

    

    def shuffle_training_data(self):

        m = train_X.shape[1]

        permutations = list(np.random.permutation(m))

        self.train_X = self.train_X[:, permutations]

        self.train_Y = self.train_Y[:, permutations]

    

    def sigmoid(self,Z):

        return 1/(1 + np.exp(-Z))

        

    def update_parameters(self, pred_Y):

        

        diff = pred_Y - self.train_Y

        dW = (1/self.m) * np.sum(diff * self.train_X, axis = 1, keepdims=True).T

        db = (1/self.m) * np.sum(diff, keepdims=True)

        self.W = self.W - (self.lr * dW)

        self.b = self.b - (self.lr * db)

        

    def compute_cost(self, pred_Y):

        

        Y = self.train_Y

        cost = (-1/m) * np.sum(Y * np.log(pred_Y) + (1 - Y) * np.log(1 - pred_Y))

        self.update_parameters(pred_Y)

        return cost

        

    def fit(self, epochs = 5, lr = 0.01):

        

        self.lr = lr

        accuracy = []

        costs = []

        

        for i in tqdm_notebook(range(epochs)):

            pred_Y = self.sigmoid(np.dot(self.W, self.train_X) + self.b)

            cost = self.compute_cost(pred_Y)

            costs.append(cost)

            

            pred_Y[pred_Y < 0.5] = 0

            pred_Y[pred_Y >= 0.5] = 1

            diff = np.sum(np.absolute(self.train_Y - pred_Y))/self.m

            acc = 100 - (100 * diff)

            accuracy.append(acc)

        

        plt.title("Accuracy v/s Iterations")

        plt.plot(accuracy)

        plt.show()

        

        plt.title("Cost v/s Iterations")

        plt.plot(costs)

        plt.show()

        

        print("Train Accuracy : ", accuracy[-1])

        

    def predict(self):

        path_1 = '../input/level_4b_test/kaggle_level_4b'

        predictions = []

        ids = []

        for images in sorted(os.listdir(path_1)):

            x = cv2.imread(path_1 + '/' + images, 0)

            x = x.reshape(x.shape[0] * x.shape[1], 1)

            predictions.append(self.sigmoid(np.dot(self.W, x) + self.b))

            ids.append(np.asarray(int(images.split('.')[0])).reshape(1,1))



        ids = np.concatenate(ids, axis=1).T.ravel()

        pred_Y = np.concatenate(predictions, axis=1).T

        pred_Y = (pred_Y >= 0.5).astype("int").ravel()

        print(ids.shape)

        print(pred_Y.shape)

        

        return ids, pred_Y

            

            
model = PerceptronWithSigmoid(train_X, train_Y)

model.fit(8500, 0.009)

ids, pred_Y = model.predict()



submission = {}

submission['ImageId'] = ids

submission['Class'] = pred_Y



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)
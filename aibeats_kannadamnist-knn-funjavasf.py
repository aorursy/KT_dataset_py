import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 
#data_dir = "../input/digit-recognizer/"

data_dir = '../input/Kannada-MNIST/'

train_row = 60000 #max = 60000

max_k = 10

run_find_k = train_row <= 5000
train = pd.read_csv(data_dir + "train.csv")

train #(60000, 785)
test = pd.read_csv(data_dir + "test.csv")

test #(5000, 785)
Origin_X_train = train.values[0:train_row,1:] 

Origin_y_train = train.values[0:train_row,0] 

Origin_y_test = test.values[:,1:]
print(Origin_X_train.shape, Origin_y_train.shape, Origin_y_test.shape)
import matplotlib

import matplotlib.pyplot as plt

row = 6



print ('Label is: ' + str(Origin_y_train[row]))



plt.imshow(Origin_X_train[row].reshape((28, 28)))

plt.show()
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

rows = 6



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



X_train, X_vali, y_train, y_vali = train_test_split(Origin_X_train,

                                                   Origin_y_train,

                                                   test_size = 0.2,

                                                   random_state = 0)



print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
class knn():

    def __init__(self):

        pass



    def train(self, X, y):

        self.X_train = X

        self.y_train = y



    def predict(self, X, k=3):

        dataSet = self.X_train

        labels = self.y_train

      

        dataSetSize = dataSet.shape[0]

        diffMat = np.tile(X,(dataSetSize,1)) - dataSet

        sqDiffMat = diffMat**2

        sumDiffMat = sqDiffMat.sum(axis=1)

        distances = sumDiffMat**0.5

        sortedDistances = distances.argsort()

        

        classCount = {}

        

        for i in range(k):

            vote = labels[sortedDistances[i]]

            classCount[vote] = classCount.get(vote,0) + 1

        max = 0

        ans = 0

        for k,v in classCount.items():

            if v > max:

                ans = k

                max = v

        return ans
from sklearn.metrics import accuracy_score



classifier = knn()

classifier.train(X_train, y_train)



def find_k():

    max = 0

    ans_k = 0



    for k in range(1, max_k):

        print ('when k = ' + str(k) + ', start training')

        predictions = np.zeros(len(y_vali))

        for i in range(X_vali.shape[0]):

            if i % 500 == 0:

                print("Computing  " + str(i+1) + "/" + str(int(len(X_vali))) + "...")

            output = classifier.predict(X_vali[i], k)

            predictions[i] = output

        accuracy = accuracy_score(y_vali, predictions)

        print ('k = '+ str(k) , ' accuracy =' + str(accuracy))

        if max < accuracy:

            ans_k = k

            max = accuracy

    print("best k =" + str(ans_k))

    return ans_k
k = find_k() if run_find_k else 3

predictions = np.zeros(Origin_y_test.shape[0])

for i in range(Origin_y_test.shape[0]):

    if i % 100 ==0:

        print("Computing  " + str(i+1) + "/" + str(int(len(Origin_y_test))) + "...")

    predictions[i] = classifier.predict(Origin_y_test[i], k)

print (predictions[4905])

plt.imshow(Origin_y_test[4905].reshape((28, 28)))
for y, cls in enumerate(classes):

    idxs = np.nonzero([i == y for i in predictions])

    idxs = np.random.choice(idxs[0], rows)

    for i , idx in enumerate(idxs):

        plt_idx = i * len(classes) + y + 1

        plt.subplot(rows, len(classes), plt_idx)

        plt.imshow(Origin_y_test[idx].reshape((28, 28)))

        plt.axis("off")

        if i == 0:

            plt.title(cls)

        



plt.show()
sample_submisison = pd.read_csv(data_dir + "sample_submission.csv")

sample_submisison #(60000, 785)
print(len(predictions))

out_file = open("submission.csv", "w")

out_file.write("id,label\n")

for i in range(len(predictions)):

    out_file.write(str(i) + "," + str(int(predictions[i])) + "\n")

out_file.close()
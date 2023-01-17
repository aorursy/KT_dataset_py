# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from matplotlib import patches

import seaborn as sns

from sklearn.preprocessing import LabelBinarizer, StandardScaler

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.tail(10)


plt.figure(figsize=(20, 15))

for ii, col in enumerate(data.columns[2:11]):

    plt.subplot(3,3,ii+1)

    plt.title(col)

    plt.legend(handles = [patches.Patch(label = 'Malignant', color=(.43,.23,.54)),

                          patches.Patch(label = 'Benign', color=(.63,.83,.24))])

    b = data[data['diagnosis'] == 'B'][col]

    m = data[data['diagnosis'] == 'M'][col]

    plt.hist(m, stacked=True, normed = True, color=(.43,.23,.54))

    plt.hist(b, stacked=True, normed = True, color=(.63,.83,.24))

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 15))

for ii, col in enumerate(data.columns[11:20]):

    plt.subplot(3,3,ii+1)

    plt.title(col)

    plt.legend(handles = [patches.Patch(label = 'Malignant', color=(.43,.23,.54)),

                          patches.Patch(label = 'Benign', color=(.63,.83,.24))])

    b = data[data['diagnosis'] == 'B'][col]

    m = data[data['diagnosis'] == 'M'][col]

    plt.hist(m, stacked=True, normed = True, color=(.43,.23,.54))

    plt.hist(b, stacked=True, normed = True, color=(.63,.83,.24))

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 15))

for ii, col in enumerate(data.columns[20:29]):

    plt.subplot(3,3,ii+1)

    plt.title(col)

    plt.legend(handles = [patches.Patch(label = 'Malignant', color=(.43,.23,.54)),

                          patches.Patch(label = 'Benign', color=(.63,.83,.24))])

    b = data[data['diagnosis'] == 'B'][col]

    m = data[data['diagnosis'] == 'M'][col]

    plt.hist(m, stacked=True, normed = True, color=(.43,.23,.54))

    plt.hist(b, stacked=True, normed = True, color=(.63,.83,.24))

    

plt.tight_layout()

plt.show()
binarizer = LabelBinarizer().fit(data['diagnosis'])

data.iloc[:,2:32] = StandardScaler().fit_transform(data.iloc[:,2:32])

data['diagnosis'] = binarizer.transform(data['diagnosis'])



data.tail(10)
train_attrs = data.iloc[:400,2:32].as_matrix()

train_labels = data.iloc[:400,1].as_matrix()

test_attrs = data.iloc[400:,2:32].as_matrix()

test_labels = data.iloc[400:,1].as_matrix()
def labeler(x):

    

    return np.array([0 if i<0.5 else 1 for i in x])



class LogisticRegression:

    

    def __init__(self, X, y, max_iter=1000):

        

        self.X = self.add_bias(X)

        self.y = y

        self.weights = np.random.randn(self.X.shape[1]) * 15

        self.max_iter = max_iter

        

    def add_bias(self, X):

        

        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    

    def sigmoid(self, z):

        

        return 1.0/(1.0 + np.exp(-z))

    

    def gradient(self, weights):

        

        def partial_derivative(weight):

            

            return 1.0/len(self.X) * sum((self.predict(self.X) - self.y) * (self.X[:, weight]))

       

        gradient = np.array([partial_derivative(weight) for weight in range(len(weights))])

        return gradient

    def gradient_descent(self, learning_rate=0.01, loop=0):

        

        while loop < self.max_iter:

            self.weights = self.weights - (learning_rate * self.gradient(self.weights))

            if loop % 100 == 0:

                acc = (test_labels == labeler(clf.predict(clf.add_bias(test_attrs)))).mean()

                print("At iteration {0} the accuracy is {1}".format(loop, acc))

            loop+=1

            

            

        return self

    def predict(self, X):

      

        return np.array([self.sigmoid(np.dot(self.weights, x)) for x in X])
clf = LogisticRegression(train_attrs, train_labels, 10000)

clf.gradient_descent()
(test_labels == labeler(clf.predict(clf.add_bias(test_attrs)))).mean()
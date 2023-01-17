# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/voice.csv')

data.iloc[:, 0:20] = MinMaxScaler().fit_transform(data.iloc[:, 0:20])

data.iloc[:, 0:20] = StandardScaler().fit_transform(data.iloc[:, 0:20])

data.head()
def stack_hist(col1):

    

    x1 = data[data['label'] == 'male'][col1]

    x2 = data[data['label'] == 'female'][col1]

    plt.hist(x1, stacked=True, normed=True, color=(.03,.32,.53))

    plt.hist(x2, stacked=True, normed=True, color=(.53,.36,.93))

    plt.xlabel(col1)

    plt.legend(handles = [patches.Patch(label='Male', color = (.03,.32,.53)), patches.Patch(label = 'Female', color = (.53,.36,.93))])

    plt.show()
sns.boxplot(x = 'label', y ='sd', data=data)
sns.boxplot(x = 'label', y ='median', data=data)
sns.boxplot(x = 'label', y ='sfm', data=data)
stack_hist('sd')

stack_hist('meanfreq')

stack_hist('centroid')
def labeler(x):

    

    if x >= 0.5:

        

        return 1

    

    else:

        return 0



class LogisticRegression:

    

    def __init__(self, X, y, max_iter=1000):

        

        self.X = self.add_bias(X)

        self.y = y

        self.weights = np.random.randn(self.X.shape[1])

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

                

                print("Loop {0}/{1}".format(loop, self.max_iter))

            loop+=1

            

        return self

    

    def predict(self, X):

        

        if len(X.shape) < 2:

            

            return labeler(self.sigmoid(np.dot(self.weights, X)))

        else:

            

            return np.array([labeler(self.sigmoid(np.dot(self.weights, x))) for x in X])
data['label'] = LabelBinarizer().fit_transform(data['label'])
data = shuffle(data)

features = data.iloc[:,0:20].as_matrix()

labels = data['label'].as_matrix()
clf = LogisticRegression(features[:2900,:], labels[:2900], 5000)

clf.gradient_descent()
accuracy_score(labels[2900:], clf.predict(clf.add_bias(features[2900:,:])))
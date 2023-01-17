import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import warnings

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore') 

plt.style.use('fivethirtyeight')  
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()
iris.drop('Id', axis=1, inplace=True)
iris.isnull().sum()
iris.info()
iris['Species'].value_counts()
sns.countplot('Species',data=iris)



plt.show()
plt.figure(figsize=(15, 10))



plt.subplot(2, 2, 1)

sns.violinplot(x='Species', y='PetalLengthCm', data=iris)  



plt.subplot(2, 2, 2)

sns.violinplot(x='Species', y='PetalWidthCm', data=iris)



plt.subplot(2, 2, 3)

sns.violinplot(x='Species', y='SepalLengthCm', data=iris)



plt.subplot(2, 2, 4)

sns.violinplot(x='Species', y='SepalWidthCm', data=iris)



plt.show()
sns.pairplot(iris, hue='Species')

plt.show()
iris.hist(edgecolor='black', linewidth=1.2)



fig = plt.gcf()

fig.set_size_inches(12, 6)

map_fn = {

    "Iris-setosa": 0,

    "Iris-versicolor": 1,

    "Iris-virginica": 2,

}



iris["Species"] = iris["Species"].map(map_fn)
Y = iris.pop("Species").values

X = iris.values
class One_hot_encoder:

    

    def fit(self, X):

        self.classes = np.unique(X)

        

    def transform(self, X):

        m = X.shape[0]

        labels = np.zeros((m, len(self.classes)))

        

        for cls in self.classes:

            labels[np.where(X[:] == cls),cls] = 1

            

        return labels

        

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
one_hot = One_hot_encoder()



one_hot.fit(Y)

Y = one_hot.transform(Y)
print("labels classes:", one_hot.classes)

print("labels shape:", Y.shape)
def sigmoid(X):

    return 1 /(1+np.exp(-X))



def sigmoid_der(X):

    return sigmoid(X)*(1-sigmoid(X))
x = np.linspace(-10, 10, 100)



plt.plot(sigmoid(x))

plt.plot(sigmoid_der(x))



plt.show()
def softmax(X):

    exps = np.exp(X.T - np.max(X, axis=-1))

    return (exps / exps.sum(axis=0)).T



def softmax_der(X):

    return softmax(X)*(1-softmax(X))
softmax(X[0:2])
def Loss(y, pred):

    return (pred-y)
n_in = X.shape[1]

n_hidden = 8

n_out = Y.shape[1]



epochs = 10000

lr = 0.02

keep_prob = 0.4



prob = 1 - keep_prob
W1 = 2*np.random.random((n_in, n_hidden))-1

b1 = np.zeros((n_hidden))



W2 = 2*np.random.random((n_hidden, n_out))-1

b2 = np.zeros((n_out))
for i in range(epochs):

    z1 = np.add(np.matmul(X, W1), b1)

    a1 = sigmoid(z1)

    

    #drop forward

    mask = np.random.binomial(1, prob, size=a1.shape)

    mask = mask / keep_prob

    out = a1 * mask

    out.reshape(a1.shape)

    

    z2 = np.add(np.matmul(a1, W2), b2)

    a2 = softmax(z2)

    

    loss = Loss(Y, a2)

    

    delta2 = loss*softmax_der(z2)

    #drop backward

    z1 = z1 * mask

    delta1 = np.dot(delta2, W2.T)*sigmoid_der(z1)

   

    W2 -= a1.T.dot(delta2) * lr

    W1 -= X.T.dot(delta1) * lr 

    

    b2 = b2 - np.sum(delta2, axis=0) * lr

    b1 = b1 - np.sum(delta1, axis=0) * lr
def predict(X):

    z1 = np.add(np.matmul(X, W1), b1)

    a1 = sigmoid(z1)

    

    z2 = np.add(np.matmul(a1, W2), b2)

    a2 = softmax(z2)

    return a2
def accuracy(y, pred):

    s = 0



    for i in range(len(y)):

        if y[i]==pred[i]:

            s += 1

            

    return s/len(y) 
acc = accuracy(np.argmax(Y, axis=1),

               np.argmax(predict(X), axis=1))



print("Accuracy: ", acc)
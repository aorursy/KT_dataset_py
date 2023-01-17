# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import imgaug.augmenters as iaa



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
X = df.iloc[:, 1:]

Y = df.iloc[:, 0]
X.head(2)
Y.head()
X = np.array(X)

Y = np.array(Y)
X = X/255.0
def plot_images(X, Y, shape):

    for i in range(20):

        plt.subplot(5, 4, i+1)

        plt.tight_layout()

        plt.imshow(X[i].reshape((shape,shape)), cmap='gray')

        plt.title('Digit:{}'.format(Y[i]))

        plt.xticks([])

        plt.yticks([])

    plt.show()
plot_images(X[:20], Y[:20], 28)
class Util:

    def __init__(self):

        self.name = "Util"

        

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    

    def sigmoid_prime(self, z):

        return -np.exp(-z)/(1 + np.exp(-z))**2

    

    def softmax(self, z):

        expA = np.exp(z)

        return expA/expA.sum(axis=1, keepdims=True)

        

    def softmax_prime(self, z):

        s = self.softmax(z)

        return s*(1-s)

    

    def tanh(self, z):

        return np.tanh(z)

    

    def tanh_prime(self, z):

        return 1 - np.tanh(z)**2

    

    def relu(self, z):

        return np.where(z>0, z, 0)

    

    def relu_prime(self, z):

        return np.where(z>0, 1, 0)

    

    def sigmoid_cost_function(self, P, T):

        m = P.shape[0]

        return -np.sum(T*np.log(P) + (1-T)*np.log(1-P))/m

    

    def softmax_cost_function(self, P, T, parameters, lambd):

        m = P.shape[0]

        sum_weights = 0

        L = len(parameters)//2

        for l in range(L):

            sum_weights += np.sum(parameters["W" + str(l + 1)])

        J = (lambd / (2*m)) * (sum_weights**2)

        J += -np.sum(T*np.log(P))/m

        return J

    

    def accuracy(self, P, Y):

        Yhat = np.argmax(P, axis=1)

        return np.mean(Yhat==Y)

    

    def yEncode(self, Y):

        T = np.zeros((len(Y), len(set(Y))))

        T[np.arange(Y.size), Y] = 1

        return T
class ImageAugmenter:

    def __init__(self):

        self.name = 'ImageAugmenter'

        

    def reshape_images(self, img_arr, shape):

        return img_arr.reshape(shape)

    

    def transform_images(self, seq, img_arr, shape):

        X_img = self.reshape_images(img_arr, (img_arr.shape[0], shape, shape))

        X_aug = seq.augment_images(X_img)

        X_aug = self.reshape_images(X_aug, (img_arr.shape[0], shape*shape))

        return X_aug

        

    def fliplr(self, X, shape):

        seq = iaa.Sequential([

            iaa.Fliplr(1)

        ])

        return self.transform_images(seq, X, shape)

    

    def flipud(self, X, shape):

        seq = iaa.Sequential([

            iaa.Flipud(1)

        ])

        return self.transform_images(seq, X, shape)

    

    def scale(self, X, shape):

        seq = iaa.Sequential([

            iaa.Affine(

                scale={"x":(0.5, 1.5), "y":(0.5, 1.5)}

            )

        ])

        return self.transform_images(seq, X, shape)

    

    def translate(self, X, shape):

        seq = iaa.Sequential([

            iaa.Affine(

                translate_percent={"x":(-0.2, 0.2), "y":(-0.2, 0.2)}

            )

        ])

        return self.transform_images(seq, X, shape)

    

    def rotate(self, X, shape):

        seq = iaa.Sequential([

            iaa.Affine(

                rotate=(-45, 45)

            )

        ])

        return self.transform_images(seq, X, shape)

    

    def shear(self, X, shape):

        seq = iaa.Sequential([

            iaa.Affine(

                shear=(-10, 10)

            )

        ])

        return self.transform_images(seq, X, shape)

    

    def compose(self, X, shape):

        seq = iaa.Sequential([

            iaa.Affine(

            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},

            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},

            rotate=(-25, 25),

            shear=(-8, 8)

            ),

            iaa.Pepper(1e-5)

        ])

        return self.transform_images(seq, X, shape)

    

    def augment(self, count, X, Y):

        X_out = np.copy(X)

        for i in range(count):

            X_scale = self.scale(X, 28)

            X_rotate = self.rotate(X, 28)

            X_trans = self.translate(X, 28)

            X_shear = self.shear(X, 28)

            X_compose = self.compose(X, 28)

            X_out = np.concatenate((X_out, X_scale), axis = 0)

            X_out = np.concatenate((X_out, X_rotate), axis = 0)

            X_out = np.concatenate((X_out, X_trans), axis = 0)

            X_out = np.concatenate((X_out, X_shear), axis = 0)

            X_out = np.concatenate((X_out, X_compose), axis = 0)

        Y = np.repeat(Y, (count*5)+1, axis = 0)

        return X_out, Y
class DNN_Core:

    def __init__(self, layer_dims):

        self.layer_dims = layer_dims

        

    def initialise_parameters(self):

        parameters = {}

        for l in range(1, len(self.layer_dims)):

            parameters["W" + str(l)] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l])*0.01

            parameters["b" + str(l)] = np.random.randn(self.layer_dims[l], 1)*0.01

        return parameters

        

    def linear_forward(self, A, W, b):

        z = np.dot(A, W) + b.T

        return z, (A, W, b)

    

    def activation_forward(self, A, W, b, activation):

        util = Util()

        z, linear_cache = self.linear_forward(A, W, b)

        if activation == 'sigmoid':

            A = util.sigmoid(z)

        elif activation == 'softmax':

            A = util.softmax(z)

        elif activation == 'tanh':

            A = util.tanh(z)

        elif activation == 'relu':

            A = util.relu(z)

        activation_cache = z

        return A, (linear_cache, activation_cache)

    

    def feed_forward(self, X, parameters):

        caches = []

        A = X

        L = len(parameters)//2

        for l in range(1, L):

            A_prev = A

            A, cache = self.activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], 'relu')

            caches.append(cache)

        P, cache = self.activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'softmax')

        caches.append(cache)

        return P, caches

    

    def linear_backward(self, dZ, linear_cache):

        A_prev, W, b = linear_cache

        m = A_prev.shape[0]

        dW = np.dot(A_prev.T, dZ)/m

        db = np.sum(dZ, axis=0, keepdims=True)/m

        dA_prev = np.dot(dZ, W.T)/m

        return dA_prev, dW, db

    

    def activation_backward(self, dA, cache, activation):

        util = Util()

        linear_cache, activation_cache = cache

        if activation == 'sigmoid':

            dZ = dA * util.sigmoid_prime(activation_cache)

        elif activation == 'softmax':

            dZ = dA

        elif activation == 'tanh':

            dZ = dA * util.tanh_prime(activation_cache)

        elif activation == 'relu':

            dZ = dA * util.relu_prime(activation_cache)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db.T

    

    def back_propogation(self, P, T, caches):

        gradients = {}

        dA = P-T

        L = len(caches)

        dA_prev, dW, db = self.activation_backward(dA, caches[L-1], 'softmax')

        gradients['dW'+str(L)] = dW

        gradients['db'+str(L)] = db

        gradients['dA'+str(L-1)] = dA_prev

        for l in reversed(range(L-1)):

            dA_prev, dW, db = self.activation_backward(dA_prev, caches[l], 'relu')

            gradients['dW'+str(l+1)] = dW

            gradients['db'+str(l+1)] = db

            gradients['dA'+str(l)] = dA_prev

        return gradients

    

    def update_patameters(self, parameters, gradients, alpha, lambd, m):

        L = len(parameters)//2

        for l in range(L):

            parameters['W'+str(l+1)] -= alpha*gradients['dW'+str(l+1)] 

            parameters['W'+str(l+1)] -= lambd*parameters['W'+str(l+1)]

            parameters['b'+str(l+1)] -= alpha*gradients['db'+str(l+1)]

        return parameters
class SGD:

    def __init__(self, layer_dims):

        self.layer_dims = layer_dims

        

    def SGD(self, X, X_val, Y, Y_val, epochs, alpha, lambd, batch_size):

        J_history = []

        J_val_history = []

        acc_history = []

        acc_val_history = []

        core = DNN_Core(self.layer_dims)

        util = Util()

        augmenter = ImageAugmenter()

        T = util.yEncode(Y)

        T_val = util.yEncode(Y_val)

        parameters = core.initialise_parameters()

        for i in range(epochs):

            start = 0

            s = np.arange(X.shape[0])

            np.random.shuffle(s)

            X = X[s]

            T = T[s]

            Y = Y[s]

            while start < X.shape[0]:

                end = start + batch_size

                if end > X.shape[0]:

                    end = X.shape[0]

                x = X[start:end, :]

                t = T[start:end, :]

                #x, t = augmenter.augment(x, t)

                P, caches = core.feed_forward(x, parameters)

                gradients = core.back_propogation(P, t, caches)

                parameters = core.update_patameters(parameters, gradients, alpha, lambd, t.shape[0])

                start = end

            # Validate at the end of 10 epochs

            if i%20 == 0:

                P, _ = core.feed_forward(X, parameters)

                P_val, _ = core.feed_forward(X_val, parameters)

                J = util.softmax_cost_function(P, T, parameters, lambd)

                J_val = util.softmax_cost_function(P_val, T_val, parameters, lambd)

                J_history.append(J)

                J_val_history.append(J_val)

                acc = util.accuracy(P, Y)

                acc_val = util.accuracy(P_val, Y_val)

                acc_history.append(acc)

                acc_val_history.append(acc_val)

                print('Epoch: ' + str(i), 'Cost: {0:.2f} Valid_Cost: {1:.2f} Accuracy: {2:.7f} Valid_Accuracy: {3:.7f}'.format(J, J_val, acc, acc_val))

        return P, parameters, J_history, J_val_history, acc_history, acc_val_history
from sklearn.model_selection import  train_test_split

layer_dims = [X.shape[1], 30, 20, 10]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.05, random_state=28) 

sgd = SGD(layer_dims)

P, parameters, J_history, J_val_history, acc_history, acc_val_history = sgd.SGD(X_train, X_val, Y_train, Y_val, 500, 1e1, 1e-4, 100)
plt.plot(J_history, color='blue')

plt.plot(J_val_history, color='red')

plt.show()
plt.plot(acc_history, color='blue')

plt.plot(acc_val_history, color='red')

plt.show()
core = DNN_Core(layer_dims)

util = Util()

P_test, _ = core.feed_forward(X_val, parameters)

print(util.accuracy(P_test, Y_val))
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test.shape
X_out = np.array(df_test)

X_out = X_out/255.0

P_out, _ = core.feed_forward(X_out, parameters)
Y_out = np.argmax(P_out, axis=1)

Y_out[1:5]
df_submission = pd.read_csv('../input/sample_submission.csv')

df_submission['Label'] = Y_out

df_submission.head()
df_submission.to_csv('output.csv', index=False)
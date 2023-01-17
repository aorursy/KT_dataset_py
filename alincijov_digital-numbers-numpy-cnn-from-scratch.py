# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear ablgebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv(dirname + "/train.csv")
df.head()
train_x = df.iloc[:,1:].values.reshape(len(df), 28, 28)
train_y = df['label'].values
class Conv3x3:
    
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9
        
    def iterate_regions(self, image):
        h,w = image.shape
        
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
                
    def forward(self, input):
        self.last_input = input
        
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
            
        return output
    
    def backprop(self, dl_dout, lr):
        dl_dfilters = np.zeros(self.filters.shape)
        
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                dl_dfilters[f] += dl_dout[i, j, f] * im_region
                
        self.filters -= lr * dl_dfilters
        
        return None
class MaxPool2:

    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j
            
    def forward(self, input):
        self.last_input = input
        
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output
    
    def backprop(self, dl_dout):
        dl_dinput = np.zeros(self.last_input.shape)
        
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))
            
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            dl_dinput[i * 2 + i2, j * 2 + j2, f2] = dl_dout[i, j, f2]
            
        return dl_dinput
class Softmax:
        
    def forward(self, input, theta):
        w, b = theta
        
        self._x_shape = input.shape
        
        input = input.flatten()
        self.last_x = input
        
        input_len, nodes = w.shape

        self._x =  model(input, theta)
        
        return softmax(self._x)
    
    
    def backprop(self, dl_dout, theta, lr):
        w, b = theta
        
        for i, gradient in enumerate(dl_dout):
            if gradient == 0:
                continue
            
            dout_dt = dsoftmax(self._x, i)

            # Gradients of loss against totals
            dl_dt = gradient * dout_dt
            
            dl_dw = self.last_x[np.newaxis].T @ dl_dt[np.newaxis]
            dl_db = dl_dt * 1
            dl_dinputs = w @ dl_dt
            
            grads = dl_dw, dl_db
            
            theta = optimize(theta, grads, lr)
            
        return dl_dinputs.reshape(self._x_shape), theta
# the model we want to use
def model(x, theta):
    w, b = theta
    return np.dot(x, w) + b
# SGD optimization
def optimize(theta, grads, lr):
    w, b = theta
    dw, db = grads

    w -= lr * dw
    b -= lr * db

    return w, b
# softmax and derivative of softmax
def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0)


def dsoftmax(x, i):
    t_exp = np.exp(x)
    s = np.sum(t_exp)

    dout_dt = -t_exp[i] * t_exp / (s ** 2)
    dout_dt[i] = t_exp[i] * (s - t_exp[i]) / (s ** 2)

    return dout_dt
# loss function and derivative
def cross_entropy(out, label):
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    
    return loss, acc


def dcross_entropy(out, label):
    # 10 digits
    gradient = np.zeros(10)
    
    # gradient descent of cross entropy
    gradient[label] = -1 / out[label]
    return gradient
def forward(x, y, theta):
    
    # normalize the data
    out = conv.forward((x / 255) - 0.5)
    out = pool.forward(out)
    out = soft.forward(out, theta)
    
    loss, acc = cross_entropy(out, y)
    
    return out, loss, acc
def train(x, y, theta, lr=.001):
    # Forward
    out, loss, acc = forward(x, y, theta)

    # Calculate initial gradient
    gradient = dcross_entropy(out, y)
    
    # Backprop
    gradient, theta = soft.backprop(gradient, theta, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc, theta
def predict(x, theta):
    out = conv.forward((x / 255) - 0.5)
    out = pool.forward(out)
    out = soft.forward(out, theta)
    return np.argmax(out)
df_test = pd.read_csv(dirname + "/test.csv")

test_x = df_test.values.reshape(len(df_test), 28, 28)
result = []

for i in range(len(test_x)):
    result.append([i + 1, predict(test_x[i], theta)])
df_final = pd.DataFrame(data=result, columns=["ImageId", "Label"])
df_final.to_csv('submission.csv', index=False)
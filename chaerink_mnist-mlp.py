# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
# Code Essence Provided by professor Sungwoo Kim of SNU



import numpy as np



class perceptron:

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):

        self.w1 = np.random.normal(0.0, pow(hidden_dim, -0.5), (input_dim, hidden_dim))

        self.w2 = np.random.normal(0.0, pow(hidden_dim, -0.5), (hidden_dim, output_dim))

        self.h = np.zeros((1, hidden_dim))

        self.lr = lr

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.output_dim = output_dim

        self.theta = 0



    def softmax(self, x):

        e_x = np.exp(x-np.max(x))

        return e_x / e_x.sum(axis=0)



    def sigmoid(self, x):

        return 1 / (1+np.exp(-x))



    def relu(self, x):

        return x * (x > 0)



    def relu_prime(self, x):

        return (x>0).astype(x.dtype)



    def tanh(self, x):

        return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))



    def feedforward(self, x):

        a = x.astype(float)

        b = self.w1.astype(float)

        self.h = self.relu(np.dot(a,b))

        return self.relu(np.dot(self.h, self.w2))





    def bprop_w2(self, g, y):

        q = (-2)*(g-y)*self.relu_prime(np.dot(self.h, self.w2))

        return np.dot(self.h.reshape(self.hidden_dim, 1), q.reshape(1, self.output_dim))



    def bprop_w1(self, g, y, x):

        q1 = (-2)*(g-y)*self.relu_prime(np.dot(self.h, self.w2))

        q2 = np.dot(self.w2, q1)

        q3 = q2*self.relu_prime(np.dot(x, self.w1))

        return np.dot(x.reshape(self.input_dim, 1), q3.reshape(1, self.hidden_dim))

    

    def training(self, input, target):

        x = np.array(input).T

        y = self.feedforward(x)

        g = np.array(target).T



        self.w1 = self.w1 - self.lr * self.bprop_w1(g, y, x)    #### w1 먼저 update 해주어야 함!!!

        self.w2 = self.w2 - self.lr * self.bprop_w2(g, y)
#### Hyperparameter ####



input_dim = 784

hidden_dim = 300

output_dim = 10

epoch = 20



train_label = train['label']

train_pixel = train.drop(columns=['label'])



def input_norm(x):

    return x/255.0*0.99 + 0.01



train_targets = pd.get_dummies(train_label)

train_targets = train_targets.replace(0, 0.01).replace(1, 0.99)



train_pixel = train_pixel.applymap(input_norm)



train_targets.head(3)
train_pixel.sample(3)[['pixel200', 'pixel300', 'pixel400', 'pixel500', 'pixel600']]
pct = perceptron(input_dim, hidden_dim, output_dim, lr=0.01)



for k in range(epoch):

    print("{0}% training in process".format(k*100/epoch))

    for i, v in train_pixel.iterrows():

        input_ = v

        target_ = train_targets.iloc[i]

        pct.training(input_, target_)



        

test_pixel = test.applymap(input_norm)

        

prediction = list()

for i,v in test_pixel.iterrows():



    prediction_list = pct.feedforward(v)

    prediction.append(np.argmax(prediction_list))



prediction[:5]
sample_submission.head()
submission = sample_submission.drop(columns=['Label'])

submission['Label'] = prediction

submission.head()
submission.to_csv('sub1_0130.csv', index=False)
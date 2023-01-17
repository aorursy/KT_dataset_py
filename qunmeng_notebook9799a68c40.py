# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import tflearn

import tensorflow as tf

# Remove regular python warnings

import warnings

warnings.filterwarnings('ignore')

# Remove TensorFlow warnings

tf.logging.set_verbosity(tf.logging.ERROR)

# Visualizations

from IPython.display import display, Math, Latex

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train=data.ix[:,1:]

labels=data.ix[:,0:1]

data = pd.concat([train,test],ignore_index=True)
norm_data = (data - data.mean()) / (data.std())

norm_data = norm_data.fillna(0)
norm_labels=[]

for value in labels.iterrows():

    new_label=np.zeros(10)

    new_label[value[1]]=1

    norm_labels.append(new_label)

norm_labels=np.array(norm_labels)
train = norm_data.as_matrix()[0:42000]

test = norm_data.as_matrix()[42000:]
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, 784])
net = tflearn.fully_connected(net, 250, activation='ReLu')

# add a second hidden layer

net = tflearn.fully_connected(net, 100, activation='ReLu')
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(train, norm_labels,show_metric=True,validation_set=0.1,batch_size=100, n_epoch=50)
for i in range(3):

    ran=np.random.randint(0,test.shape[0])

    pred=model.predict(test)[ran]

    pred_digit=pred.index(max(pred))

    digit=test[ran].reshape(28,28)

    plt.imshow(digit, cmap='gray_r')

    plt.text(1, -1,"PREDICTION: {}".format(pred_digit),fontsize=20) 

    plt.show()
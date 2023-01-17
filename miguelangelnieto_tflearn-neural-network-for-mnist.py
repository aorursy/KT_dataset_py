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
print(train.shape)

print(test.shape)

print(data.shape)

print(labels.shape)
norm_data = (data - data.mean()) / (data.std())

norm_data = norm_data.fillna(0)
labels[0:5]
norm_labels=[]

for value in labels.iterrows():

    new_label=np.zeros(10)

    new_label[value[1]]=1

    norm_labels.append(new_label)

norm_labels=np.array(norm_labels)
print(labels.ix[12:12,0:1])

print(norm_labels[12])
train = norm_data.as_matrix()[0:42000]

test = norm_data.as_matrix()[42000:]
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, 784])
x=np.arange(-10,10,1)

y=np.maximum(x, 0)

plt.plot(x,y)

plt.xlim(-10,10)

plt.show()
net = tflearn.fully_connected(net, 128, activation='ReLu')

# add a second hidden layer

net = tflearn.fully_connected(net, 64, activation='ReLu')

# third layer, better going deeper than wider

net = tflearn.fully_connected(net, 32, activation='ReLu')
i=np.array([1,2,3,4,1,2,3,7])
o=np.exp(i)/np.sum(np.exp(i))

o
int(np.sum(o))
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
ids=[]

predictions=[]

pred=model.predict(test)

for i, values in enumerate(pred):

    pred_digit=values.index(max(values))

    ids.append(i+1)

    predictions.append(pred_digit)

    

# Make predictions



sub = pd.DataFrame({

        "ImageId": ids,

        "Label": predictions

    })



sub.to_csv("digit_submission.csv", index=False)
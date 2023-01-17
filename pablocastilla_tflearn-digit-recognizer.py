import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tflearn

import tensorflow as tf

from subprocess import check_output

from keras.utils.np_utils import to_categorical
df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



# Split data into training set and validation set

y_train = df.ix[:,0].values

x_train = df.ix[:,1:].values



#One Hot encoding of labels.

y_train_one_hot = to_categorical(y_train)
print(x_train.shape, y_train_one_hot.shape,df_test.shape)
tf.reset_default_graph()



net = tflearn.input_data([None, 784])



# Hidden layer(s)

net = tflearn.fully_connected(net, 128, activation='ReLU')

net = tflearn.fully_connected(net, 64, activation='ReLU')

net = tflearn.fully_connected(net, 32, activation='ReLU')



# Output layer and training model

net = tflearn.fully_connected(net, 10, activation='softmax')

net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')



model = tflearn.DNN(net)
model.fit(x_train, y_train_one_hot, validation_set=0.1, show_metric=True, batch_size=300, n_epoch=50)
def prediction(predictions):

    return np.argmax(predictions,1)



predictions = prediction(model.predict(df_test))

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})



submissions.to_csv("submission.csv", index=False, header=True)
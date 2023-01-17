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



# import warnings

# warnings.filterwarnings("ignore")

import csv



import tflearn

import tensorflow as tf

from keras.utils.np_utils import to_categorical
df_trn = pd.read_csv("../input/train.csv")

df_tst = pd.read_csv("../input/test.csv")



x_trn = df_trn.ix[:,1:].values

y_trn = df_trn.ix[:,0].values

y_trn_cat = to_categorical(y_trn)
tf.reset_default_graph()



net = tflearn.input_data([None, 784])



net = tflearn.fully_connected(net, 200, activation='ReLU')

net = tflearn.fully_connected(net, 100, activation='ReLU')

net = tflearn.fully_connected(net,  50, activation='ReLU')



net = tflearn.fully_connected(net, 10, activation='softmax')

net = tflearn.regression(net, optimizer='sgd', learning_rate=0.07, loss='categorical_crossentropy')



model = tflearn.DNN(net)
model.fit(x_trn, y_trn_cat, validation_set=0, show_metric=True, batch_size=500, n_epoch=120)
np.argmax(model.predict(df_tst),1)[0:100]
def prediction(predictions):

    return np.argmax(predictions,1)



predictions = prediction(model.predict(df_tst))

submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})



submissions.to_csv("tests.csv", index=False, header=True)
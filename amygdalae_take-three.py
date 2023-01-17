# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import pickle, gzip

print(tf.__version__)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) 

# will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#pkl = gzip.open('../input/mnist-for-tf/mnist.pkl.gz','rb')

#data = pickle.load(pkl, encoding='latin1')

#len(data)
#(trainX, trainY), (testX, testY), (validX, validY) = data

#print(trainX.shape)

#print(testX.shape)

#print(validX.shape)
data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"

data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)



df_train.head()
train_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"pixels": df_train[df_train.columns[1:]].values/255},

        y=df_train["label"],

        batch_size=100,

        num_epochs=3,

        shuffle=True)

feature_columns = [tf.feature_column.numeric_column("pixels", shape=784)]
classifier = tf.estimator.LinearClassifier(

                feature_columns=feature_columns, 

                n_classes=10,

                model_dir="./models/linear1"

                )
classifier.train(input_fn=train_input_fn, steps=1000)
print(check/_output(["ls", "./models/linear1"]).decode("utf8"))
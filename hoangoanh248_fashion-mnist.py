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
!python -V
data_train_file = "../input/fashion-mnist_train.csv"

data_test_file = "../input/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)
df_train.head()
# Select all columns but the first

features = df_train[df_train.columns[1:]]

features.head()
labels = df_train['label']

labels.describe(include='all')
features.values.shape
df_train.describe()
import tensorflow as tf
train_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"pixels": features.values/255},

        y=labels,

        batch_size=100,

        num_epochs=3,

        shuffle=True)
feature_columns = [tf.feature_column.numeric_column("pixels", shape=784)]
classifier = tf.estimator.LinearClassifier(

                feature_columns=feature_columns, 

                n_classes=10,

                model_dir="./models/linear1"

                )
classifier.train(input_fn=train_input_fn)
features = df_test[df_test.columns[1:]]

labels = df_test["label"]



evaluate_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"pixels": features.values/255},

        y=labels,

        batch_size=100,

        num_epochs=1,

        shuffle=False)
classifier.evaluate(input_fn=evaluate_input_fn)["accuracy"]
# Bonus round 1: predictions

features = df_test[df_test.columns[1:]]

labels = df_test["label"]



predict_input_fn = tf.estimator.inputs.numpy_input_fn(        

        x={'pixels': features.iloc[5000:5005].values/255},

        batch_size=1,

        num_epochs=1,

        shuffle=False)

predictions = classifier.predict(input_fn=predict_input_fn)



for prediction in predictions:

    print("Predictions:    {} with probabilities {}\n".format(

        prediction["classes"], prediction["probabilities"]))

print('Expected answers values: \n{}'.format(

    labels.iloc[5000:5005]))
# Bonus round 2: visualizing our predictions

# Import, setup, and a utility for int->string class conversion

import matplotlib.pyplot as plt

%matplotlib inline

class_table = [

    "T-shirt/top",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandal",

    "Shirt",

    "Sneaker",

    "Bag",

    "Ankle boot"

]



def get_label_cls(label):

    """given an int label range [0,9], return the string description of that label"""

    return class_table[label]



get_label_cls(3)

for i in range(5000,5005): 

    sample = np.reshape(df_test[df_test.columns[1:]].iloc[i].values/255, (28,28))

    plt.figure()

    plt.title("labeled class {}".format(get_label_cls(df_test["label"].iloc[i])))

    plt.imshow(sample, 'gray')
DNN = tf.estimator.DNNClassifier(

                feature_columns=feature_columns, 

                hidden_units=[40,30,20],

                n_classes=10,

                model_dir="./models/deep1"

                )
DNN.train(input_fn=train_input_fn)
DNN.evaluate(input_fn=evaluate_input_fn)["accuracy"]
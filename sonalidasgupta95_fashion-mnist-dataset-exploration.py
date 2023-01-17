# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


data_train=pd.read_csv("../input/fashion-mnist_train.csv")
data_test=pd.read_csv("../input/fashion-mnist_test.csv")
print ("train size: ",data_train.size, "test data size: ",data_test.size)

import matplotlib.pyplot as plt
%matplotlib inline
for i in range(5000,5005):
    sample=np.reshape(data_test[data_test.columns[1:]].iloc[i].values/255,(28,28))
    plt.figure()
    plt.title("Labeled class : {}".format(data_test["label"].iloc[i]))
    plt.imshow(sample, 'gray')

features=data_train[data_train.columns[1:]] 
labels=data_train['label']
import tensorflow as tf
input_fn=tf.estimator.inputs.numpy_input_fn(
x={"pixels":features.values/255},
y=labels,
batch_size=100,
num_epochs=3,
shuffle=True) 

feature_columns=[tf.feature_column.numeric_column("pixels",shape=784)]

#LinearClassifier
classifier=tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                        n_classes=10,
                                        model_dir="./models/linear1")

classifier.train(input_fn=input_fn)

test_features=data_test[data_test.columns[1:]]
labels=data_test["label"]
evaluate_input_fn=tf.estimator.inputs.numpy_input_fn(x={"pixels": test_features.values/255},
                                                      y= labels,
                                                      batch_size=100,
                                                    num_epochs=1,
                                                   shuffle=False)
accuracy=classifier.evaluate(input_fn=evaluate_input_fn)["accuracy"]
print("accuracy: ",accuracy)

#prediction
features=data_test[data_test.columns[1:]]
labels=data_test["label"]
predict_fn=tf.estimator.inputs.numpy_input_fn(x={"pixels":features.iloc[5000:5005].values/255},
                                             batch_size=1,
                                              num_epochs=1,
                                              shuffle=False
                                             )
predictions=classifier.predict(input_fn=predict_fn)

for prediction in predictions:
    print("Predictions: {} with probabilities {}\n".format(prediction["classes"],prediction["probabilities"]))
print("Expected labels: {}".format(labels.iloc[5000:5005]))

#visualizing
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





def get_label_class(label):
    return class_table[label]

get_label_class(0)

import matplotlib.pyplot as plt
%matplotlib inline
data_test=pd.read_csv("../input/fashion-mnist_test.csv")
for i in range(5000,5005): 
    sample = np.reshape(data_test[data_test.columns[1:]].iloc[i].values/255, (28,28))
    plt.figure()
    plt.title("labeled class {}".format(get_label_class(data_test["label"].iloc[i])))
    plt.imshow(sample, 'gray')
    
    
#DNN Classifier
DNN=tf.estimator.DNNClassifier(
feature_columns=[tf.feature_column.numeric_column("pixels",shape=784)],
hidden_units=[40,30,20],
n_classes=10,
model_dir="./models/deep1")
train_input_fn=tf.estimator.inputs.numpy_input_fn(x={"pixels": data_train[data_train.columns[1:]].values/255
                                                    },
                                                 y=data_train["label"],
                                                 batch_size=100,
                                                 num_epochs=3,
                                                 shuffle=True)
DNN.train(input_fn=train_input_fn)
accuracy=DNN.evaluate(input_fn=evaluate_input_fn)["accuracy"]
print("accuarcy of DNN: ",accuracy)


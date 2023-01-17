import numpy as np 

import pandas as pd 

import os



import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns
# !pip install tensorflow==2.0.0-alpha0
print("TensorFlow version: {}".format(tf.__version__))

print("Eager execution: {}".format(tf.executing_eagerly()))
## Downloading Dataset



train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"



train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),

                                           origin=train_dataset_url)



print("Local copy of the dataset file: {}".format(train_dataset_fp))
data = pd.read_csv(train_dataset_fp, names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], skiprows=1)

data.head()
## Correlation Matrix

corrMatt = data[data.columns].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False



fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True);
from sklearn.model_selection import train_test_split



X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=data["species"], random_state=100)
## Define Model

class IrisClasifier(tf.keras.Model):

    def __init__(self):

        super(IrisClasifier, self).__init__()

        self.layer1 = tf.keras.layers.Dense(10, activation='relu')

        self.layer2 = tf.keras.layers.Dense(10, activation='relu')

        self.outputLayer = tf.keras.layers.Dense(3, activation='softmax')

        

    def call(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        return self.outputLayer(x)
model = IrisClasifier()

model.compile(optimizer=tf.keras.optimizers.Adam(),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=300, batch_size=10, verbose=1);
# Evaluation

scores = model.evaluate(X_test, y_test)

print("\nAccuracy: %.2f%%" % (scores[1]*100))
## Get predictions

predictions = model.predict(X_test)

prediction1 = pd.DataFrame({"IRIS1":predictions[:,0], "IRIS2":predictions[:,1], "IRIS3":predictions[:,2],})

prediction1.round(decimals=4).head()
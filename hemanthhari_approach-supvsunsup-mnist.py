# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn import datasets

from sklearn import manifold

%matplotlib inline
data = datasets.fetch_openml('mnist_784',version=1,return_X_y=True)

pixel_values,targets= data

targets= targets.astype(int)
single_image= pixel_values[1,:].reshape(28,28)

plt.imshow(single_image,cmap='gray')
tsne = manifold.TSNE(n_components=2,random_state=42)

transformed_data = tsne.fit_transform(pixel_values[:3000,:])
tsne_df = pd.DataFrame(np.column_stack((transformed_data,targets[:3000])),columns=["x","y","targets"])

tsne_df.loc[:,"targets"]= tsne_df.targets.astype(int)
tsne_df.head(10)
grid= sns.FacetGrid(tsne_df,hue="targets",size=8)

grid.map(plt.scatter,"x","y").add_legend()
import tensorflow as tf # Import tensorflow library

import matplotlib.pyplot as plt # Import matplotlib librar

mnist = tf.keras.datasets.mnist # Object of the MNIST dataset

(x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data
plt.imshow(x_train[0], cmap="gray") # Import the image

plt.show() # Plot the image
# Normalize the train dataset

x_train = tf.keras.utils.normalize(x_train, axis=1)

# Normalize the test dataset

x_test = tf.keras.utils.normalize(x_test, axis=1)
#Build the model object

model = tf.keras.models.Sequential()

# Add the Flatten Layer

model.add(tf.keras.layers.Flatten())

# Build the input and the hidden layers

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Build the output layer

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# Compile the model

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x=x_train, y=y_train, epochs=55) # Start training process
# Evaluate the model performance

test_loss, test_acc = model.evaluate(x=x_test, y=y_test)

# Print out the model accuracy 

print('\nTest accuracy:', test_acc)
predictions = model.predict([x_test]) # Make prediction
print(np.argmax(predictions[1000]))
plt.imshow(x_test[1000], cmap="gray") # Import the image

plt.show() # Show the image
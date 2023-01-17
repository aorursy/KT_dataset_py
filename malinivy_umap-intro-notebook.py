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
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images / 255.0
test_images=test_images / 255.0
training_labels[0:10]
print(training_images.shape)
print(test_images.shape)
data = np.array(np.vstack([training_images]), dtype=np.float64) / 255.0
print(data.shape)
target = np.hstack([training_labels])
new_data = data.reshape( data.shape[0],(data.shape[1]*data.shape[2]),)
print(new_data.shape)
new_data[59999,0:10]
import umap
embedding = umap.UMAP(n_components=2,
                     random_state=42).fit_transform(new_data)
embedding.shape
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))
plt.scatter(embedding[:60000, 0], embedding[:60000, 1], 
            c=training_labels, 
            cmap="Spectral", 
            s=10)
plt.axis('off');
from tensorflow.keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
print(trainX.shape)
Train_data = trainX.reshape( trainX.shape[0],(trainX.shape[1]*trainX.shape[2]),)
print(Train_data.shape)
mapper = umap.UMAP().fit_transform(Train_data)

plt.figure(figsize=(12,12))
plt.scatter(mapper[:60000, 0], mapper[:60000, 1], 
            c=trainY, 
            cmap="Spectral", 
            s=10)

embedding_5 = umap.UMAP(n_neighbors=5).fit_transform(Train_data)
plt.figure(figsize=(12,12))
plt.scatter(embedding_5[:60000, 0], embedding_5[:60000, 1], 
            c=trainY, 
            cmap="Spectral", 
            s=10)
plt.axis('off');
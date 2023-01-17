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
import tensorflow as tf

from sklearn.cluster import KMeans

IMG_DIM = 28
CLASSES = 10
LABELS_TO_NAMES = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}
input_ds = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
plt.style.use('grayscale')
plt.figure(figsize=(25, 25))
rows = 6
start = 0
for i in range(rows * rows):
    plt.subplot(rows, rows, i + 1)
    plt.imshow(input_ds.iloc[start + i, 1:].to_numpy().reshape(IMG_DIM, IMG_DIM))
    plt.title(LABELS_TO_NAMES[input_ds.iloc[start + i, 0]])
pixels_for_cl = []

def compute_average_pixels_for_class(cl):
    pixels = []
    ds_for_cl = input_ds[input_ds.label == cl]
    for i in range(1, IMG_DIM * IMG_DIM + 1):
        pixels.append(int(ds_for_cl["pixel%d" % i].mean()))
    return pixels

        
for cl in range(CLASSES):
    pixels_for_cl.append(compute_average_pixels_for_class(cl))
    
plt.figure(figsize=(15, 15))
rows = 4
for i in range(len(pixels_for_cl)):
    pixels = pixels_for_cl[i]
    plt.subplot(rows, rows, i + 1)
    plt.imshow(np.array(pixels).reshape((IMG_DIM, IMG_DIM)))
    plt.title(LABELS_TO_NAMES[i])
pixels_for_cl = [np.array(p).astype(float) for p in pixels_for_cl]

def predict(sample):
    predicted = 0
    score = 1000000000
    
    scores = []
    for cl, pixels in enumerate(pixels_for_cl):
        val = 0
        loss = tf.keras.losses.MeanAbsoluteError()
        val = loss(pixels, sample.to_numpy().astype(float)).numpy()
        scores.append(val)
        if val < score:
            score = val
            predicted = cl
    return predicted

predictions = []
correct_predictions = 0

for sample_num, row in input_ds.iterrows():
    label = row[0]
    sample = row.iloc[1:]

    predicted = predict(sample)
    predictions.append(predicted)
    if label == predicted:
        correct_predictions += 1

print("%d correct predictions out of %d" % (correct_predictions, len(input_ds)))
confusion_matrix = np.zeros((CLASSES, CLASSES), dtype=int)

input_ds.iloc[0, 0]
for i in range(len(input_ds)): 
    confusion_matrix[predictions[i], input_ds.iloc[i, 0]] += 1
    confusion_matrix[input_ds.iloc[i, 0], predictions[i]] += 1
import seaborn as sns
sns.heatmap(confusion_matrix)
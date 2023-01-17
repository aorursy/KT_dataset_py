import umap

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

plt.figure(figsize=(20,10))
kannada_train = pd.read_csv("../input/Kannada-MNIST/train.csv")
kannada_train_sampled = kannada_train.sample(n = 10000, random_state = 123)
kannada_train_sampled.groupby('label')['label'].count()


image = np.array(kannada_train_sampled.iloc[2,1:785], dtype='float')

pixels = image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.title(kannada_train_sampled.iloc[2,0])

plt.show()
data = kannada_train_sampled.iloc[:, 1:].values.astype(np.float32)

target = kannada_train_sampled['label'].values
%time reduce = umap.UMAP(random_state = 123) #just for reproducibility

%time  embedding = reduce.fit_transform(data)
df = pd.DataFrame(embedding, columns=('x', 'y'))

df["class"] = target

plt.scatter(df.x, df.y, s= 5, c=target, cmap='Spectral')

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))

plt.title('Kannada MNIST clustered (dimensions reduced - Visulization) with UMAP', fontsize=20);

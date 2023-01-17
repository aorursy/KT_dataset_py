import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_test = pd.read_csv(os.path.join('/kaggle/input/Kannada-MNIST/test.csv'))

df_test.sample(n=5)
df_train = pd.read_csv(os.path.join('/kaggle/input/Kannada-MNIST/train.csv'))

df_train.sample(n=5)
X_train = np.array(df_train.loc[:, df_train.columns != 'label'])

X_test  = np.array(df_test.loc[:, df_test.columns != 'id'])



y_train = np.array(df_train['label'])



print(f"X_train: {X_train.shape}\nX_test: {X_test.shape}\ny_train: {y_train.shape}")
rows = 5

cols = 10

fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))



for label in range(cols):

    digits = df_train.loc[df_train['label'] == label]

    digits = digits.drop('label', axis=1)

    ax[0][label].set_title(label)

    for j in range(rows):

        ax[j][label].axis('off')

        ax[j][label].imshow(digits.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')
sns.distplot(y_train, kde=False)
pixel_counts = (df_train.loc[:, df_train.columns != 'label'] / 255).astype(int)

pixel_counts = pixel_counts.sum(axis=0).values

pixel_counts = pixel_counts.reshape((28, 28))

sns.heatmap(pixel_counts)
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 5))

ax = ax.flatten()



for label in range(10):

    pixel_counts = (df_train.loc[:, df_train.columns != 'label'] / 255).astype(int)

    pixel_counts = pixel_counts.loc[df_train['label'] == label]

    pixel_counts = pixel_counts.sum(axis=0).values

    pixel_counts = pixel_counts.reshape((28, 28))

    ax[label].axis('off')

    sns.heatmap(pixel_counts, ax=ax[label])
import os

import random



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2
PATH_DATASET = "../input/lego-minifigures-classification"

PATH_INDEX = os.path.join(PATH_DATASET, "index.csv")

PATH_TEST = os.path.join(PATH_DATASET, "test.csv")

PATH_METADATA = os.path.join(PATH_DATASET, "metadata.csv")
df_index = pd.read_csv(PATH_INDEX)

df_index
df_test = pd.read_csv(PATH_TEST)

df_test
n_classes = df_index["class_id"].unique().shape[0]

print(f"Total classes: {n_classes}")
df_metadata = pd.read_csv(PATH_METADATA)

print(f"Number of rows {df_metadata.shape[0]}")

df_metadata.head()
print('Minifigure names: ' + ' ||| '.join(df_metadata['minifigure_name'].tolist()))
df_metadata[df_metadata['minifigure_name'] == 'SPIDER-MAN']
df_index = pd.merge(df_index, df_metadata[['class_id', 'minifigure_name']], on='class_id')

df_test = pd.merge(df_test, df_metadata[['class_id', 'minifigure_name']], on='class_id')

df_index
df_index['tmp_name'] =  df_index['class_id'].astype(str) + ' - ' + df_index['minifigure_name']

df_test['tmp_name'] =  df_test['class_id'].astype(str) + ' - ' + df_test['minifigure_name']



plt.figure(figsize=(16, 16))





plt.subplot(1, 2, 1)

sns.countplot(y="tmp_name", data=df_index)

plt.title("Train Classes Distibution", fontsize=18)

plt.xticks(fontsize=14)

plt.xlabel('number of samples', fontsize=16);

plt.yticks(fontsize=14)

plt.ylabel('class id', fontsize=16)

plt.legend(fontsize=15)



plt.subplot(1, 2, 2)

sns.countplot(y="class_id", data=df_test.sort_values('class_id'))

plt.title("Test Classes Distibution", fontsize=18)

plt.xticks(fontsize=14)

plt.xlabel('number of samples', fontsize=16);

plt.yticks(fontsize=14)

plt.ylabel('class id', fontsize=16)

plt.legend(fontsize=15);
plt.figure(figsize=(16, 10))

for ind, el in enumerate(df_index.sample(15).iterrows(), 1):

    plt.subplot(3, 5, ind)

    image = cv2.imread(os.path.join(PATH_DATASET, el[1]['path']))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)

    plt.title(f"{el[1]['class_id']}: {el[1]['minifigure_name']}")

    plt.xticks([])

    plt.yticks([])
plt.figure(figsize=(16, 10))

for ind, el in enumerate(df_test.sample(15).iterrows(), 1):

    plt.subplot(3, 5, ind)

    image = cv2.imread(os.path.join(PATH_DATASET, el[1]['path']))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)

    plt.title(f"{el[1]['class_id']}: {el[1]['minifigure_name']}")

    plt.xticks([])

    plt.yticks([])
plt.figure(figsize=(16, 5))

for ind, el in enumerate(df_index[df_index['minifigure_name'] == 'YODA'].sample(5).iterrows(), 1):

    plt.subplot(1, 5, ind)

    image = cv2.imread(os.path.join(PATH_DATASET, el[1]['path']))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)

    plt.title(f"{el[1]['class_id']}: {el[1]['minifigure_name']}")

    plt.xticks([])

    plt.yticks([])
plt.figure(figsize=(16, 5))

for ind, el in enumerate(df_test[df_test['minifigure_name'] == 'YODA'].iterrows(), 1):

    plt.subplot(1, 5, ind)

    image = cv2.imread(os.path.join(PATH_DATASET, el[1]['path']))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)

    plt.title(f"{el[1]['class_id']}: {el[1]['minifigure_name']}")

    plt.xticks([])

    plt.yticks([])
n_rows = df_index['class_id'].unique().shape[0]

n_cols = 5



for row_ind, group in enumerate(df_index.groupby('class_id')):

    plt.figure(figsize=(20, 5))

    for col_ind, el in enumerate(group[1].sample(n_cols).iterrows()):

        plt.subplot(1, n_cols, col_ind + 1)

        image = cv2.imread(os.path.join(PATH_DATASET, el[1]['path']))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)

        plt.xticks([])

        plt.yticks([])

    plt.show()
plt.figure(figsize=(64, 16))

plt.subplots_adjust(wspace=0, hspace=0)



paths = df_index['path'].tolist()

paths = random.sample(paths, 100)

for ind, path in enumerate(paths, 1):

    plt.subplot(5, 20, ind)

    image = cv2.imread(os.path.join(PATH_DATASET, path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (512, 512))

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])
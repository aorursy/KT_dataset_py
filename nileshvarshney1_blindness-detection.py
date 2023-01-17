import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns



import cv2

import glob

from zipfile import ZipFile
df_train = pd.read_csv("../input/train.csv")

print("No of Observation : {}".format(df_train.shape[0]))

plt.rcParams["figure.figsize"] = [16,5]

sns.countplot(df_train["diagnosis"])

plt.show()
folders = glob.glob("../input/train_images/*")

read_images = []        

#, cv2.IMREAD_GRAYSCALE

for image in folders[0:20]:

    read_images.append(cv2.imread(image,cv2.COLOR_BGR2RGB))
plt.figure(figsize=(24,24))

for index in range(len(read_images[0:16])):

    diag = folders[index].split('/')[-1].split('.')[0]

    tp = df_train[df_train['id_code']== diag]['diagnosis']

    ax = plt.subplot(4,4,index+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(read_images[index])

    ax.title.set_text('diagnosis = {}'.format(tp.to_string(index=False)))

plt.show()
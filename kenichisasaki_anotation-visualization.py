# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# -*- coding: utf-8 -*-

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.patches as mpatches

import seaborn as sns

import pandas as pd

import numpy as np

import japanize_matplotlib

import os



train_file = pd.read_csv(r"../input/kuzushiji-recognition/train.csv")



unicode_chart = pd.read_csv(r"../input/kuzushiji-recognition/unicode_translation.csv")

train_file.labels = train_file.labels.str.split(' ')

train_file.labels.head(3)
#changed the train label dimension

label_np = np.array(train_file.labels[0])

label_np = label_np.reshape(int(len(label_np)/5), 5)

label_list = np.array([])

for i in label_np[:, 0]:

    label_list= np.append(label_list, unicode_chart.char[unicode_chart.Unicode == i])

#label_list = pd.DataFrame(label_list, columns=["chart_index", "char"])

print(label_list)
for k in range(5):

    label_np = np.array(train_file.labels[k])

    # labeling skips when no characters in the image where the array value become NaN.

    if pd.isnull(label_np).all() == True:

        continue

    label_np = label_np.reshape(int(len(label_np)/5), 5)

    label_list = np.array([])

    for i in label_np[:, 0]:

        label_list= np.append(label_list, unicode_chart.char[unicode_chart.Unicode == i])

    

    image = cv2.imread(os.path.join(r"../input/kuzushiji-recognition/train_images", train_file.image_id[k] + ".jpg"))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   

    height, width, channels = img.shape



    fig =plt.figure(figsize=(width/200, height/200))

    ax = fig.add_subplot(1,1,1)

    j = 0

    for i in label_np:

        rect = mpatches.Rectangle((int(i[1]), int(i[2])), int(i[3]), int(i[4]), fill=False, edgecolor='red', linewidth=1)

        ax.add_patch(rect)





        ax.text((int(i[1]) + int(i[3]))/width, 1 - int(i[2])/height, label_list[j],

                horizontalalignment='left',fontsize=20,

                verticalalignment='center',

                rotation='horizontal',

                transform=ax.transAxes)

        j = j+1

    ax.imshow(img)

    #plt.savefig(os.path.join("./anotated", train_file.image_id[k]+"_anotated.jpg"))

    plt.show()

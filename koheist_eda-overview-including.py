%matplotlib inline

import os

import cv2

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

import glob

import numpy as np

import pandas as pd

import collections
train_df = pd.read_csv("../input/landmark-retrieval-2020/train.csv")

train_df
landmark_id = train_df.landmark_id.unique()

print("Number of landmark_id : ", len(landmark_id))
imagedata = [] #初期化

landmark_id_list = []



for ID_N in range(80):

    train_df_id = train_df.loc[train_df.landmark_id == landmark_id[ID_N]]

    train_df_id = train_df_id.reset_index(drop=True)

    landmark_id_list.append(landmark_id[ID_N])

    

    num1 = str(train_df_id.id[0])[0]

    num2 = str(train_df_id.id[0])[1]

    num3 = str(train_df_id.id[0])[2]

    filename = str(train_df_id.iloc[0, 0])

    filepath = "../input/landmark-retrieval-2020/train/" +num1+ "/" +num2+"/" +num3+ "/" + filename + ".jpg"

    imagedata.append(cv2.imread(filepath))
fig = plt.figure(figsize=(20,24))



for i in range(80):

    plt.subplot(8, 10, i+1)

    img = imagedata[i]

    plt.title(landmark_id_list[i])

    plt.grid(False)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.tick_params(labelbottom=False,

                    labelleft=False,

                    labelright=False,

                    labeltop=False)
ID_N = 0



train_df_id = train_df.loc[train_df.landmark_id == landmark_id[ID_N]]

train_df_id = train_df_id.reset_index(drop=True)



imagedata = []



for i in range(len(train_df_id)):

    num1 = str(train_df_id.id[i])[0]

    num2 = str(train_df_id.id[i])[1]

    num3 = str(train_df_id.id[i])[2]

    filename = str(train_df_id.iloc[i, 0])   



    filepath = "../input/landmark-retrieval-2020/train/" +num1+ "/" +num2+"/" +num3+ "/" + filename + ".jpg"

    imagedata.append(cv2.imread(filepath))



print("landmark_id =", landmark_id[ID_N])
fig = plt.figure(figsize=(16,8))



for i in range(12 if len(train_df_id) > 12 else (len(train_df_id))):

    plt.subplot(3, 4, i+1)

    img = imagedata[i]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
c = collections.Counter(train_df.landmark_id)

c = collections.Counter(list(c.values()))

c = sorted(c.items())

#sns.barplot(c)



x = []

y = []



for i in c:

    x.append(i[0])

    y.append(i[1])
sns.set()

sns.set_palette("winter_r", 8, 0)



fig = plt.figure(figsize=(24, 6))

ax = fig.add_subplot(1, 1, 1)

sns.barplot(x[:100],y[:100])

ax.set(xlabel ='Number of photos per landmark',ylabel='Number of samples' )

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

ax.legend()
sns.barplot(x[:19],y[:19])

plt.xlabel("Number of photos per landmark")

plt.ylabel("Number of samples")
total1 = 0

total2 = 0



for i in c:

    if int(i[0]) < 7:

        total1 = total1 + int(i[1])

    else:

        total2 = total2 + int(i[1])



print("写真が7枚未満の合計:", total1, "写真が7枚以上の合計:", total2)
pd.DataFrame(c).describe()[1]
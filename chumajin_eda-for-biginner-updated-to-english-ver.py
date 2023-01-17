import numpy as np 

import pandas as pd 
all_path = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        all_path.append(os.path.join(dirname, filename))

all_path[:10]
sample_path = '/kaggle/input/landmark-recognition-2020/sample_submission.csv'

traincsv_path = '/kaggle/input/landmark-recognition-2020/train.csv'
# sample submissionの中身を見る # View contents of sample submission 

sample = pd.read_csv(sample_path)

sample
# traincsvの中身を見る # View contents of train.csv

traincsv = pd.read_csv(traincsv_path)

traincsv
# traincsvの中身はidとlandmark_idだということがわかる。 # I could understand that the contents of traincsv are composed of id and landmark_id.

# まずは、idごとにpathをくっつけることを考える。 # Firstly, I considered attaching the path for each id.

# all_pathからtrainが入っているものと、jpgが入っているものを抜く # I tried to extract the one that contained the "train" and ".jpg" from the all_path.
all_path[:10]
# trainフォルダの中の.jpgファイルだけ抜きたい # I wanted to extract only ".jpg" file in train directory.
# trainがつくやつだけ抜く # Firstly, I extracted only "train" in all path.

train_impath = [s for s in all_path if "train" in s]
# <わからない人は・・・>↑は以下と同じ。 # If you cannot understand, the above code is the same as below.



train_impath2 = []



for s in all_path:

    if "train" in s:

        train_impath2.append(s)
# 同様にさらにjpgファイルだけ抜く # After that, I extracted ".jpg" in train_impath.

train_impath = [s for s in train_impath if ".jpg" in s]

train_impath[:10]
# ここから.jpgの前のidのみを抜きたい  # From these pathes, I extracted id from path in each, in order to merge with train.csv.

train_id = [s.split("/")[-1] for s in train_impath]

train_id = [s.split(".")[0] for s in train_id]

train_id[:10]
# dataframe化 # Making dataframe.

df = pd.DataFrame()

df["path"] = train_impath

df["id"] = train_id
df.head(3)
traincsv.head(3)
# mergeしてidが同じところをくっつける # Merge with the same id.

traindf = pd.merge(traincsv,df,on="id",how="left")

traindf.head(3)
import cv2

import matplotlib.pyplot as plt



example = traindf[traindf["landmark_id"]==1]

for a in example["path"]:

    plt.figure()

    img = cv2.imread(a)

    plt.imshow(img)

    plt.axis("off")
traindf.head(3)
traindf.to_csv("traindf.csv",index=False)
import collections

l = np.array(traindf["landmark_id"])

c = collections.Counter(l) # get unique counts
#print(c) # 辞書型で、それぞれのidに対する個数が出ている # c is the type of dictionaly and composed of the id and the number of each id.
# dataframe化したいので、リスト化する # List the c in order to make the dataframe.

key = list(c.keys())

cnt = list(c.values())
dfcnt = pd.DataFrame()

dfcnt["id"] = key

dfcnt["count"] = cnt

dfcnt.head(3)
# 並び替え # sort_values

dfcnt = dfcnt.sort_values("count")

dfcnt = dfcnt.reset_index(drop=True)
dfcnt.tail(3)
dfcnt.to_csv("dfcnt.csv",index=False)
plt.scatter(dfcnt["id"],dfcnt["count"])

plt.xlabel("id",fontsize = 15)

plt.ylabel("count",fontsize = 15)
import cv2

import matplotlib.pyplot as plt



example = traindf[traindf["landmark_id"]==dfcnt["id"].iloc[-1]]





for a in range(10):

    plt.figure()

    img = cv2.imread(example["path"].iloc[a])

    plt.imshow(img)

    plt.axis("off")
# 黒い枠ぶちが最も多い画像 # I can understand the images have black borders.
# all_pathから抜く # extract the .jpg file path from all_path

test_impath = [s for s in all_path if "test" in s]

test_impath = [s for s in test_impath if ".jpg" in s]

test_impath[:10]
for a in range(10):

    plt.figure()

    img = cv2.imread(test_impath[a])

    plt.imshow(img)

    plt.axis("off")
# このテスト画像に訓練画像のidを関連付けてconfidenceを出すのが課題。 # The challenge is to associate the id of the training image with this test image to give confidence.
sample
dfcnt.tail(3)
dfcnt["id"].iloc[-1]
test_landmarks = str(dfcnt["id"].iloc[-1]) + " " + str(1.0)

test_landmarks
sample.head(3)
sample["landmarks"] = test_landmarks

sample.head(3)


sample.to_csv("submission.csv",index=False)
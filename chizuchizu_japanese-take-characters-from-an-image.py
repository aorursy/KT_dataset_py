from PIL import Image, ImageDraw, ImageFont

from os import listdir

from glob import glob

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/unicode_translation.csv').values}
fontsize = 50

# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip



font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
df_train.head()

"""

Image_id

一般的なID列



labels

[ユニコード、　x, y, weight, high]

xとyは始点

weightとhighは枠の大きさを表している

つまり文字を抽出したかったら

[x, y, x+weight, y+high]

の長方形を抜き取れば良い

"""



"""

提出するもの

csv形式

id列とlabels列

id列は画像のIDを入れればOK



labels列

[ユニコード、ｘ座標、ｙ座標]

ここの座標は真ん中で良いそう

"""



"""

となると、



前処理

1.文字と座標を抜き取って単純な画像分類モデルのトレーニングデータのようなものを作成する



モデル作成

2.１で作成したデータを学習させる

3.OCRみたいなもので画像から文字を検出するモデルも作成（古代のかなだから学習済みモデルがあるかはわからない）



予測

4.テストデータから文字を取り出す

5.2で作った画像分類モデルに4の画像データを学習させる



submit

6.データの整形＆submit



という流れができるのかな（間違っていたら教えてください）



とはいえ4781もラベルがあるデータとなると多少の工夫は必要そう

それに実行時間も相当長くなりそうだ......

32x32の画像を１０万枚学習させたとして、どれくらいの時間がかかるのだろうか（やってみなくちゃわからない）

"""
len(unicode_map)
# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an image containing the bounding boxes and characters annotated



"""

アレンジしました

"""

def visualize_training_data(image_fn, labels):

    # Convert annotation string to array



    labels = np.array(str(labels).split(' ')).reshape(-1, 5)

    # print("labelsです", labels)

    # print("label.shape です", labels.shape, len(labels))

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')



    memo2 = np.array(labels)[:, 0]  # ラベルが入る

    memo3 = np.zeros((len(labels), 32, 32, 4)).astype("int32")  # Imageデータが入る

    # print(memo3.shape)

    """

    weight, highを見ていると64くらいでちょうどいい？感じがするので64にしました。

    """

    # print(memo3)

    # print(labels.shape)



    for i, (codepoint, x, y, w, h) in enumerate(labels):

        # print(i)

        x, y, w, h = int(x), int(y), int(w), int(h)

        """

        ここから画像の切り抜きを行う

        """

        # print(np.asarray(img_crop).shape)

        

        # 画像から文字の部分を抜き取る

        img_crop = imsource.crop((x, y, x+w, y+h))

        

        img_crop = np.asarray(img_crop.resize((32, 32)))

        

        # print(np.array(img_crop).shape)

        """

        print(np.array(img_crop).shape)

        

        # (64, 64, 4)

        """

        # arrayに入れる

        memo3[i, :, :, :] = np.asarray(img_crop).astype("int32")

    

    # ラベルと取り出した文字のデータもreturn

    return memo2, memo3
from glob import glob

from tqdm import tqdm

import gc

import matplotlib.font_manager as fm



prop = fm.FontProperties(fname="./NotoSansCJKjp-Regular.otf")

"""

適当にコメント付けます

"""

total = 800

i = 0

for x in glob("../input/train_images/*.jpg"):

    # print(x)

    # print(df_train.head())

    # path = x.split("/")[3]

    # local

    path = x.split("/")[3]

    path = path.split(".")[0]



    memo1 = df_train.values[df_train["image_id"] == path].flatten()

    # print(memo1)

    # print(memo1[1])



    try:

        """

        1ページ全てイラストだった場合は飛ばす。（nanになるためValueError)

        """

        labels = memo1[1]

        # print(labels)

        

        memo2, memo3 = visualize_training_data(x, labels)



        # print(memo3.shape)

        

        # save

        # np.save("npyy/" + str(i) + "image.npy", memo3)

        # np.save("npyy/" + str(i) + "labels.npy", memo2)



        if i == 0:

            image_data = memo3.copy()

            labels_data = memo2.copy()

        elif i % 400 == 0:

            """

            400枚ごとに1つのファイルを作成します。

            """

            np.save("image_data_" + str(i) + ".npy", image_data)

            np.save("labels_data_" + str(i) + ".npy", labels_data)

            del image_data, labels_data

            gc.collect()

            image_data = memo3.copy()

            labels_data = memo2.copy()

        else:

            image_data = np.append(image_data, memo3, axis=0)

            labels_data = np.append(labels_data, memo2, axis=0)

        # イラストページでカウントしないように

        i += 1

        # i.update(10)

    except ValueError:

        print("イラストページです！！")

    

    if i % 50 == 0:

        """

        取り出した画像を表示します

        """

        plt.rcParams["font.size"] = 15



        char = memo2[10]

        plt.title(unicode_map[char], fontproperties=prop, fontsize=50)

        plt.imshow(memo3[10, :, :, :])

        plt.show()

        

    if i == total:

        np.save("image_data_" + str(i) +".npy", image_data)

        np.save("labels_data_" + str(i) + ".npy", labels_data)

        break
"""

カーネルのDiskの制限で3800枚全ての画像の保存は出来ませんでした。（メモリエラーかもしれない?）



解決策も何かあればコメントください。



僕はあとでローカル環境で実行してみようと思っています。

"""
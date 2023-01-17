import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print("利用可能データの一覧")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#csvファイル読み込み

train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

#これでも同じ(csvファイル読み込み)

#train_df = pd.read_csv('../input/digit-recognizer/train.csv')



# 基本情報の表示

train_df.info()

# 冒頭のデータの表示

train_df.head()
# ラベルの種類の確認



#'label'列のユニークな要素を抽出

label_classes = train_df['label'].unique()



# ユニークな要素を表示

print("labels:")

print(label_classes)



# ソートして出力

print("sorted_labels:")

print(np.sort(label_classes))



# クラスの数の確認

print("Num of classes:")

print(len(label_classes))
#csvファイル読み込み

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



# 基本情報の表示

test_df.info()

# 冒頭のデータの表示

test_df.head()
#csvファイル読み込み

spsb_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')



# 基本情報の表示

spsb_df.info()

# 冒頭のデータの表示

spsb_df.head()
# データ数などを変数で管理するために代入



# trainデータの数 : 42,000

n_train = len(train_df) # 行数



# pixel数 : 748

n_pixels = len(train_df.columns) - 1 # xx_df.columns は列数。-1なのはラベル分を引いている。



# labelの種類の数 : 10

n_class = len(set(train_df['label']))



# trainデータの画像のpixel部分

train_df_p = train_df.drop(columns='label')



# trainデータのラベル部分

train_df_l = train_df['label']
# PyTorchの、画像用のライブラリをimportします

import torch

from torchvision.utils import make_grid



# Jupyter Notebook上でグラフを表示するためのライブラリ及びコマンド

import matplotlib.pyplot as plt

%matplotlib inline
#適当な行番号nを指定

n = 0



a_image_df = train_df_p.iloc[n,:]

a_label = train_df_l[n]



# numpyのarrayにする： a_image_df.values 

# arrayを28 * 28 の行列にreshape: .reshape(28,28)

# 各要素を255で割って0～1の値にする。

a_image_array = a_image_df.values.reshape(28,28) / 255. 



#print(a_image_array.shape)

#print(a_image_array)



plt.rcParams['figure.figsize'] = (3, 3) # 画像サイズ

plt.axis('off') # 軸表示off

plt.gray() # 白黒表示(そのままでは変な色になる)

plt.imshow(a_image_array) # 画像データの追加。

plt.show() # 画像の表示

print("label: "+str(a_label))
label_0_index = np.where(train_df_l == 0)

print(label_0_index)
label_0_index = np.where(train_df_l == 0)[0]

print(label_0_index)
label_index = [np.where(train_df_l == x)[0] for x in range(10)]

print("label == 0")

print(label_index[0])

print("label == 1")

print(label_index[1])
n_each_class = [len(x) for x in label_index]

print(n_each_class)



x_axis = np.arange(n_class)

print(x_axis)



plt.rcParams['figure.figsize'] = (5, 3)

plt.bar(x_axis, n_each_class)

plt.xticks(x_axis)

plt.xlabel('Class', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
import seaborn as sns



graph_result = sns.countplot(train_df_l)
n = 5 # 数字を表示したい行数



#各数字毎のindexの配列から、n個ずつindexを取ってきます。

zero2nine_2d = [[x[i] for x in label_index] for i in range(n)]



#このままだと2次元配列になってしまうので、numpyの機能を使ってflatten(一次元配列化)します。

zero2nine = np.array(zero2nine_2d).flatten()



# 該当するindexの行をtrain_df_pから取ってくる。

# .values : 値を取り出してarrayにする

# / 255. : 255.で割って0.0～1.0にする。

# .reshape(-1, 28, 28 ) : -1のところは入力に合わせて変わる。残りの28*28ずつn*10個の3次元配列にする

all_images = (train_df_p.iloc[zero2nine,:].values / 255.).reshape(-1, 28, 28)



#print("データの個数(n*10), 28(画素行),28(画素列)")

print(all_images.shape)



grid = make_grid(torch.Tensor(all_images).unsqueeze(1), nrow=10)

plt.rcParams['figure.figsize'] = (10, n)

plt.imshow(grid.numpy().transpose((1,2,0)))

plt.axis('off')

plt.show()
print(all_images.shape)

print(torch.Tensor(all_images).size())

print(torch.Tensor(all_images).unsqueeze(0).size())

print(torch.Tensor(all_images).unsqueeze(1).size())

print(torch.Tensor(all_images).unsqueeze(2).size())

print(torch.Tensor(all_images).unsqueeze(3).size())

print("---------")

print(grid.size())

print(grid.numpy().shape)

print(grid.numpy().transpose((1,2,0)).shape)
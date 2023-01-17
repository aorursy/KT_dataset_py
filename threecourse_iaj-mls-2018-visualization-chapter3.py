import numpy as np

# 1次元配列の作成
ary = np.arange(10)
print(ary)

# 2次元配列の作成
# 1次元配列を作った後、reshape関数で変形する
ary = np.arange(6).reshape(2, 3)
print(ary)

# Pythonのリストから1次元配列の作成
ary = np.array([1, 2, 3, 4, 5])
print(ary)

# Pythonのリストから2次元配列の作成
ary = np.array([[1, 2, 3], [4, 5, 6]])
print(ary)

# 配列の要素数を出力
print(ary.shape)

# 配列の形はshapeで確認する（変更するときはreshape）
ary1 = np.array([1,2,3,4,5,6])
ary2 = np.array([[1,2,3,4,5,6]])
ary3 = np.array([[1],[2],[3],[4],[5],[6]])
print(ary1, ary1.shape) # ベクトル
print(ary2, ary2.shape) # 1*6の行列
print(ary3, ary3.shape) # 6*1の行列

# 配列の型を出力
print(ary.dtype)
ary = np.arange(6).reshape(2, 3)
print(ary)

# スカラー値を加えると、要素ごとに加えられる
ary += 1
print(ary)

# 乗算なども同様に、要素ごとに演算が行われる
ary *= 10
print(ary)

# 論理演算
print(ary == 10)
print(ary > 50)

# 対数などの数学計算
print(np.log(ary))
ary = np.arange(6).reshape(2, 3)
print(ary)

# 和、縦方向の和、横方向の和
print(ary.sum(), ary.sum(axis=0), ary.sum(axis=1))

# 最大値、縦方向の最大値、横方向の最大値
print(ary.max(), ary.max(axis=0), ary.max(axis=1))
# （参考）ブロードキャスト - 演算を行う際に配列の形状を調整する
ary1 = np.arange(3) * 10
ary2 = np.arange(6).reshape(2, 3)
print(ary1)
print(ary2)

# ary1の形状が拡張されて加算されている
ary2 += ary1
print(ary2)
# 1次元配列での値の取得 
ary = np.arange(6)
print(ary)

# 値の取得
print(ary[0], ary[2], ary[4])

# スライシング（範囲での取得）
print(ary[2:4], ary[2:], ary[:4])
# 2次元配列での値の取得 
ary = np.arange(12).reshape(3, 4)
print(ary)

# 値の取得
print(ary[0, 0], ary[1, 1], ary[2, 2])

# スライシング（範囲での取得）

# 1次元配列で取得
print(ary[1, :])
print(ary[1:, 1])

# 2次元配列を取得
print(ary[:2, :])
print(ary[:, 1:])
# 位置を指定して代入や演算することも可能
ary = np.arange(12).reshape(3, 4)
print(ary)

ary[0, 0] = 99
ary[1:,1:] += 10
print(ary)
ary = np.arange(12).reshape(3, 4)
print(ary)

# （参考）複雑な値の取得

# インデックスの配列による範囲の取得
ids = np.array([0,2])
print(ary[ids, :])

# True/Falseの配列による範囲の取得
mask = np.array([True, False, True])
print(ary[mask, :])
# reshapeで変形が可能
ary = np.arange(12).reshape(3, 4)
print(ary)

# 転置
print(ary.T)

# (-1)でその次元以外を決めることができる
ary2 = ary.reshape(2, -1) 
print(ary2)

# vstack, hstackで配列を結合することができる
# vertical/horizontal
ary3 = np.vstack([ary, ary])
print(ary3)
ary4 = np.hstack([ary, ary])
print(ary4)
# ary, ary1は以下とします。
ary = np.arange(6)
ary1 = np.array([[0, 1, 2],
                 [13, 14, 15]])

# YOUR CODE GOES HERE
import pandas as pd

# 2次元配列からデータフレームの作成
# 実際はcsvなどを読み込むことが多いので、あまり使わないかも
# 2次元配列と列名を指定する
ary = np.arange(6).reshape(2, 3)
df = pd.DataFrame(ary, columns=["A", "B", "C"])
print(df)
# csvの読み込み
df = pd.read_csv("../input/mlsinput/ch3-1.csv")
print(df)                
# Excelの読み込み - 左上から２次元テーブルにしておけば、Excelシートをそのまま読み込むことができる
df = pd.read_excel("../input/mlsinput/ch3-1.xlsx", "tbl")
print(df)
import os

ary = np.arange(12).reshape(3, 4)
df = pd.DataFrame(ary, columns=["A", "B", "C", "D"])

# 区切りをカンマ、インデックスを出力する
df.to_csv("ch3-out1-1.txt")

# 区切りをタブ、インデックスを出力しない
df.to_csv("ch3-out1-2.txt", sep="\t", index=False)
# Excelへの出力
writer = pd.ExcelWriter('ch3-out1.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()
import pandas as pd

ary = np.arange(600).reshape(100, 6)
df = pd.DataFrame(ary, columns=["A","B","C","D","E","F"])

# 最初と最後の数行を表示
print(df.head())
print(df.tail(3))

# データフレームの要素の配列を取得
ary = df.values
print(type(ary), ary.shape)

# データフレームの行数、要素数
print(len(df))
print(df.shape)

# データフレームの列（インデックス型で保持されている。その中身をリストとして取り出したい場合はvaluesを使う）
print(df.columns, df.columns.values)

# データフレームのインデックス（インデックス型で保持されている。その中身をリストとして取り出したい場合はvaluesを使う）
print(df.index, df.index.values)
ary = np.arange(20).reshape(5, 4)
df = pd.DataFrame(ary, columns=["A","B","C","D"])

# 行ベースでの取得
# まずは、場所でとるilocから覚えたほうがわかりやすい
# （インデックスでとるlocという方法もある）
print(df.iloc[1])  # Seriesとなる
print(df.iloc[1:3]) # 抽出されたDataFrameとなる

# 列ベースでの取得
print(df["A"])  # Seriesとなる
print(df[["B", "C"]]) # 抽出されたDataFrameとなる

# Boolean Indexing - 条件分岐して抽出したいときに
mask = df["A"] > 6
print(mask)
print(df[mask])

# Boolean Indexingを一文で書くとこんな感じ
print(df[df["A"] > 6])
ary = np.arange(20).reshape(5, 4)
df = pd.DataFrame(ary, columns=["A","B","C","D"])

# 列の追加
df["E"] = np.array([1,2,3,4,5])
print(df)

# 列の加工
df["E"] *= np.array([1,2,3,4,5])
print(df)

# （参考）np.whereを使う
df["D"] = np.where(df["D"] % 3 == 0, 100, df["D"])
print(df)
df = pd.read_csv("../input/mlsinput/ch3-2.csv")
print(df)
g = df.groupby("shop") # groupbyではgroupybyオブジェクトというのが作成される
print(g)
g = g[["count","sales"]].sum() # sumなどの集計関数を使うことで、データフレームになる（集計キーはインデックスになる）
print(g)
g = g.reset_index() # インデックスの扱いに慣れないうちは、reset_index関数でインデックスを列に戻して扱うのがおすすめ
print(g)
df = pd.read_csv("../input/mlsinput/ch3-2.csv")

# YOUR CODE GOES HERE
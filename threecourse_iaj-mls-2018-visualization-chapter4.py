import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # 見た目をseabornベースとし、少し綺麗にする
fig, ax = plt.subplots() # figureオブジェクト、axesオブジェクトの作成
x = np.arange(1, 10)
y = x * x
ax.plot(x, y) # ax.plotで折れ線グラフを表示する
plt.show() # plt.showで図を作成する
# 2*3の図を作成する
f, axes = plt.subplots(2, 3, figsize=(9,6))
for ax_y in range(2):
    for ax_x in range(3):
        x = np.arange(50) * 0.1
        # axes[y, x]で指定する
        if ax_y == 0:
           y = np.sin(x * (ax_x + 1))
        else:
           y = np.cos(x * (ax_x + 1))
        axes[ax_y, ax_x].plot(x, y)
        
plt.show()
fig, ax = plt.subplots()
x = np.arange(50) * 0.1
y = x
y2 = np.sin(x)
y3 = np.cos(x)

# 折れ線グラフのプロット
ax.plot(x, y)
ax.plot(x, y2)
ax.plot(x, y3)
plt.show()
fig, ax = plt.subplots()

# 正規乱数を生成
rand = np.random.RandomState(71)
r = rand.standard_normal((2, 100))
x = r[0, :]
y = r[1, :]

# 散布図のプロット
ax.scatter(x, y)

plt.show()
fig, ax = plt.subplots()

# 正規乱数を生成
rand = np.random.RandomState(71)
r = rand.standard_normal((2, 100))
x = r[0, :]
y = r[1, :]

# 散布図のプロット
ax.scatter(x, y, c="red", s=10)

plt.show()
fig, ax = plt.subplots()
y = np.array([2,3,4,6,7,11,4,6])
x = np.arange(len(y))

# 棒グラフのプロット
plt.bar(x, y)
plt.show()
fig, ax = plt.subplots()
y1 = np.array([1,1,1,1,2,2,2,2])
y2 = np.array([2,3,4,5,5,4,3,2])
x = np.arange(len(y1))

# 棒グラフのプロット
plt.bar(x, y1)
plt.bar(x, y2, bottom=y1)
plt.show()
fig, ax = plt.subplots()

# 正規乱数を生成
rand = np.random.RandomState(71)
x = rand.standard_normal(100)

# ヒストグラムのプロット
ax.hist(x, bins=30)
plt.show()
fig, ax = plt.subplots()

# 2次元配列を作成
x = np.zeros((8, 8))
x += np.arange(8)
x += np.arange(8).reshape(8, 1)

# ヒートマップのプロット
# どのaxesオブジェクトに出力するかを指定するため、ax=axと引数を設定している
# わかりやすくするために、annot=Trueで数値を表示している
sns.heatmap(x, ax=ax, annot=True)
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# パレットの作成
palettes = []
palettes.append(sns.color_palette("Set1", 8))
palettes.append(sns.color_palette("husl", 8))
palettes.append(sns.color_palette('Greens', 8))

x = np.arange(50) * 0.1
ys = []
for i in range(8):
    y = np.sin(x + 0.35 * i)
    ys.append(y)

# 折れ線グラフのプロット
for p in range(3):
    for i in range(8):
        palette = palettes[p]
        axes[p].plot(x, ys[i], c=palette[i])

plt.show()
fig, ax = plt.subplots()

x = np.arange(50) * 0.1
ys = []
for i in range(8):
    y = np.sin(x + 0.35 * i)
    ys.append(y)

# 折れ線グラフのプロット
for i in range(8):
    ax.plot(x, ys[i], label="line {}".format(i))

# タイトル
ax.set_title("sin curve")

# x軸ラベル, y軸ラベル
ax.set_xlabel("x")
ax.set_ylabel("y", rotation=0)

# xの範囲を指定
ax.set_xlim(0.5, 3.0)

# yの範囲を指定
ax.set_ylim(-2, 2)

# 凡例の表示
ax.legend(loc="upper right")

# x軸目盛り, y軸目盛りの変更
# (他にもいろいろな指定方法があります)
ax.set_xticks([1,2])
ax.set_xticklabels(["one", "two"])
ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])

plt.show()
# YOUR CODE GOES HERE
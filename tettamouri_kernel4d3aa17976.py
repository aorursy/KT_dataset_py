import pandas as pd 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# CSVからDataFrameを読み込み
df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

# Xへ特徴量を代入
X = np.array(df["sqft_living"]).reshape([-1,1])

# yへ正解データを代入
y = np.array(df["price"])

# 線形回帰モデルのインスタンス生成
regressor = LinearRegression()

# Xとyで学習
regressor.fit(X,y)

# 予測値生成
y_hat = regressor.predict(X.reshape([-1,1]))

# プロット
## グラフ面積
plt.figure(figsize=(6.5,5))
## 各データのプロット
plt.scatter(X,y,color='darkgreen',label="Data", alpha=.1)
## 各データの１次元線分のプロット
plt.plot(X,regressor.predict(X),color="red",label="Predicted Regression Line")
## 縦横ラベル
plt.xlabel("Living Space (sqft)", fontsize=15)
plt.ylabel("Price ($)", fontsize=15)
## 縦横のメモリ表記
plt.xticks(np.arange(0, 8000 + 1, 1000))
plt.yticks(np.arange(0, 6e6, 1e6))
## 縦横のメモリ範囲
plt.xlim(0, 8000)
plt.ylim(0, 6e6)
## Y軸のメモリ数値の表記（指数表記を利用して強引にデフォ表記にしている）
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,10))
## 凡例表示
plt.legend()


plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)



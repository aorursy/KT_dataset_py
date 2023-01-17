import cv2 
import numpy as np
import matplotlib.pyplot as plt
from urllib import request
%matplotlib inline

# 警告文を非表示にする
import warnings
warnings.filterwarnings('ignore')

# 画像のダウンロード
url = 'https://storage.googleapis.com/kagglesdsdata/datasets%2F723403%2F1257642%2Frena.png?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1592698652&Signature=b13a878H8NOuasRah09eS2GdqNfXrlcyiZ55KbH9q1i5gAl9K2Gz6h7MOmIrFWYnfPlag4EmPqJD%2FYc3EmEhRsxqhaFIelwOoL6mTZTNCq34L80d0mcNHROmMUJj%2FlHricKfT1LRZWReI1m07I1iT5y6uzXktQHd0SC6pGACOatzb19A%2BQK8usGhbtQPfxCmDYwbdaId6RlC%2Bvy97aA09shwUmaeEzPnDp9szalSA7cptDmPiiTUvhTVV9jv0yrjLagm6qAhcmgLDaWzqa89iFcEH%2FxN6CG%2Bl%2BJJGbz3WFd9IhM5KJVqeVj%2FF5bde7Xilid%2Ftzhv2AEyzk2GoA9WtA%3D%3D'
request.urlretrieve(url, 'rena.png')

# 画像の読み込み
img = cv2.imread('rena.png')

print(type(img)) # 型の確認
print(img.shape) # 画像サイズの確認


# matplotlibによる画像の表示
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# 画像データの保存
cv2.imwrite('new_sample.png', img)
import cv2
import matplotlib.pyplot as plt
%matplotlib inline 

# 画像を読み込み
url = 'https://storage.googleapis.com/kagglesdsdata/datasets%2F723403%2F1257642%2Frena.png?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1592698652&Signature=b13a878H8NOuasRah09eS2GdqNfXrlcyiZ55KbH9q1i5gAl9K2Gz6h7MOmIrFWYnfPlag4EmPqJD%2FYc3EmEhRsxqhaFIelwOoL6mTZTNCq34L80d0mcNHROmMUJj%2FlHricKfT1LRZWReI1m07I1iT5y6uzXktQHd0SC6pGACOatzb19A%2BQK8usGhbtQPfxCmDYwbdaId6RlC%2Bvy97aA09shwUmaeEzPnDp9szalSA7cptDmPiiTUvhTVV9jv0yrjLagm6qAhcmgLDaWzqa89iFcEH%2FxN6CG%2Bl%2BJJGbz3WFd9IhM5KJVqeVj%2FF5bde7Xilid%2Ftzhv2AEyzk2GoA9WtA%3D%3D'
request.urlretrieve(url, 'rena.png')

cv_img = cv2.imread('rena.png')
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
plt.imshow(cv_img)


# コントラスト調整
# ルックアップテーブルの生成
min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )

# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255

# ローコントラストLUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

# 変換
src = cv2.imread("reni.jpg", 1)
high_cont_img = cv2.LUT(cv_img, LUT_HC)
low_cont_img = cv2.LUT(cv_img, LUT_LC)


plt.imshow(high_cont_img)

plt.imshow(low_cont_img)
#平均化フィルタ
average_square = (10,10)
blur_img = cv2.blur(cv_img, average_square)
plt.imshow(blur_img)
#ガウス分布に基づくノイズ
row,col,ch= cv_img.shape
mean = 0
sigma = 100
gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss = gauss.reshape(row,col,ch)
gauss_img = cv_img + gauss
plt.imshow(gauss_img)
#反転
hflip_img = cv2.flip(cv_img, 1)
vflip_img = cv2.flip(cv_img, 0)
plt.imshow(hflip_img)
plt.imshow(vflip_img)
# 画像のリサイズ
resize = cv2.resize(cv_img, (120, 120))

# リサイズ前の画像を出力
plt.subplot(1, 2, 1)
plt.imshow(cv_img)

# リサイズ後の画像を出力
plt.subplot(1, 2, 2)
plt.imshow(resize)
#画像の切り抜き
crop_img = cv_img[100:400, 100:400, :]
# クロップを行った後の画像を出力
plt.imshow(crop_img)
# グレースケール変換
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 画像の表示
plt.imshow(gray_img, cmap='gray')
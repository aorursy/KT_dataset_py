import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
img = cv2.imread('../input/cake-data/1.jpg') #1番目のケーキのデータ

# print(img) #見てみる
print(img.ravel()) # ほぐす
print(img.shape) # 縦と横のピクセル数
# グレースケール画像を作成
# img = cv2.imread(元画像のパス, cv2.IMREAD_GRAYSCALE)
# cv2.imwrite(グレースケール画像のパス, img)
#　ヒストグラムを出す為の関数
def plot_hist(img):
    img_hist = np.histogram(img.ravel(), 256, [0, 256])
    hist = img_hist[0]
    plt.bar(np.arange(256), hist)
    plt.show()
#チョコタルト のヒストグラム
plot_hist(cv2.imread('../input/cake-data/1.jpg'))
# plot_hist(cv2.imread('cake/1.jpg', cv2.IMREAD_GRAYSCALE)) #グレースケール画像の場合
# ロールケーキ のヒストグラム
plot_hist(cv2.imread('../input/cake-data/2.jpg'))
#ブルーベリーのケーキ のヒストグラム
plot_hist(cv2.imread('../input/cake-data/3.jpg'))
# CSVデータを読み込む

target_data = pd.read_csv('../input/csv-data/class.csv') 
print(target_data.head())
# print(target_data['chocolate'].head()) # チョコレート判定の列のみ
images_data = np.empty((0, 256), int)

for i in range(1,41):
    png = '../input/cake-data/' + str(i) + '.jpg'
    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE) #グレースケールにして読み込む
    hist = np.histogram(img.ravel(), 256, [0,256])
    images_data = np.append(images_data, np.array([hist[0]]), axis=0)
    
print(images_data.shape)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, Y_train, Y_test = train_test_split(images_data, target_data['chocolate'], random_state=0)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
# K近傍法で学習させる

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)  # 教師データXの入力データと、対応する期待する出力の値を与える
#テストデータで試す
print(knn.predict(np.array([X_test[0], X_test[1],  X_test[2],  X_test[3], ]))) #分類結果
print(Y_test) 
Y_pred = knn.predict(X_test) 
print(Y_pred) #10個すべての分類結果
print(np.mean(Y_pred == Y_test)) #正解率
images = []

for i in range(1,41): # cakeフォルダの、1.jpg 〜 40.jpg  を順に処理
    file = '../input/cake-data/' + str(i) + '.jpg'
    img = cv2.imread(file) #ファイルを読み込む    グレーにする場合は  img = cv2.imread(file, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (2000, 1000)) #サイズがバラバラなので、ここで合わせる
    images.append(img)
# print(images)
images_data = np.empty((40,len(images[0].ravel())),int) #空のNumpy配列を作成

for i in range(40):
    images_data[i] = np.array([images[i].ravel()]) #1次元配列に直したものを、代入
print(images_data.shape)
X_train, X_test, Y_train, Y_test = train_test_split(images_data, target_data['chocolate'], random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
# K近傍法で学習させる

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)  # 教師データXの入力データと、対応する期待する出力の値を与える
#テストデータで試す

print(knn.predict(np.array([X_test[0], X_test[1],  X_test[2],  X_test[3], ]))) #分類結果

print(Y_test) 
Y_pred = knn.predict(X_test) 
print(Y_pred) #10個すべての分類結果
print(np.mean(Y_pred == Y_test)) #正解率
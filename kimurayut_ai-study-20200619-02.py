!wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
#一人
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from urllib import request
%matplotlib inline

# 警告文を非表示にする
import warnings
warnings.filterwarnings('ignore')

cascade_path = './haarcascade_frontalface_alt.xml'

color = (255, 255, 255) # 白

url = 'https://storage.googleapis.com/kagglesdsdata/datasets%2F723403%2F1257642%2Frena.png?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1592698652&Signature=b13a878H8NOuasRah09eS2GdqNfXrlcyiZ55KbH9q1i5gAl9K2Gz6h7MOmIrFWYnfPlag4EmPqJD%2FYc3EmEhRsxqhaFIelwOoL6mTZTNCq34L80d0mcNHROmMUJj%2FlHricKfT1LRZWReI1m07I1iT5y6uzXktQHd0SC6pGACOatzb19A%2BQK8usGhbtQPfxCmDYwbdaId6RlC%2Bvy97aA09shwUmaeEzPnDp9szalSA7cptDmPiiTUvhTVV9jv0yrjLagm6qAhcmgLDaWzqa89iFcEH%2FxN6CG%2Bl%2BJJGbz3WFd9IhM5KJVqeVj%2FF5bde7Xilid%2Ftzhv2AEyzk2GoA9WtA%3D%3D'
request.urlretrieve(url, 'rena.png')

#ファイル読み込み
image = cv2.imread('rena.png')
#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

print("face rectangle")
print(facerect)

if len(facerect) > 0:
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        # thicknessは 線や円などの太さです。デフォルト値は1ですが、値を大きくする事で太くなります。
        # 例えば円などの図形に対し -1 が指定された場合，そのオブジェクトは塗りつぶされます。 
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        # 下記のコードと同じ意味です。
        # cv2.rectangle(image, (186, 98),(280, 192),(255,255,255), thickness=2)

    #認識結果の保存
    cv2.imwrite("detected.jpg", image)
    #認識結果の表示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
#複数人
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from urllib import request
%matplotlib inline

# 警告文を非表示にする
import warnings
warnings.filterwarnings('ignore')

cascade_path = './haarcascade_frontalface_alt.xml'

color = (255, 255, 255) # 白

url = 'https://storage.googleapis.com/kagglesdsdata/datasets%2F723463%2F1257733%2FAKB.jpg?GoogleAccessId=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com&Expires=1592701632&Signature=Ar%2FveVMdOgi1ui%2FVveEs8O%2B9%2BPu59D%2BDdmAI2Pm8sDefvJS49PB31u%2FXM4SxPqILvTxV1TghckmMxuf3Nutre4i2VkNfGPjCSzXzmozNjwxDn5%2FRVrnF9f3rTtRqtJjagG4d0FxfiR2sjjy5li2OjCxTacih9D73P4d3ii8U5e9u58syVghVVIKydWx8wW0e6Jp6X6T4PlJoQ8IPctd5l26NdqA%2Bz9a2U6vSOOClMWM6er0IUMt1rHs2uVCDH38gIJM3SzEJ533s0zAUgPXhJFFhO680buf8uzOiA70OaU2b7yDZzIW8nGaVGwpwsVZvPpp1%2BVlY589VsqXzFZIhiQ%3D%3D'
request.urlretrieve(url, 'rena.png')

#ファイル読み込み
image = cv2.imread('rena.png')
#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

print("face rectangle")
print(facerect)

if len(facerect) > 0:
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        # thicknessは 線や円などの太さです。デフォルト値は1ですが、値を大きくする事で太くなります。
        # 例えば円などの図形に対し -1 が指定された場合，そのオブジェクトは塗りつぶされます。 
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        # 下記のコードと同じ意味です。
        # cv2.rectangle(image, (186, 98),(280, 192),(255,255,255), thickness=2)

    #認識結果の保存
    cv2.imwrite("detected.jpg", image)
    #認識結果の表示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
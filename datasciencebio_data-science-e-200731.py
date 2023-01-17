import pandas as pd

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras.models import Model

from tensorflow.keras import metrics

from tensorflow.keras import backend as K
# Reading the folder architecture of Kaggle to get the dataset path.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# 訓練データとテストデータを読み込む

mnist_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

mnist_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# データの行と列の数を見る

print(mnist_train.shape, mnist_test.shape)
# 前から5つの要素を見る

mnist_train.head()
# 各列ごとの要約統計量を見る。個数、平均、標準偏差、最小値、1/4分位数、中央値、3/4分位数、最大値)

mnist_train.describe()
# データの欠損がないかを見る

mnist_train.isna().any().any()
# データを入力と出力の特徴量に分割して学習させることで、何を取り入れて何を捨てるかに基づいてモデルを学習させることができる。

mnist_train_data = mnist_train.loc[:, "pixel0":]

mnist_train_label = mnist_train.loc[:, "label"]



# 画像の配列を最大値で除算して0-1の範囲になるようにします. 

# 画像のピクセルの値の範囲は255なので, ここでは255とします. 

mnist_train_data = mnist_train_data/255.0

mnist_test = mnist_test/255.0
# グラフを見る 

digit_array = mnist_train.loc[3, "pixel0":]

arr = np.array(digit_array) 



#.reshape(a, (28,28))

image_array = np.reshape(arr, (28,28))



digit_img = plt.imshow(image_array, cmap=plt.cm.binary)

plt.colorbar(digit_img)

print("IMAGE LABEL: {}".format(mnist_train.loc[3, "label"]))
from sklearn.preprocessing import StandardScaler



standardized_scalar = StandardScaler()

standardized_data = standardized_scalar.fit_transform(mnist_train_data)

standardized_data.shape
cov_matrix = np.matmul(standardized_data.T, standardized_data)

cov_matrix.shape
from scipy.linalg import eigh



lambdas, vectors = eigh(cov_matrix, eigvals=(782, 783))

vectors.shape
vectors = vectors.T

vectors.shape
new_coordinates = np.matmul(vectors, standardized_data.T)

print(new_coordinates.shape)

new_coordinates = np.vstack((new_coordinates, mnist_train_label)).T
df_new = pd.DataFrame(new_coordinates, columns=["f1", "f2", "labels"])

df_new.head()
sns.FacetGrid(df_new, hue="labels", size=6).map(plt.scatter, "f1", "f2").add_legend()

plt.show()
from sklearn import decomposition



pca = decomposition.PCA()

pca.n_components = 2

pca_data = pca.fit_transform(standardized_data)

pca_data.shape
pca_data = np.vstack((pca_data.T, mnist_train_label)).T
df_PCA = pd.DataFrame(new_coordinates, columns=["f1", "f2", "labels"])

df_PCA.head()
sns.FacetGrid(df_new, hue="labels", size=12).map(plt.scatter, "f1", "f2").add_legend()

plt.savefig("PCA_FacetGrid.png")

plt.show()
# 関数の意味http://yshampei.hatenablog.com/entry/2017/11/18/180402 https://qiita.com/Sa_qiita/items/fc61f776cef657242e69

pca.n_components = 784

pca_data = pca.fit_transform(standardized_data)

percent_variance_retained = pca.explained_variance_ / np.sum(pca.explained_variance_)



cum_variance_retained = np.cumsum(percent_variance_retained)


plt.figure(1, figsize=(10, 6))

plt.clf()

plt.plot(cum_variance_retained, linewidth=2)

plt.axis("tight")

plt.grid()

plt.xlabel("number of compoments")

plt.ylabel("cumulative variance retained")

plt.savefig("pca_cumulative_variance.png")

plt.show()

# グラフでどの数字がどれだけあるか見る

sns.countplot(mnist_train.label)

print(list(mnist_train.label.value_counts().sort_index()))
# データフレームを配列に変換

mnist_train_data = np.array(mnist_train_data)

mnist_train_label = np.array(mnist_train_label)
# さっきのトレーニングデータの形を28×28×1のグレースケールに再編成

mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape)
# 必要なライブラリを読み込む

# TensorFlow is Google's open source AI framework and we are using is here to build model.

# Keras is built on top of Tensorflow and gives us

# NO MORE GEEKY STUFF, Know more about them here:  https://www.tensorflow.org     https://keras.io



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D

from tensorflow.keras.optimizers import Adadelta

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import LearningRateScheduler
# ラベルをエンコードしてクラスラベルにし、それをカテゴリ変数として変換する

nclasses = mnist_train_label.max() - mnist_train_label.min() + 1

mnist_train_label = to_categorical(mnist_train_label, num_classes = nclasses)

print("Shape of ytrain after encoding: ", mnist_train_label.shape)
def build_model(input_shape=(28, 28, 1)):

    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))

    return model



    

def compile_model(model, optimizer='adam', loss='categorical_crossentropy'):

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    

    

def train_model(model, train, test, epochs, split):

    history = model.fit(train, test, shuffle=True, epochs=epochs, validation_split=split)

    return history
# モデルの構築、コンパイル、訓練を行うために構築された上記の関数を使用してモデルを訓練する

cnn_model = build_model((28, 28, 1))

compile_model(cnn_model, 'adam', 'categorical_crossentropy')



# モデルを好きなだけエポック「一つの訓練データを何回繰り返して学習させるか」してトレーニングするが、、80以上でトレーニングしても何の役にも立ちませんし、最終的には過学習を増加させることがわかった

model_history = train_model(cnn_model, mnist_train_data, mnist_train_label, 80, 0.2)
def plot_model_performance(metric, validations_metric):

    plt.plot(model_history.history[metric],label = str('Training ' + metric))

    plt.plot(model_history.history[validations_metric],label = str('Validation ' + metric))

    plt.legend()

    plt.savefig(str(metric + '_plot.png'))
plot_model_performance('accuracy', 'val_accuracy')
plot_model_performance('loss', 'val_loss')
# 訓練データと同様にテストデータを配列に変換する

mnist_test_arr = np.array(mnist_test)

mnist_test_arr = mnist_test_arr.reshape(mnist_test_arr.shape[0], 28, 28, 1)

print(mnist_test_arr.shape)
# テストデータの予測

predictions = cnn_model.predict(mnist_test_arr)
# 確信度の一番高いものを判定してリストに追加「https://www.tensorflow.org/tutorials/keras/classification?hl=ja」

predictions_test = []



for i in predictions:

    predictions_test.append(np.argmax(i))
# 提出

submission =  pd.DataFrame({

        "ImageId": mnist_test.index+1,

        "Label": predictions_test

    })



submission.to_csv('my_submission.csv', index=False)
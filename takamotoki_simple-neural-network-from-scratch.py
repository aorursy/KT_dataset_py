# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')
train_data.head()
train_data.describe()
test_data.head()
X_train = train_data.drop('label',axis = 1).values

y_train = train_data['label'].values



X_test = test_data.values
X_train.shape, y_train.shape, X_test.shape
index = 0

image = X_train[index].reshape(28,28)

# X_train[index]: (784,)

# image: (28, 28)

plt.imshow(image, 'gray')

plt.title('label : {}'.format(y_train[index]))

plt.show()
X_train = X_train.astype(np.float)

X_test = X_test.astype(np.float)

X_train /= 255

X_test /= 255



print(X_train.max()) # 1.0

print(X_train.min()) # 0.0
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

y_train_one_hot = enc.fit_transform(y_train[:, np.newaxis])



print(y_train.shape)

print(y_train_one_hot.shape)

print(y_train_one_hot.dtype)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_one_hot, test_size=0.2)



print(X_train.shape, y_train.shape)

print(X_val.shape, y_val.shape)
class GetMiniBatch:

    """

    ミニバッチを取得するイテレータ



    Parameters

    ----------

    X : 次の形のndarray, shape (n_samples, n_features)

      訓練データ

    y : 次の形のndarray, shape (n_samples, 1)

      正解値

    batch_size : int

      バッチサイズ

    seed : int

      NumPyの乱数のシード

    """

    def __init__(self, X, y, batch_size = 20, seed=0):

        self.batch_size = batch_size

        np.random.seed(seed)

        shuffle_index = np.random.permutation(np.arange(X.shape[0]))

        self._X = X[shuffle_index]

        self._y = y[shuffle_index]

        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)

        

    def __len__(self):

        return self._stop

    

    def __getitem__(self,item):

        p0 = item*self.batch_size

        p1 = item*self.batch_size + self.batch_size

        return self._X[p0:p1], self._y[p0:p1]     

    

    def __iter__(self):

        self._counter = 0

        return self

    

    def __next__(self):

        if self._counter >= self._stop:

            raise StopIteration()

        p0 = self._counter*self.batch_size

        p1 = self._counter*self.batch_size + self.batch_size

        self._counter += 1

        return self._X[p0:p1], self._y[p0:p1]
class ScratchSimpleNeuralNetrowkClassifier():

    """

    シンプルな三層ニューラルネットワーク分類器



    Parameters

    ----------

    activation : str

        活性化関数：sigmoid(シグモイド関数) or tanh(ハイパボリックタンジェント関数)

    n_nodes1 : int

        1層目のノード数

    n_nodes2 : int

        2層目のノード数

    n_output : int

        出力のクラス数



    Attributes

    ----------

    self.W1 : 次の形のndarray, shape (self.n_features, self.n_nodes1)

        1層目の重み

    self.B1 : 次の形のndarray, shape (self.n_nodes1)

        1層目のバイアス

    self.W2 : 次の形のndarray, shape (self.n_nodes1, self.n_nodes2)

        2層目の重み

    self.B2 : 次の形のndarray, shape (self.n_nodes2)

        2層目のバイアス

    self.W3 : 次の形のndarray, shape (self.n_nodes2, self.n_output)

        3層目の重み

    self.B3 : 次の形のndarray, shape (self.n_output)

        3層目のバイアス

    self.A1 : 次の形のndarray, shape (self.batch_size, self.n_nodes1)

        1層目の線形結合の値

    self.Z1 : 次の形のndarray, shape (self.batch_size, self.n_nodes1)

        1層目の活性化関数の線形変換による値

    self.A2 : 次の形のndarray, shape (self.batch_size, self.n_nodes2)

        2層目の線形結合の値

    self.Z2 : 次の形のndarray, shape (self.batch_size, self.n_nodes2)

        2層目の活性化関数の線形変換による値

    self.epochs : int

        エポック数(初期値：10)

    self.batch_size : int

        バッチサイズ(初期値：20)

    self.n_features : int

        特徴量の数

    self.val_is_true : boolean

        検証用データの有無    

    self.loss : 空のndarray

        訓練データに対する損失の記録

    self.loss_val : 空のndarray

        検証データに対する損失の記録

    """

    def __init__(self, activation, n_nodes1, n_nodes2, n_output):

        self.activation = activation      # 活性化関数     

        self.n_nodes1 = n_nodes1      # 1層目のノード数

        self.n_nodes2 = n_nodes2      # 2層目のノード数

        self.n_output = n_output        # 出力のクラス数（10 : 3層目のノード数）        

        

    # 問題 1

    def __initialize_weights(self):

        """

        重みを初期化する関数

        """

        sigma = 0.01 # ガウス分布の標準偏差

        

        self.W1 = sigma * np.random.randn(self.n_features, self.n_nodes1)

        self.B1 = sigma * np.random.randn(self.n_nodes1)

        

        self.W2 = sigma * np.random.randn(self.n_nodes1, self.n_nodes2)

        self.B2 = sigma * np.random.randn(self.n_nodes2)

        

        self.W3 = sigma * np.random.randn(self.n_nodes2, self.n_output)

        self.B3 = sigma * np.random.randn(self.n_output)

    

    def __forward_propagation(self, X):

        """

        フォワードプロバゲーション

        """

        # 1層目

        self.A1 = np.dot(X, self.W1) + self.B1        

        self.Z1 = self.__activation_function(self.A1)

        

        # 2層目

        self.A2 = np.dot(self.Z1, self.W2) + self.B2        

        self.Z2 = self.__activation_function(self.A2)           

        

        # 3層目

        A3 = np.dot(self.Z2, self.W3) + self.B3



        return self.__softmax_function(A3)

    

    def __activation_function(self, A):

        """

        活性化関数 :

        シグモイド関数    

        ハイパボリックタンジェント関数

        """

        if self.activation == 'sigmoid':           

            return 1 / (1 + np.exp(-A))

        elif self.activation == 'tanh':            

            return np.tanh(A)   

        else:

            raise NameError("name \"" + str(self.activation) + "\" is not defined. set either \"sigmoid\" or \"tanh\" .") 



    def __softmax_function(self, A):

        """

        ソフトマックス関数にて各クラスに属する確率を計算

        合計値: 1.0

        """        

        return np.exp(A) / np.sum(np.exp(A), axis=1, keepdims=True)

    

    # 問題 3

    def __cross_entropy_error(self, y, z):

        """

        交差エントロピー誤差で目的関数を計算する関数

        """

        batch_size = y.shape[0]



        return -np.sum(y * np.log(z)) / batch_size



    def __backpropagation(self):

        """

        バックプロパゲーション

        """

        # 3層目

        partial_A3 = self.Z3 - self.y_



        partial_B3 = np.sum(partial_A3, axis=0)     

        partial_W3 = np.dot(self.Z2.T,  partial_A3)

        partial_Z2 = np.dot(partial_A3, self.W3.T)         



        self.W3, self.B3 = self.__stochastic_gradient_descent(self.W3, partial_W3, self.B3, partial_B3) # W3とB3の重みの更新値        



        # 2層目

        partial_A2 = self.__derivative_function(self.A2, partial_Z2)



        partial_B2 = np.sum(partial_A2, axis=0)

        partial_W2 = np.dot(self.Z1.T,  partial_A2)    

        partial_Z1 = np.dot(partial_A2, self.W2.T)         



        self.W2, self.B2 = self.__stochastic_gradient_descent(self.W2, partial_W2, self.B2, partial_B2) # W2とB2の重みの更新値



        # 1層目

        partial_A1 = self.__derivative_function(self.A1, partial_Z1)



        partial_B1 = np.sum(partial_A1, axis=0)

        partial_W1 = np.dot(self.X_.T,  partial_A1)

        

        self.W1, self.B1 = self.__stochastic_gradient_descent(self.W1, partial_W1, self.B1, partial_B1) # W1とB1の重みの更新値

 

    def __derivative_function(self, A, Z):

        """

        合成関数の偏微分：

        シグモイド関数の導関数

        ハイパボリックタンジェント関数の導関数

        """        

        if self.activation == 'sigmoid':

            return Z * np.multiply((1.0 - self.__sigmoid_function(A)), self.__sigmoid_function(A))

        elif self.activation == 'tanh':                

            return Z * (1.0 - (np.tanh(A) ** 2))

            

    def __stochastic_gradient_descent(self, W, partial_W, B, partial_B):

        """

        確率的勾配降下法により重みを更新する関数

        """        

        lr = 0.001 # 学習率

        W_prime = W - lr * partial_W

        B_prime = B - lr * partial_B

        

        return W_prime, B_prime

  

    def fit(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=20):

        """

        ニューラルネットワーク分類器を学習する。



        Parameters

        ----------

        X : 次の形のndarray, shape (n_samples, n_features)

            訓練データの特徴量

        y : 次の形のndarray, shape (n_samples, )

            訓練データの正解値

        X_val : 次の形のndarray, shape (n_samples, n_features)

            検証データの特徴量

        y_val : 次の形のndarray, shape (n_samples, )

            検証データの正解値

        epochs : int

            エポック数(初期値：10)

        batch_size : int

            バッチサイズ(初期値：20)        

        """

        self.epochs = epochs                            # エポック数     

        self.batch_size = batch_size               # バッチサイズ

        self.n_features = X.shape[1]               # 特徴量の数(784) 

        self.val_is_true = False                        # 検証用データの有無(初期値：False)

        self.loss = np.zeros(self.epochs)        # 学習曲線・目的関数の出力用(訓練データ)

        self.loss_val = np.zeros(self.epochs) # 学習曲線・目的関数の出力用(検証データ)

        

        self.__initialize_weights() # 重みの初期化

        

        for epoch in range(self.epochs):

            get_mini_batch = GetMiniBatch(X, y, batch_size=self.batch_size)

            for mini_X_train,  mini_y_train in get_mini_batch:

                self.X_ = mini_X_train

                self.y_ = mini_y_train



                # フォワードプロバゲーション

                self.Z3 = self.__forward_propagation(self.X_)



                # 交差エントロピー誤差

                self.loss[epoch] = self.__cross_entropy_error(self.y_, self.Z3)



                # バックプロパゲーション

                self.__backpropagation()



            if not(X_val is None) and not(y_val is None):

                self.val_is_true = True

                

                # フォワードプロバゲーション

                self.y_val_pred = self.__forward_propagation(X_val)



                # 交差エントロピー誤差

                self.loss_val[epoch] = self.__cross_entropy_error(y_val, self.y_val_pred)      



    def predict(self, X):

        """

        ニューラルネットワーク分類器を使い推定する。



        Parameters

        ----------

        X : 次の形のndarray, shape (n_samples, n_features)

            サンプル



        Returns

        -------

            次の形のndarray, shape (n_samples, 1)

            推定結果

        """

        # フォワードプロバゲーション

        y_pred = self.__forward_propagation(X)    



        return np.argmax(y_pred, axis=1)

    

    def plot_learning_curve(self):

        """

        学習曲線をプロットする。    

        """

        plt.plot(range(1, self.epochs + 1), self.loss, color="r", marker="o", label="train loss")

        if self.val_is_true:

            plt.plot(range(1, self.epochs + 1), self.loss_val, color="g", marker="o", label="val loss")

            

        plt.title("Learning Curve")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.grid()

        plt.legend(loc="best")

        plt.show()
model = ScratchSimpleNeuralNetrowkClassifier(activation='tanh', n_nodes1=400, n_nodes2=200, n_output=10)

model.fit(X_train, y_train, X_val, y_val, epochs = 20, batch_size = 20)

y_pred = model.predict(X_val)
np.bincount(y_pred)
np.sum(y_val, axis=0).astype('int')
#直列化 

y_val = np.argmax(y_val, axis=1)
print("accuracy : {}".format(accuracy_score(y_val, y_pred)))
model.plot_learning_curve()
"""

語分類結果を並べて表示する。画像の上の表示は「推定結果/正解」である。



Parameters:

----------

y_pred : 推定値のndarray (n_samples,)

y_val : 検証データの正解ラベル(n_samples,)

X_val : 検証データの特徴量（n_samples, n_features)

"""

import numpy as np

import matplotlib.pyplot as plt

num = 36 # いくつ表示するか

true_false = y_pred==y_val

false_list = np.where(true_false==False)[0].astype(np.int)

if false_list.shape[0] < num:

    num = false_list.shape[0]

fig = plt.figure(figsize=(6, 6))

fig.subplots_adjust(left=0, right=0.8,  bottom=0, top=0.8, hspace=1, wspace=0.5)

for i in range(num):

    ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])

    ax.set_title("{} / {}".format(y_pred[false_list[i]],y_val[false_list[i]]))

    ax.imshow(X_val.reshape(-1,28,28)[false_list[i]], cmap='gray')
prediction = model.predict(X_test)
submission = pd.DataFrame({'ImageId': np.array(range(1,28001)), 'Label': prediction})

submission.to_csv("submission.csv", index=False)
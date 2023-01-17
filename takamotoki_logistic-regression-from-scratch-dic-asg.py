import numpy as np

import pandas as pd



from sklearn.datasets import load_iris



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import SGDClassifier #比較用



from sklearn import metrics



from matplotlib.colors import ListedColormap

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

import seaborn as sns



import os # reading the input files we have access to

print(os.listdir('../input'))
#read sample file 

data = np.loadtxt('../input/log-sample/log_sample.csv', delimiter=',', skiprows=1)
data.shape
x_columns = ['browse_experiences', 'hours_of_study', 'exam_results']

data_frame = pd.DataFrame(data, columns=x_columns)

data_frame
#特徴量X

X = data[:, 0:2]

#目的変数y

y = data[:, 2]
#正規化

def normalize_x_features(X):

    X_norm = np.zeros((X.shape[0], X.shape[1]))

    mean = np.zeros((1, X.shape[1]))

    std = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):

        mean[:, i] = np.mean(X[:, i])

        std[:, i] = np.std(X[:, i])

        X_norm[:, i] = (X[:, i] - float(mean[:, i])) / float(std[:, i])

    return X_norm, mean, std
X_norm, mean, std = normalize_x_features(X)
X_norm[0:5, :]
print(X_norm.mean())

print(X_norm.std())
fig, axes = plt.subplots(1,2,figsize=(12,6))

left = sns.countplot(x='browse_experiences',

                  hue='exam_results',

                  data=data_frame,

                  ax=axes[0])



right = sns.barplot(x='browse_experiences',

                  y='exam_results',

                  data=data_frame,

                  ax=axes[1]).set_ylabel('Pass Rate')



axes[0].set_title('Count of Pass and Fail')

axes[1].set_title('Pass Rate')
plt.figure(figsize=(8, 6))

ax = sns.countplot(x="hours_of_study", hue="exam_results", data=data_frame)

plt.show()
fig, axes = plt.subplots(1,2,figsize=(12,6))



left_fail = sns.kdeplot(data_frame['browse_experiences'][data_frame['exam_results'] == 0],

                        shade=True, color="r", label='fail', ax=axes[0])

left_pass = sns.kdeplot(data_frame['browse_experiences'][data_frame['exam_results'] == 1],

                        shade=True, color="b", label='pass', ax=axes[0])



right_fail = sns.kdeplot(data_frame['hours_of_study'][data_frame['exam_results'] == 0],

                         shade=True, color="r", label='fail', ax=axes[1])

right_pass = sns.kdeplot(data_frame['hours_of_study'][data_frame['exam_results'] == 1],

                         shade=True, color="b", label='pass', ax=axes[1])





axes[0].set_title('Density of Pass and Fail')

axes[1].set_title('Density of Pass and Fail')
class ScratchLogisticRegression():

    """

    ロジスティック回帰のスクラッチ実装



    Parameters

    ----------

    num_iter : int

      イテレーション数

    lr : float

      学習率

    bias : bool

      バイアス項を入れる場合はTrue

    verbose : bool

      学習過程を出力する場合はTrue

    lambda_ : float

    　L2正則化パラメータ

    to_pickle_ : bool

    　学習した重みの保存有無



    Attributes

    ----------

    self.coef_ : 次の形のndarray, shape (n_features,)

      パラメータ

    self.loss : 次の形のndarray, shape (self.iter,)

      訓練データに対する損失の記録

    self.val_loss : 次の形のndarray, shape (self.iter,)

      検証データに対する損失の記録



    """

    def __init__(self, num_iter, lr, bias, verbose, lambda_=1.0, to_pickle_=False):

        # ハイパーパラメータを属性として記録

        self.iter = num_iter

        self.lr = lr

        self.lambda_ = lambda_

        self.bias = bias

        self.verbose = verbose

        self.to_pickle_ = to_pickle_

        # 損失を記録する配列を用意

        self.loss = np.zeros(self.iter)

        self.val_loss = np.zeros(self.iter)

        

    def __add_x0(self, X):

        """

        バイアス項を入れる場合、

        値1を特徴量Xの1列目に結合

        """

        x0 = np.ones((X.shape[0], 1))

        return np.concatenate((x0, X), axis=1)

    

    def __sigmoid_function(self, X):

        """

        シグモイド関数にて仮定関数を計算

        """

        return 1 / (1 + np.exp(-np.dot(X, self.theta)))               

       

    def __gradient_descent(self, X, y, h):

        """

        パラメータtheta更新用のメソッド



        Parameters

        ----------

        X : 次の形のndarray, shape (n_samples, n_features)

          訓練データ

        y : 次の形のndarray, shape (n_samples, )

          訓練データの正解値

        h : 次の形のndarray, shape (n_samples, )

        　シグモイド関数の出力結果



        Returns

        -------

        self.theta：回帰係数ベクトル shape (n_features, )



        """

        m = len(y)

        gradient = np.dot(X.T, (h - y)) / m

        if self.bias:

            l2norm = (self.lambda_ / m) * self.theta

            l2norm[0] = 0 #切片の列は0

            self.theta -= self.lr * gradient + l2norm

        else:

            l2norm = (self.lambda_ / m) * self.theta

            self.theta -= self.lr * gradient + l2norm

            

        return self.theta    



    def __loss_function(self, h, y):

        """

        損失関数を計算

        """

        m = len(y)

        loss = np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() +

                      (self.lambda_ * np.square(self.theta)) / (2 * m))



        return loss



    def fit(self, X, y, X_val=None, y_val=None):

        """

        ロジスティック回帰を学習する。検証データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。



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

        """

        if self.bias:

            X = self.__add_x0(X)

            if not(X_val is None) and not(y_val is None):

                X_val = self.__add_x0(X_val)



        self.theta = np.random.rand(X.shape[1])



        for i in range(self.iter):

            h = self.__sigmoid_function(X)

            self.theta = self.__gradient_descent(X, y, h)

            self.loss[i] = self.__loss_function(h, y)



            if not(X_val is None) and not(y_val is None):

                h_val = self.__sigmoid_function(X_val)

                self.val_loss[i] = self.__loss_function(h_val, y_val)



            if self.verbose:

                #verboseをTrueにした場合は学習過程を出力

                print("Training Data" + "*"*40)

                print("theta:{}".format(self.theta))

                print("loss:{}".format(self.loss[i]))

                if not(X_val is None) and not(y_val is None):

                    print("Validation Data" + "*"*40)

                    print("theta:{}".format(self.theta))

                    print("loss in cost function:{}".format(self.val_loss[i])) 

        

        if self.to_pickle_:

            pickle_obj = self.theta

            pd.to_pickle(pickle_obj, 'pickle_obj.pkl')



    def predict_proba(self, X):

        """

        ロジスティック回帰を使い確率を推定する。



        Parameters

        ----------

        X : 次の形のndarray, shape (n_samples, n_features)

            サンプル



        Returns

        -------

            次の形のndarray, shape (n_samples, 1)

            ロジスティック回帰による推定結果

        """

        if self.bias:

            X = self.__add_x0(X)

    

        return self.__sigmoid_function(X)

    

    def predict(self, X):

        """

        ロジスティック回帰を使いラベルを推定する。



        Parameters

        ----------

        X : 次の形のndarray, shape (n_samples, n_features)

            サンプル



        Returns

        -------

            次の形のndarray, shape (n_samples, 1)

            ロジスティック回帰による推定結果

        """

        return self.predict_proba(X).round()
model = ScratchLogisticRegression(num_iter=100, lr=0.1, bias=True, verbose=True)

model.fit(X_norm, y)
y_pred = model.predict(X_norm)

y_pred_proba = model.predict_proba(X_norm)
y_pred
y_pred_proba
def evaluate(y_test, y_pred):

    """

    2値分類の評価指標を計算・描画する。

    """

    acc = metrics.accuracy_score(y_test, y_pred)

    precision = metrics.precision_score(y_test, y_pred)

    recall = metrics.recall_score(y_test, y_pred)

    f1 = metrics.f1_score(y_test, y_pred)

    

    plt.figure(figsize=(1.6, 1.2))

    cm_def = metrics.confusion_matrix(y_test, y_pred)

    sns.heatmap(cm_def, annot=True, cmap='Blues')

    plt.title('Confusion Matrix')

    plt.show()

    return acc, precision, recall, f1
print('Accuracy : {:.2f}\n'

      'Precision : {:.2f}\n'

      'Recall : {:.2f}\n'

      'F1 : {:.2f}\n'

      .format(*evaluate(y, y_pred)))
iris_dataset = load_iris()
#オリジナルデータの作成

x_origin_columns = ['sepal_length', 'petal_length']

X_origin = pd.DataFrame(iris_dataset.data[:,[0,2]], columns=x_origin_columns)

y_origin = pd.DataFrame(iris_dataset.target, columns=['Species'])

df_origin = pd.concat([X_origin, y_origin], axis=1)
rows_to_drop = df_origin.index[df_origin['Species'] == 0 ] #'setosa'の行を削除

df = df_origin.drop(rows_to_drop, axis=0).reset_index(drop=True)

iris_mapping = {1: 0, 2: 1} #'versicolor'を0,  'virginica'を1にマッピング

df['Species'] = df['Species'].map(iris_mapping)

X = df.iloc[:, :-1].to_numpy()

y = df.loc[:, 'Species'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
scikit_model = SGDClassifier(loss="log")

scikit_model.fit(X_train_scaled, y_train.ravel())

scikit_pred = scikit_model.predict(X_test_scaled)

scikit_pred_proba = scikit_model.predict_proba(X_test_scaled)
print('Accuracy : {:.2f}\n'

      'Precision : {:.2f}\n'

      'Recall : {:.2f}\n'

      'F1 : {:.2f}\n'

      .format(*evaluate(y_test, scikit_pred)))
scratch_model = ScratchLogisticRegression(num_iter=100, lr=0.1, bias=True, verbose=True)

scratch_model.fit(X_train_scaled, y_train.ravel(), X_test_scaled, y_test.ravel())

scratch_pred = scratch_model.predict(X_test_scaled)

scratch_pred_proba = scratch_model.predict_proba(X_test_scaled)
print('Accuracy : {:.2f}\n'

      'Precision : {:.2f}\n'

      'Recall : {:.2f}\n'

      'F1 : {:.2f}\n'

      .format(*evaluate(y_test, scratch_pred)))
plt.figure()

plt.title("model loss")

plt.xlabel("iter")

plt.ylabel("loss")



x_plot = range(0, scratch_model.iter)

plt.plot(x_plot, scratch_model.loss, '-', color="b", label="train loss")

plt.plot(x_plot, scratch_model.val_loss, '-', color="r", label="val loss")



plt.legend(loc="best")



plt.show()
def decision_region(X, y, model, step=0.01, title='decision region',

                    xlabel='sepal_length', ylabel='petal_length', target_names=['versicolor', 'virginica']):

    """

    2値分類を2次元の特徴量で学習したモデルの決定領域を描く。

    背景の色が学習したモデルによる推定値から描画される。

    散布図の点は訓練データまたは検証データである。



    Parameters

    ----------------

    X : ndarray, shape(n_samples, 2)

        特徴量

    y : ndarray, shape(n_samples,)

        ラベル

    model : object

        学習したモデルのインスンタスを入れる

    step : float, (default : 0.1)

        推定値を計算する間隔を設定する

    title : str

        グラフのタイトルの文章を与える

    xlabel, ylabel : str

        軸ラベルの文章を与える

    target_names= : list of str

        凡例の一覧を与える

    """

    # setting

    scatter_color = ['red', 'blue']

    contourf_color = ['pink', 'skyblue']

    n_class = 2

    # pred

    mesh_f0, mesh_f1  = np.meshgrid(np.arange(np.min(X[:,0])-0.5,

                                              np.max(X[:,0])+0.5, step),

                                    np.arange(np.min(X[:,1])-0.5,

                                              np.max(X[:,1])+0.5, step))



    mesh = np.c_[np.ravel(mesh_f0),np.ravel(mesh_f1)]

    y_pred = model.predict(mesh).reshape(mesh_f0.shape)

    # plot

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.contourf(mesh_f0, mesh_f1, y_pred, n_class-1, cmap=ListedColormap(contourf_color))

    plt.contour(mesh_f0, mesh_f1, y_pred, n_class-1, colors='y', linewidths=3, alpha=0.5)

    for i, target in enumerate(set(y)):

        plt.scatter(X[y==target][:, 0], X[y==target][:, 1], s=80,

                    color=scatter_color[i], label=target_names[i], marker='o')

    patches = [mpatches.Patch(color=scatter_color[i], label=target_names[i]) for i in range(n_class)]

    plt.legend(handles=patches)

    plt.legend()

    plt.show()
scikit_pred, scratch_pred
decision_region(X_test_scaled, scratch_pred, scratch_model)
scratch_model = ScratchLogisticRegression(num_iter=100, lr=0.1,

                                          bias=True, verbose=False,

                                          to_pickle_=True)

scratch_model.fit(X_train_scaled, y_train.ravel())
pickle_obj = pd.read_pickle('pickle_obj.pkl')

print(pickle_obj)
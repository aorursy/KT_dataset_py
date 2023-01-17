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
# データセットをインポート

import pandas as pd

data_train = pd.read_csv('/kaggle/input/titanic/train.csv', sep = ',', header = 0)

data_test = pd.read_csv('/kaggle/input/titanic/test.csv', sep = ',', header = 0)

print(data_train)

data_train.info()

print(data_test)

data_test.info()
# Nameの敬称で分類し、それぞれの平均年齢を算出

# 年齢の欠損補充はdata_train と dat_test を合体して行う（このデータ間の相関は'Survived'と関係ないので）

combine1 = [data_train]

for data_train in combine1:

    data_train['Salutation'] = data_train.Name.str.extract(' ([A-Za-z]+).', expand=False)

for data_train in combine1:

    data_train['Salutation'] = data_train['Salutation'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data_train['Salutation'] = data_train['Salutation'].replace('Mlle', 'Miss')

    data_train['Salutation'] = data_train['Salutation'].replace('Ms', 'Miss')

    data_train['Salutation'] = data_train['Salutation'].replace('Mme', 'Mrs')



combine2 = [data_test]

for data_test in combine2:

    data_test['Salutation'] = data_test.Name.str.extract(' ([A-Za-z]+).', expand=False)

for data_test in combine2:

    data_test['Salutation'] = data_test['Salutation'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data_test['Salutation'] = data_test['Salutation'].replace('Mlle', 'Miss')

    data_test['Salutation'] = data_test['Salutation'].replace('Ms', 'Miss')

    data_test['Salutation'] = data_test['Salutation'].replace('Mme', 'Mrs')





Salutation_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

for data_train in combine1:

    data_train['Salutation'] = data_train['Salutation'].map(Salutation_mapping)

    data_train['Salutation'] = data_train['Salutation'].fillna(0)

for data_test in combine2:

    data_test['Salutation'] = data_test['Salutation'].map(Salutation_mapping)

    data_test['Salutation'] = data_test['Salutation'].fillna(0)



Age_Salutation_dn1 = data_train[['Age', 'Salutation']].dropna()

Age_Salutation_dn2 = data_test[['Age', 'Salutation']].dropna()

import numpy as np



Age_Salutation_dn1 = np.array(Age_Salutation_dn1)

Age_Salutation_dn2 = np.array(Age_Salutation_dn2)



t_age_Mr, t_age_Miss, t_age_Mrs, t_age_Master, t_age_Rare = 0, 0, 0, 0, 0

Mr_count, Miss_count, Mrs_count, Master_count, Rare_count = 0, 0, 0, 0, 0

for i in range(len(Age_Salutation_dn1)):

    age1 = Age_Salutation_dn1[i][0]

    if Age_Salutation_dn1[i][1] == 1:

        t_age_Mr += age1

        Mr_count += 1

    elif Age_Salutation_dn1[i][1] == 2:

        t_age_Miss += age1

        Miss_count += 1

    elif Age_Salutation_dn1[i][1] == 3:

        t_age_Mrs += age1

        Mrs_count += 1

    elif Age_Salutation_dn1[i][1] == 4:

        t_age_Master += age1

        Master_count += 1

    else:

        t_age_Rare += age1

        Rare_count += 1

for i in range(len(Age_Salutation_dn2)):

    age2 = Age_Salutation_dn2[i][0]

    if Age_Salutation_dn2[i][1] == 1:

        t_age_Mr += age2

        Mr_count += 1

    elif Age_Salutation_dn2[i][1] == 2:

        t_age_Miss += age2

        Miss_count += 1

    elif Age_Salutation_dn2[i][1] == 3:

        t_age_Mrs += age2

        Mrs_count += 1

    elif Age_Salutation_dn2[i][1] == 4:

        t_age_Master += age2

        Master_count += 1

    else:

        t_age_Rare += age2

        Rare_count += 1

    





m_age_Mr = t_age_Mr / Mr_count

m_age_Miss = t_age_Miss / Miss_count

m_age_Mrs = t_age_Mrs / Mrs_count

m_age_Master = t_age_Master / Master_count

m_age_Rare = t_age_Rare / Rare_count



print('Mr:', m_age_Mr)

print('Miss:', m_age_Miss)

print('Mrs:', m_age_Mrs)

print('Master:', m_age_Master)

print('Rare:', m_age_Rare)
import pandas as pd

from pandas import Series, DataFrame

import math



# すべてのデータを数値に置き換える

data_set = DataFrame(data_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)) # 使わない列を削除

data_set = data_set.fillna({'Fare':0, 'Cabin':0,'Embarked':0}) # 'Fare'と'Cabinと'Embarked'の空欄を0で埋める 

data_set = data_set.replace({'male':0, 'female':1, 'S':1, 'C':2, 'Q':3}) # 文字列を数値に置き換え(Sex, Embarked)

data_set = data_set.replace({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}, regex=True) # 文字列を含むデータを部分一致で数値に置き換え(Cabin)



# 年齢の欠損補充（Mr,Miss,Mrs,Master,Rare別に）

for i in range(len(data_set)):

    salutation = data_set['Salutation'][i]

    if salutation == 1:

        if pd.isnull(data_set['Age'][i]):

            data_set['Age'].loc[i] = m_age_Mr

    elif salutation == 2:

        if pd.isnull(data_set['Age'][i]):

            data_set['Age'].loc[i] = m_age_Miss

    elif salutation == 3:

        if pd.isnull(data_set['Age'][i]):

            data_set['Age'].loc[i] = m_age_Mrs

    elif salutation == 4:

        if pd.isnull(data_set['Age'][i]):

            data_set['Age'].loc[i] = m_age_Master

    else:

        if pd.isnull(data_set['Age'][i]):

            data_set['Age'].loc[i] = m_age_Rare



# 新たに項目を追加する

# 自分を含めた家族人数

data_set['FamilySize'] = data_set['SibSp'] + data_set['Parch'] + 1

# 一人かどうか、FamilySizeが2以上5未満、FamilySizeが5以上、Cabinが0かどうか

isAlone = []

FamilySize_M = []

FamilySize_L = []

isCabinNo = []

roundAgeArray = []

roundFareArray = []

for i in range(len(data_set)):

    if data_set['FamilySize'][i] == 1:

        isAlone.append(1)

    else:

        isAlone.append(0)

    if data_set['Cabin'][i] == 0:

        isCabinNo.append(1)

    else:

        isCabinNo.append(0)

    if data_set['FamilySize'][i] < 2:

        FamilySize_M.append(0)

        FamilySize_L.append(0)

    elif data_set['FamilySize'][i] < 5:

        FamilySize_M.append(1)

        FamilySize_L.append(0)

    else:

        FamilySize_M.append(0)

        FamilySize_L.append(1)

    # 年代

    roundAge = int(data_set['Age'][i] / 10)

    roundAgeArray.append(roundAge)

    # 料金

    if data_set['Fare'][i] > 0:

        roundFare = int(math.log(data_set['Fare'][i]))

    else:

        roundFare = 0

    roundFareArray.append(roundFare)

#    print(roundFare)



data_set['isAlone'] = isAlone

data_set['FamilySize_M'] = FamilySize_M

data_set['FamilySize_L'] = FamilySize_L

data_set['isCabinNo'] = isCabinNo

data_set['roundAge'] = roundAgeArray

data_set['roundFare'] = roundFareArray



data_set.info()

# 数値に置き換えした後のtrain.csvの内容を確認

import numpy as np



testData = np.array(data_set)

testData = testData.astype('int32')

for i in range(len(testData)):

    print(testData[i])



data_set = data_set.dropna() # 空欄のある行を削除

x = DataFrame(data_set[['Pclass', 'Sex', 'Age', 'Salutation', 'FamilySize', 'isAlone']])

t = DataFrame(data_set['Survived'])



# numpyの配列に変換

x = np.array(x)

t = np.array(t)



# numpyで型を変換

t = t.ravel()



x = x.astype('float32')

t = t.astype('int32')

# 中を確認

print('x shape:', x.shape)# n行m列の行列(n, m)になっていればOK

print(x[:10])

print('t shape:', t.shape)# n行1列のベクトル(n,)になっていればOK

print(t[:10])
# TupleDatasetで、入力値と目的値がペアになったデータセットを作成

from chainer.datasets import TupleDataset

dataset = TupleDataset(x, t)



# データセットを訓練用(train)・検証用(valid)・テスト(test)用に分割

from chainer.datasets import split_dataset_random

train_val, innerTest = split_dataset_random(dataset, int(len(dataset) * 0.9), seed=0)

train, valid = split_dataset_random(train_val, int(len(train_val) * 0.7), seed=0)

# 全データを使って訓練する

#train, valid = split_dataset_random(dataset, int(len(dataset) * 0.7), seed=0)



# SerialIteratorでミニバッチを作成

from chainer.iterators import SerialIterator

train_iter = SerialIterator(train, batch_size=64, repeat=True, shuffle=True)#バッチサイズ



print(dataset[0])
# ニューラルネットワークの学習

import chainer

import chainer.links as L

import chainer.functions as F



class Net(chainer.Chain):



  def __init__(self, n_in=6, n_hidden=100, n_out=2):

    super().__init__()

    with self.init_scope():

      self.l1 = L.Linear(n_in, n_hidden)

      self.l2 = L.Linear(n_hidden, n_hidden)

      self.l3 = L.Linear(n_hidden, n_out)



  # 活性化関数(relu,sigmoid,tanh)

  def forward(self, x):

    h = F.sigmoid(self.l1(x))

    h = F.sigmoid(self.l2(h))

    h = self.l3(h)

    return h



net = Net()



# 目的関数（交差エントロピーに、正則化項としての重み減衰を適用）

from chainer import optimizers

from chainer.optimizer_hooks import WeightDecay

#optimizer = optimizers.MomentumSGD(lr=0.0005, momentum=0.91) #学習率:0.0003、勾配係数:0.9

optimizer = optimizers.Adam(alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-08, eta=1.0,

          weight_decay_rate=0.00001, amsgrad=False, adabound=False, final_lr=0.1, gamma=0.001)

optimizer.setup(net)

#for param in net.params():

#  if param.name != 'b': #バイアス以外

#    param.update_rule.add_hook(WeightDecay(0.00001)) #重み減衰:0.00001



# ニューラルネットワークの訓練

gpu_id = 0 #使用するGPU番号

n_epoch =2000 #エポック数



# ネットワークをGPUメモリ上に転送

net.to_gpu(gpu_id)



# 結果保存用の配列（リストと辞書）

results_train, results_valid = {}, {}

results_train['loss'], results_train['accuracy'] = [], []

results_valid['loss'], results_valid['accuracy'] = [], []



train_iter.reset()



count = 1



for epoch in range(n_epoch):

  while True:

    # ミニバッチの取得

    train_batch = train_iter.next()

    # x と t に分割

    # データをGPUに転送するために、concat_examples に gpu_id を渡す

    x_train, t_train = chainer.dataset.concat_examples(train_batch, gpu_id)

    # 予測値と目的関数（交差エントロピーを採用）の計算

    y_train = net(x_train)

    loss_train = F.softmax_cross_entropy(y_train, t_train)

    acc_train = F.accuracy(y_train, t_train)

    #勾配の初期化と計算

    net.cleargrads()

    loss_train.backward()

    # パラメータの更新

    optimizer.update()

    # カウントアップ

    count += 1



    # １エポック終えたら、valid データで評価する

    if train_iter.is_new_epoch:

      # 検証用データに対する結果の確認

      with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

        x_valid, t_valid = chainer.dataset.concat_examples(valid, gpu_id)

        y_valid = net(x_valid)

        loss_valid = F.softmax_cross_entropy(y_valid, t_valid)

        acc_valid = F.accuracy(y_valid, t_valid)

      # GPU から CPU にデータを戻す

      loss_train.to_cpu()

      loss_valid.to_cpu()

      acc_train.to_cpu()

      acc_valid.to_cpu()

      # 結果の表示

      if epoch % 10 == 0:

          print('epoch: {}, iteration: {}, loss(train): {:.4f}, loss(valid): {:.4f}, acc(train): {:.4f}, acc(valid): {:.4f}'.format(epoch, count, loss_train.array.mean(), loss_valid.array.mean(), acc_train.array.mean(), acc_valid.array.mean()))

      # グラフ可視化用に保存

      results_train['loss'].append(loss_train.array)

      results_train['accuracy'].append(acc_train.array)

      results_valid['loss'].append(loss_valid.array)

      results_valid['accuracy'].append(acc_valid.array)



      break

# グラフ表示

import matplotlib.pyplot as plt



# 損失（lose）

plt.plot(results_train['loss'], label='train')

plt.plot(results_valid['loss'], label='valid')

plt.title('Graph(loss)')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend()

plt.show()



# 精度（accuracy）

plt.plot(results_train['accuracy'], label='train')

plt.plot(results_valid['accuracy'], label='valid')

plt.title('Graph(accuracy)')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend()

plt.show()
# テストデータに対する損失と精度を計算

# データセットのデータを全て訓練用として使用している場合は、この操作は無意味です

x_test, t_test = chainer.dataset.concat_examples(innerTest, device=gpu_id)

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

    y_test = net(x_test)

    loss_test = F.softmax_cross_entropy(y_test, t_test)

    acc_test = F.accuracy(y_test, t_test)

print('test loss: {:.4f}'.format(loss_test.array.get()))

print('test accuracy: {:.4f}'.format(acc_test.array.get()))
# ニューラルネットワークの保存

net.to_cpu()

chainer.serializers.save_npz('net.npz', net)



# 保存されているか確認

!ls
# 推論用データ

import math



# 整形



test = DataFrame(data_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)) # 使わない列を削除



# 年齢の欠損補充（Mr,Miss,Mrs,Master,Rare別に）

for i in range(len(test)):

    salutation = test['Salutation'][i]

    if salutation == 1:

        if pd.isnull(test['Age'][i]):

            test['Age'].loc[i] = m_age_Mr

    elif salutation == 2:

        if pd.isnull(test['Age'][i]):

            test['Age'].loc[i] = m_age_Miss

    elif salutation == 3:

        if pd.isnull(test['Age'][i]):

            test['Age'].loc[i] = m_age_Mrs

    elif salutation == 4:

        if pd.isnull(test['Age'][i]):

            test['Age'].loc[i] = m_age_Master

    else:

        if pd.isnull(test['Age'][i]):

            test['Age'].loc[i] = m_age_Rare









test = test.fillna({'Fare':0, 'Cabin':0,'Embarked':0}) # 'Fare'と'Cabin'と'Embarked'の列にある空欄を0で埋める 

test = test.replace({'male':0, 'female':1, 'S':1, 'C':2, 'Q':3}) # 文字列を数値に置き換え

test = test.replace({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}, regex=True) # 文字列を含むデータを部分一致で数値に置き換え

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

isAlone = []

FamilySize_M = []

FamilySize_L = []

isCabinNo = []

roundAgeArray = []

roundFareArray = []

for i in range(len(test)):

    if test['FamilySize'][i] == 1:

        isAlone.append(1)

    else:

        isAlone.append(0)

    if test['Cabin'][i] == 0:

        isCabinNo.append(1)

    else:

        isCabinNo.append(0)

    if test['FamilySize'][i] < 2:

        FamilySize_M.append(0)

        FamilySize_L.append(0)

    elif test['FamilySize'][i] < 5:

        FamilySize_M.append(1)

        FamilySize_L.append(0)

    else:

        FamilySize_M.append(0)

        FamilySize_L.append(1)



    roundAge = int(test['Age'][i] / 10)

    roundAgeArray.append(roundAge)

    if test['Fare'][i] > 0:

        roundFare = int(math.log(test['Fare'][i]))

    else:

        roundFare = 0

    roundFareArray.append(roundFare)



test['isAlone'] = isAlone

test['FamilySize_M'] = FamilySize_M

test['FamilySize_L'] = FamilySize_L

test['isCabinNo'] = isCabinNo

test['roundAge'] = roundAgeArray

test['roundFare'] = roundFareArray



test = DataFrame(test[['Pclass', 'Sex', 'Age', 'Salutation', 'FamilySize', 'isAlone']])



test.info()
# 推論

import chainer

import chainer.links as L

import chainer.functions as F

#ニューラルネットワークのクラス

class newNet(chainer.Chain):

    def __init__(self,n_in=6, n_hidden=100, n_out=2):

        super().__init__()

        with self.init_scope():

            self.l1 = L.Linear(n_in, n_hidden)

            self.l2 = L.Linear(n_hidden, n_hidden)

            self.l3 = L.Linear(n_hidden, n_out)



    def forward(self, x):

        h = F.sigmoid(self.l1(x))

        h = F.sigmoid(self.l2(h))

        h = self.l3(h)

        return h



loaded_net = newNet()



chainer.serializers.load_npz('net.npz', loaded_net)



import numpy as np



# 行列をニューラルネットワークに渡す

test = np.array(test)

#test = np.delete(test, 0, 0)

test = test.astype('float32')

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

    result = loaded_net(test)

#    print(result)

PassengerId = 891

outputArray = []

for i in range(len(test)):

    PassengerId += 1

    predict = np.argmax(result[i,:].array)

    innerArray = [PassengerId, predict]

    outputArray.append(innerArray)

#    print(predict)

#outputArray = np.array(outputArray)

#outputArray.shape

#outputArray

import pandas as pd

df = pd.DataFrame(outputArray, columns=['PassengerId', 'Survived'])

df.to_csv(path_or_buf='gender_submission.csv', index=False)# index=Falseで行番号を出力しない

df_test_list = pd.DataFrame(outputArray)

#df_test_list.to_csv(path_or_buf='test.csv', index=False)

df_test_list
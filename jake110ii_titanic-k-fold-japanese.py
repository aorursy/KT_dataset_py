import numpy as  np

import pandas as pd #データ分析を効率的に行うためのライブラリ

import os #OSに依存しているさまざまな機能を利用するためのモジュール

import torch

import torch.nn as nn #ニューラルネットワークの構築

import torch.optim as optim #最適化手法

from torch.optim import lr_scheduler

from torch.autograd import Variable

# import torchvision

# from torchvision import datasets, models, transforms

# from torchvision.transforms import functional as F

import random #疑似乱数の生成、randomモジュールはMersenne Twisterアルゴリズムに基づく

import tensorflow as tf



import time

from tqdm import tqdm #走らせた処理の進捗状況をプログレスバーとして表示するためのパッケージ

from sklearn.model_selection import StratifiedKFold #データセットを分割

import tensorboardX as tb



import matplotlib.pyplot as plt

import seaborn as sns
#tensorboardでインスタンス化

writer = tb.SummaryWriter(logdir = './data_tensorboard')
for _ in tqdm(range(100)):

    time.sleep(0.01)
dic = {'name':'Taro', 'age':26}

print(dic['name']) #値tf.random.set_seedを取得する

def seed_everything(seed=1234):  #defoltではseed=1234, 指定されればその値

#     print(seed)

    random.seed(seed) #random.seed()を用いてseedを固定

    os.environ['PYTHONHASHSEED'] = str(seed) #環境変数をキーで取得（取得できない場合はエラー）

#     print(os.environ['PYTHONHASHSEED'])

    

#     tf.random.set_seed(seed)

    tf.set_random_seed(seed)  #tensorflowのバージョンで，上の文かこの文を分ける必要

    

    np.random.seed(seed)

    

    torch.manual_seed(seed) #pytorchでlossが毎回変わる問題の対処法

    

#     torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True #pytorchでlossが毎回変わる問題の対処法



kaeru_seed = 1337

seed_everything(seed=kaeru_seed)



batch_size = 32

train_epochs = 7
train = pd.read_csv('../titanic/train.csv') #PassengerId  Survived  Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked  

test = pd.read_csv('../titanic/test.csv')  #PassengerId                   Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked

target = train["Survived"] #インデックス[]を指定することで、行・列または要素の値を取得  最後に正解不正解を確認するため

print(type(train))

# print('target \n%s' %target)

# print(test)

print(train.shape)  

print(test.shape)  #testは，survivedの要素がないから一列少ない

train.head() 

train = pd.concat([train, test], sort=True) #指定した軸の方向にDataFrameを結合するPandasのconcat関数，　行方向にテストとtrainのデータを結合



print(train.shape) #結合した結果のshape

# train.head() #先頭の行の要素を返すメソッド

# test.head()

#print(train.head())

# print(train.iloc[:,2].unique())   #unique():

print(train.loc[:,'Embarked'].unique())   #unique():Embarked列の要素の重複しないデータセットを返す．　データの種類が['S' 'C' 'Q' nan]のどれか

# print(len(train.loc[:,'Name'].unique()) )

# print(train.loc[:,'Name'].unique()) 

# print(train.loc[:,'Name'].duplicated().value_counts())

# print(train.loc[:,'Name'].value_counts().index[1])

train[train.duplicated('Name',keep=False)].head()#名前が同じ人間のデータを表示．　同姓同名が二人いた！ってこと

# test.head()

# train.head() #先頭の行の要素を返すメソッド

def get_text_features(train):

    """trainデータに各個人の名前の長さのデータを追加

    """

    train['Length_Name'] = train['Name'].astype(str).map(len)

    return train



train = get_text_features(train)

train.head()  #データの最後の列に名前の長さのデータが追加されていることを確認
#categorical columnsとnumerical columnsを指定、cat_colsは数値でないものを指定している。

cat_cols = ['Cabin','Embarked','Name','Sex','Ticket']

num_cols = list(set(train.columns) - set(cat_cols) - set(["Survived"]))



print(set(train.columns))

print(cat_cols)

print(num_cols)
def encode(encoder, x):

    """名前のような数字ではないデータを通し番号を使って，数字化する．　要素に値がない（Nanの）場合は，except文に入るはず，，，

          encoderは辞書が送られてくる，　xはキー

    """

    len_encoder = len(encoder)

    try:

        id = encoder[x]  #x　:　key(辞書型の), encoder[x] : valueを返す

#         print('x=',x)

    except KeyError:

        print('except x=',x)

        id = len_encoder

    return id



# print(cat_cols)

encoders = [{} for cat in cat_cols]   #ここでは，リストのcat_colsと同じ形のリストを各要素を辞書形式として初期化

print(encoders)



# print(cat_cols)

for i, cat in enumerate(cat_cols):   #enumerateでi　にリストの中の要素のindexを代入, catにはリストの要素を代入

#     print('i=',i,', cat=',cat)

#     print('encoding %s ...' % cat, end=' ')#直下の文と同じ意味 

    print('encoding ',cat,' ...',end='')

    encoders[i] = {l: id for id, l in enumerate(train.loc[:, cat].astype(str).unique())}   #1列でfor文を使う書き方, uniqueにすることで重複しないデータのみを扱う

#     print('encoders[i]=',encoders[i])

#     print(train[cat],'---->',end='')

#     print(type(encoders[i]))

#     print(train[cat].astype(str))

    train[cat] = train[cat].astype(str).apply(lambda x: encode(encoders[i], x))

#     print(train[cat])

    print('Done')



embed_sizes = [len(encoder) for encoder in encoders]    #リストencodersの各要素のサイズをリストembed_sizesにまとめる

print(encoders[1])    #encordersはリストになっていて，indexが1番目の要素(Embarkedのデータの種類数)を表示．ここで4つだから，次の行で4ついなっていることが確認できる

print(embed_sizes)   #リストencodersの各要素のサイズをリストにまとめたもの

train.head()
from sklearn.preprocessing import StandardScaler #前処理用のライブラリ

 

Mean= train[num_cols].mean()

train[num_cols] = train[num_cols].fillna(Mean[num_cols]) #欠損値NaNをそれぞれの平均値で穴埋め



# Reblace NAN with age averages

# age_mean= train_df["Age"].mean()

# train_df["Age"] = train_df["Age"].fillna(age_mean)



print('scaling numerical columns')

scaler = StandardScaler()

# print(type(scaler))

train[num_cols] = scaler.fit_transform(train[num_cols])   #train[num_colsを標準化（規格化）して，データ自体の偏りを無くす

print(num_cols)

train[num_cols].head()
class CustomLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, p=0.5):  #p:ドロップアウトの割合, in_features:入力数 , out_features：出力数

        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias) #線形結合を計算するクラス

        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p) #過学習を防ぐ

        

    def forward(self, x):

        x = self.linear(x)

        x = self.relu(x)

        x = self.drop(x)

        return x
# net = nn.Sequential(CustomLinear(11, 32), nn.Linear(32, 1)) #ネットワークを定義  入力層：1，隠れ層：1，出力層：1構造12x32--->32x1
print(type(train))

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corrmat, vmax=1.0, square=True);
abs(corrmat["Survived"][1:]).plot(kind='bar',stacked=True, figsize=(10,5))
#ドロップ

train_d = train.drop(["Name","PassengerId","SibSp","Ticket","Embarked","Cabin"], axis=1)

test_d = test.drop(["Name","PassengerId","SibSp","Ticket","Embarked","Cabin"], axis=1)

train_d.head()
net = nn.Sequential(CustomLinear(train_d.shape[1]-1, 32), nn.Linear(32, 1)) #ネットワークを定義  入力層：1，隠れ層：1，出力層：1構造12x32--->32x1

print(train_d.shape[1]-1)
# print(train.loc[:,'Survived'])

X_train = train_d.loc[np.isfinite(train.Survived), :]  #isFiniteによって，Survivedに値が入っているものだけを取り出す. つまり，　元々の訓練データのみをX_trainに代入

# print(X_train)

print(type(X_train))        #ここでは，DataFrame

X_train = X_train.drop(["Survived"], axis=1).values   #axisはデータをドロップする方向指定．axis=1は列方向にSurvivedのデータを削除

print(type(X_train))        #.valuesのおかげで，　numpyの値に変わる

y_train = target.values   #targetにはSuvivedのデータが入っていて，valuesでその値だけを取り出し



X_test = train_d.loc[~np.isfinite(train.Survived), :]#~np.isFiniteによって，Survivedに値が入ってい”ない”ものだけを取り出す.

print(X_test.loc[:,'Survived'])

X_test = X_test.drop(["Survived"], axis=1).values



# X_train.head()
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=kaeru_seed).split(X_train, y_train))

print(X_train.shape)

print(type(splits))

print(len(splits))

print(splits[0][0].shape)   #元々のtrainを5分割したうちの4/5が訓練データとして使用. 891x4/5=712.8のint部分

print(splits[0][1].shape)   #元々のtrainを5分割したうちの1/5がテストデータとして使用.

print(891*4/5)
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
train_preds = np.zeros((len(X_train)))

test_preds = np.zeros((len(X_test)))

print(train_preds.shape)

print(test_preds.shape)



seed_everything(kaeru_seed) #上で定義した関数seed_everything



x_test_cuda = torch.tensor(X_test, dtype=torch.float32) #.cuda()

print('x_test_cuda.shape=',x_test_cuda.shape)



test = torch.utils.data.TensorDataset(x_test_cuda)

# print('test=',test[0])#0番目の人のデータを確認，12個のデータを出力

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

# print(batch_size)

print(type(test_loader))
#DataLoaderの確認

com=', '

for i in test_loader:  #test_loaderは要素がリスト. iに一つずつリストが代入

    print(type(i),com,len(i),com,i[0].shape)#リストiは要素一つのみで，その0番目がテンソルになってる．　テンソルの形はバッチ処理した形



print(32*13+2)   #下のデータの数
writer = tb.SummaryWriter(logdir = './data_tensorboard')

for i, (train_idx, valid_idx) in enumerate(splits): #Kfoldのループ部分

    """X_train, y_train, X_val, y_val をtensor化（PyTorchで扱える形に変換）し、

    .cuda()（GPUで計算するために特徴量をGPUに渡す処理）をする"""

    x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.float32) #.cuda()

    y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32) #.cuda()

    x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.float32) #.cuda()

    y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32) #.cuda()

#     print(x_train_fold.shape)

#     print(y_train_fold.shape)  #yは答え．　Survivedのデータ

#     print(x_val_fold.shape)

#     print(y_val_fold.shape)

#     print(891/5)

    

    model = net #modelを呼び出す

    #model.cuda() #modelもGPUに渡す

    

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum") #損失関数を呼び出す

    optimizer = torch.optim.Adam(model.parameters()) #深層学習における勾配法

    

    #dataloaderで扱える形（Dataset）にする

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)

    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    num_fold = f'Fold {i + 1}'

    print(f'Fold {i + 1}')

    

    #epoch分のループを回す

    train_epochs=100  #同じ3回でも精度にばらつきがある,  6回だと83％くらい

    print('train_epochs=',train_epochs)

    for epoch in range(train_epochs):

        start_time = time.time() #時刻を取得

        

        

        model.train() #modelをtrain modeにする 誤差を伝播させるモード

        avg_loss = 0.

        

        #x_train_foldとy_train_foldをbatch_size個ずつ渡すループ

        for x_batch, y_batch in tqdm(train_loader, disable=True):

            y_pred = model(x_batch) #predict

            loss = loss_fn(y_pred, y_batch) #lossの計算

            optimizer.zero_grad()

            loss.backward() #勾配を計算

            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        

        model.eval()  #誤差を伝播させないモード，　誤差関数で評価するのみ

        valid_preds_fold = np.zeros((x_val_fold.size(0)))

        test_preds_fold = np.zeros(len(X_test))

        avg_val_loss = 0.

        

        #X_test_foldをbatch_sizeずつ渡すループ

        for i, (x_batch, y_batch) in enumerate(valid_loader):

            y_pred = model(x_batch).detach() #勾配計算が必要ないとき

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)

            valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        

        elapsed_time = time.time() - start_time 

        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))

        writer.add_scalar('Fold{0}/loss/epoch'.format(i) ,avg_val_loss,epoch + 1)

        

    for i, (x_batch,) in enumerate(test_loader):

        y_pred = model(x_batch).detach()

        

        """batch_sizeのリストになっているのを単一階層のリストに変換して、cpuに値を渡し、

            テンソルからnumpy.array()に変換したものをsigmoid 関数に渡す"""

        test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]



    train_preds[valid_idx] = valid_preds_fold

    test_preds += test_preds_fold / len(splits) #予測値のkfold数（データセットの分割数）で割った値を加える



writer.close()
from sklearn.metrics import accuracy_score #正解率計算用のメソッド



def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.01 for i in range(100)]):

        score = accuracy_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

#             print(threshold)

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'accuracy_score': best_score}

    return search_result
search_result = threshold_search(y_train, train_preds)

search_result
sub = pd.read_csv('../titanic/gender_submission.csv')

sub.Survived = (test_preds > search_result['threshold']).astype(np.int8)

sub.to_csv('simple_nn_submission.csv', index=False) #pandasでcsvファイルの書き出し．　テストデータに対する答えを提出する用のファイル

sub.head()
print(sub)   #418人のテスト用データに対する，予測したSurvivedの値を追加したもの
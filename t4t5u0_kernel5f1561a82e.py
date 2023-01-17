import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset, TensorDataset

from tqdm import tqdm

import matplotlib.pyplot as plt
train = pd.read_csv(r"../input/digit-recognizer/train.csv", dtype=np.float32)



# split data into features(pixels) and labels(numbers from 0 to 9)

# ラベル(0-9)

targets_numpy = train.label.values

# 画像データセット (784x1に伸ばされている)

features_numpy = train.loc[:, train.columns != "label"].values/255



# 訓練用とテスト用にデータセットを分割

features_train, features_test, targets_train, targets_test = train_test_split(

    features_numpy, targets_numpy, test_size=0.2, random_state=42)



# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable

# NumPy配列からTensorに変換

featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long



# create feature and targets tensor for test set.

# NumPy配列からTensorに変換

featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long



# batch_size, epoch and iteration

# バッチサイズとエポック数を決定

batch_size = 100

n_iters = 10000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



# Pytorch train and test sets

# 画像とラベルの組をデータセット に変換

train = TensorDataset(featuresTrain,targetsTrain)

test = TensorDataset(featuresTest,targetsTest)



# data loader

# ミニバッチ用のデータローダーを作成

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)



# visualize one of the images in data set

# 画像を可視化

plt.imshow(features_numpy[10].reshape(28,28))

plt.axis("off")

plt.title(str(targets_numpy[10]))

plt.savefig('graph.png')

plt.show()
# 入力は28*28 の 784次元

input_dim = 28*28

# 出力は0-9 の10次元

output_dim = 10
# ネットワーク定義。今回は３層のMLP

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(MLP, self).__init__()

        # ネットワーク定義

        self.mlp = nn.Sequential(

            # 全結合層

            nn.Linear(input_dim, 256),

            # 活性化関数

            nn.ReLU(),

            # 全結合層

            nn.Linear(256, 256),

            # 活性化関数

            nn.ReLU(),

            # 全結合層

            nn.Linear(256, output_dim),

            # 活性化関数

            nn.Softmax()

        )

    def forward(self, x):

        return self.mlp(x)
net = MLP(input_dim, output_dim)

loss_fn = nn.CrossEntropyLoss()



lr = 0.001

optimizer = optim.Adam(net.parameters(), lr=lr)
# 評価関数

def eval_net(net, data_loader, device="cpu"):

    net.eval()

    labels = []

    labels_preds= []

    for image, label in data_loader:

        image = image.to(device)

        label = label.to(device)

        with torch.no_grad():

            _, label_pred = net(image).max(1)

        labels.append(label)

        labels_preds.append(label_pred)



    labels = torch.cat(labels)

    labels_preds = torch.cat(labels_preds)



    acc = (labels == labels_preds).float().sum() / len(labels)

    return acc
# 訓練用の関数

def train_net(net, train_loader, test_loader, optimizer, loss_fn, n_iters=10, device="cpu"):

    # 

    train_losses = []

    train_acc = []

    val_acc = []

    for epoch in range(n_iters):

        running_loss = 0.

        net.train()

        n = 0

        srore = 0



        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):

            images = images.to(device)

            labels = labels.to(device)

            label_pred = net(images)

            loss = loss_fn(label_pred, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            n += len(images)

        train_losses.append(running_loss / len(train_loader))

        # train_acc.append(n_a)

        val_acc.append(eval_net(net, test_loader, device))

        print(epoch, train_losses[-1], val_acc[-1], flush=True)
# データをGPUに渡す

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net.to(device)



# 訓練を実行

train_net(net, train_loader, test_loader, optimizer=optimizer, loss_fn=loss_fn, n_iters=20, device=device)

torch.save(net.state_dict(), r'./parameter.prm')
# 検証用データセットを作成

test = pd.read_csv(r"../input/digit-recognizer/test.csv", dtype=np.float32)



# データを正規化し、Tensorを作る

features_test = test.values/255

features_test = torch.from_numpy(features_test)

test = TensorDataset(features_test)



# データローダを作成

test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
# 推論用の関数

def prediction(data_loader, device='cpu'):

    net.load_state_dict(torch.load(r'./parameter.prm'))

    net.eval()

    test_pred = torch.LongTensor()



    for i, images in enumerate(data_loader):

        # print(images[0].size())

        images = images[0].to(device)

        output = net(images)

        _, pred = output.cpu().data.max(1, keepdim=True)

        test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred
# 推論を行う

test_pred = prediction(test_loader, device=device)



# データの整形

out_df = pd.DataFrame(np.c_[np.arange(1, len(test)+1)[:,None],

    test_pred.numpy()], columns=['ImageId', 'Label'])



# 出力

out_df.head()

out_df.to_csv('submission.csv', index=False)
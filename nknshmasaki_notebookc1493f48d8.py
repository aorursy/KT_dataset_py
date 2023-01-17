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
sample = pd.read_csv(r"/kaggle/input/digit-recognizer/sample_submission.csv",dtype = np.int32)

train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv",dtype = np.float32)

test = pd.read_csv(r"/kaggle/input/digit-recognizer/test.csv",dtype = np.float32)
print(train.shape)

train.head()
print(test.shape)

test.head()
print(sample.shape)

sample.head()
submissions=pd.DataFrame({"ImageId": list(range(1,len(test)+1)), "Label": 0})

submissions.to_csv("my_submission.csv", index=False, header=True)
import matplotlib.pyplot as plt

%matplotlib inline
X_train = (train.iloc[:,1:].values).astype('float32') # 2列目は画素値

y_train = train.iloc[:,0].values.astype('int32') # 1列目はラベルデータ

X_test = test.values.astype('float32')



print(X_train.shape)

print(y_train.shape)
# 28 × 28 の行列に画像を変換

X_train = X_train.reshape(X_train.shape[0], 28, 28)



plt.figure(figsize=(10,10))

for i in range(0, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i], size=21, color="white");
classes = np.unique(y_train)

print("Digit class = ",classes)



fig = plt.figure()

ax = fig.add_subplot(1,1,1)

for item in ax.get_xticklabels():

    item.set_color("white")

 

for item in ax.get_yticklabels():

    item.set_color("white")



ax.hist(y_train, bins=10)

fig.show()
# Import Libraries

import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import DataLoader

import pandas as pd

from sklearn.model_selection import train_test_split
# train と validation に分割

train_image, validation_image, train_label, validation_label = train_test_split(X_train,

                                                                             y_train,

                                                                             test_size = 0.2,

                                                                             random_state = 42) 
print(train_image.shape)

print(train_label.shape)
print(type(train_image))
train_image = torch.from_numpy(train_image)

print(type(train_image))
validation_image = torch.from_numpy(validation_image)

train_label = torch.from_numpy(train_label).type(torch.LongTensor)

validation_label = torch.from_numpy(validation_label).type(torch.LongTensor)
# バッチサイズとエポック数の定義

batch_size = 100

num_epochs = 30



# データローダにデータを読み込ませる

train = torch.utils.data.TensorDataset(train_image,train_label)

validation = torch.utils.data.TensorDataset(validation_image,validation_label)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)

validation_loader = DataLoader(validation, batch_size = batch_size, shuffle = False)
# モデルの作成

# ベクトル(input_dim) -> ベクトル(100) -> ReLU -> ベクトル(output_dim) -> softmax

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LogisticRegressionModel, self).__init__()



        self.linear1 = nn.Linear(input_dim, 100)

        self.activation1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(100, output_dim)

        self.activation2 = nn.Softmax(dim=1)

    

    def forward(self, x):

        x = self.linear1(x)

        x = self.activation1(x)

        x = self.linear2(x)

        out = self.activation2(x)

        return out
# モデルの読み込み

input_dim = 28*28 # 入力サイズの指定: 長さ784(28×28)のベクトル

output_dim = 10  # 出力サイズの指定: 長さ10(ラベル数)のベクトル

model = LogisticRegressionModel(input_dim, output_dim)
# 損失関数の定義(cross entropy)

criterion = nn.CrossEntropyLoss()



# オプティマイザの定義(今回はSGD)

learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
import time



# 学習過程でロスと精度を保持するリスト

train_losses, val_losses = [], []

train_accu, val_accu = [], []



for epoch in range(num_epochs):

    epoch_start_time = time.time()



    # 学習用データで学習

    train_loss = 0

    correct=0

    model.train()

    for images, labels in train_loader:

        # 勾配の初期化

        optimizer.zero_grad()

        

        # 順伝播

        outputs = model(images.view(-1, 28*28))



        # ロスの計算と逆伝播

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        

        # 正答数を計算

        predicted = torch.max(outputs.data, 1)[1]

        correct += (predicted == labels).sum()

        

        train_loss += loss.item()



    train_losses.append(train_loss/len(train_loader))

    train_accu.append(correct/len(train_loader))



    # 検証用データでlossと精度の計算

    val_loss = 0

    correct = 0

    model.eval()

    with torch.no_grad():

        for images, labels in validation_loader: 

            # 順伝播

            outputs = model(images.view(-1, 28*28))

            

            # ロスの計算

            val_loss += criterion(outputs, labels)



            # 正答数を計算

            predicted = torch.max(outputs.data, 1)[1]

            correct += (predicted == labels).sum()



    val_losses.append(val_loss / len(validation_loader))

    val_accu.append(correct / len(validation_loader))



    print(f"Epoch: {epoch+1}/{num_epochs}.. ",

          f"Time: {time.time()-epoch_start_time:.2f}s..",

          f"Training Loss: {train_losses[-1]:.3f}.. ",

          f"Training Accu: {train_accu[-1]:.3f}.. ",

          f"Val Loss: {val_losses[-1]:.3f}.. ",

          f"Val Accu: {val_accu[-1]:.3f}")

# loss と accuracy の可視化

plt.figure(figsize=(12,12))

plt.subplot(2,1,1)

ax = plt.gca()

ax.set_xlim([0, epoch + 2])

plt.ylabel('Loss')

plt.plot(range(1, epoch + 2), train_losses[:epoch+1], 'r', label='Training Loss')

plt.plot(range(1, epoch + 2), val_losses[:epoch+1], 'b', label='Validation Loss')

ax.grid(linestyle='-.')

plt.legend()

plt.subplot(2,1,2)

ax = plt.gca()

ax.set_xlim([0, epoch+2])

plt.ylabel('Accuracy')

plt.plot(range(1, epoch + 2), train_accu[:epoch+1], 'r', label='Training Accuracy')

plt.plot(range(1, epoch + 2), val_accu[:epoch+1], 'b', label='Validation Accuracy')

ax.grid(linestyle='-.')

plt.legend()

plt.show()
# テストデータを使って回答を作る

X_test = torch.from_numpy(X_test)

model.eval()

output = model(X_test)

prediction = torch.argmax(output, 1)
# 回答データの出力

submissions = pd.DataFrame({"ImageId": list(range(1,len(test)+1)), "Label": prediction.cpu().tolist()})

submissions.to_csv("my_submission.csv", index=False, header=True)
# device = torch.device("cuda")



# # to でGPUを利用するように指定

# model_gpu = LogisticRegressionModel(input_dim, output_dim)

# model_gpu.to(device)



# optimizer = torch.optim.SGD(model_gpu.parameters(), lr=learning_rate)



# import time

# # 学習過程でロスと精度を保持するリスト

# train_losses, val_losses = [], []

# train_accu, val_accu = [], []



# for epoch in range(num_epochs):

#     epoch_start_time = time.time()



#     # 学習用データで学習

#     train_loss = 0

#     correct=0

#     model_gpu.train()

#     for images, labels in train_loader:

#         # to でGPUを利用するように指定

#         images = images.to(device)

#         labels = labels.to(device)



#         # 勾配の初期化

#         optimizer.zero_grad()

        

#         # 順伝播

#         outputs = model_gpu(images.view(-1, 28*28))



#         # ロスの計算と逆伝播

#         loss = criterion(outputs, labels)

#         loss.backward()

#         optimizer.step()

        

#         # 正答数を計算

#         predicted = torch.max(outputs.data, 1)[1]

#         correct += (predicted == labels).sum()

        

#         train_loss += loss.item()



#     train_losses.append(train_loss/len(train_loader))

#     train_accu.append(correct/len(train_loader))



#     # 検証用データでlossと精度の計算

#     val_loss = 0

#     correct = 0

#     model_gpu.eval()

#     with torch.no_grad():

#         for images, labels in validation_loader: 

#             # to でGPUを利用するように指定

#             images = images.to(device)

#             labels = labels.to(device)



#             # 順伝播

#             outputs = model_gpu(images.view(-1, 28*28))

            

#             # ロスの計算

#             val_loss += criterion(outputs, labels)



#             # 正答数を計算

#             predicted = torch.max(outputs.data, 1)[1]

#             correct += (predicted == labels).sum()



#     val_losses.append(val_loss / len(validation_loader))

#     val_accu.append(correct / len(validation_loader))



#     print(f"Epoch: {epoch+1}/{num_epochs}.. ",

#           f"Time: {time.time()-epoch_start_time:.2f}s..",

#           f"Training Loss: {train_losses[-1]:.3f}.. ",

#           f"Training Accu: {train_accu[-1]:.3f}.. ",

#           f"Val Loss: {val_losses[-1]:.3f}.. ",

#           f"Val Accu: {val_accu[-1]:.3f}")

import torch

import torch.nn as nn

import torch.nn.functional as F
LEARNING_RATE = 0.01

BATCH_SIZE = 32

EPOCHS = 10
# ここでは，28 * 28の白黒画像(グレースケール)を入力すると仮定して，10クラスの分類を行うものとします



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # 入力チャンネル数1，出力チャンネル数6, kernel size 5のCNNを定義する

        self.conv1 = nn.Conv2d(1, 6, 5)

        # 次に，線形変換を行う層を定義してあげます: y = Wx + b

        # nn.Linearは fully-connected layer (全結合層)のことです．

        # self.conv1のあと，maxpoolingを通すことで，

        # self.fc1に入力されるTensorの次元は 6 * 12 * 12 (Channel, height, width) になっています．

        # これを10クラス分類なので，10次元に変換するようなLinear層を定義します

        self.fc1 = nn.Linear(6 * 12 * 12, 10)



    

    def forward(self, x):

        batch_size = x.shape[0]

        # forward関数の中では，，入力 x を順番にレイヤーに通していきます．みていきましょう．    

        # まずは，画像をCNNに通します

        x = self.conv1(x)



        # 活性化関数としてreluを使います

        x = F.relu(x)

        

        # 次に，MaxPoolingをかけます．

        x = F.max_pool2d(x, (2, 2))

        

        # 少しトリッキーなことが起きます．

        # CNNの出力結果を fully-connected layer に入力するために

        # 1次元のベクトルにしてやる必要があります

        # 正確には，　(batch_size, channel, height, width) --> (batch_size, channel * height * width)

        x = x.view(batch_size, -1)

        

        # linearと活性化関数に通します

        x = self.fc1(x)

        x = F.relu(x)

        return x
net = Net()
print(net)
sample_input = torch.randn(1, 1, 28, 28)



print(sample_input)

print(sample_input.shape)  # 次元をみてみよう，正しく生成されていますね
sample_output = net(sample_input)



# Q.これは何を表しているんでしたっけ？

print(sample_output.shape)
sample_target = torch.randn(1, 10)
criterion = nn.MSELoss()
sample_loss = criterion(sample_output, sample_target)

print(sample_loss)
net.zero_grad()



# ここで backpropagate が実行されます．

sample_loss.backward()
# optimizerを用意します

optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)



# epochループを回します．

# [発展] このループの内側で，実際には Dataloader を使用した iteration ループを記述することになりますが，今回は割愛します．

for epoch in range(EPOCHS):

    optimizer.zero_grad()  # さっきの net.zero_grad()の代わりにこれで gradient をゼロにする

    sample_output = net(sample_input) # 順伝播を行う

    loss = criterion(sample_output, sample_target) # 出力と正解を比較して損失の計算

    print(f'Loss value: {loss.item()}')

    loss.backward()  # 逆伝播の計算

    optimizer.step()  # 逆伝播の結果からモデルを更新する
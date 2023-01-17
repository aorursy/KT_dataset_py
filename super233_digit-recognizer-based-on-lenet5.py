%matplotlib inline
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

import torch.nn as nn

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import torchvision
# 因为是图像数据，数据集中值的范围是[0, 255]，所以这里设置数据类型为uint8类型

df = pd.read_csv('../input/digit-recognizer/train.csv', dtype='uint8')

df.head()
# 划分训练集和测试集，比例默认为0.75和0.25

train_df, test_df = train_test_split(df, shuffle=True)
temp_data = df.iloc[3]

image = temp_data.to_numpy()[1:].reshape((28, 28))

plt.imshow(image)

plt.show()
class DigitDataset(Dataset):

    """将官方提供的CSV数据封装为适合Pytorch编写的NN使用的数据集类"""

    def __init__(self, df, with_label, transform=None):

        """ df：DataFrame类型的数据

            with_label：是否具有label，用于分辨train.csv和test.csv的数据

            transform：表示需要进行的图像变换

        """



        super().__init__()

        self.df = df

        self.transform = transform

        self.with_label = with_label



    def __len__(self):

        """返回整个DaTaset的大小"""



        return len(self.df)



    def __getitem__(self, idx):

        """对Dataset中第idx行的数据进行数据预处理，返回处理后的数据和其label"""



        # 取出第i行数据

        temp_data = self.df.iloc[idx].to_numpy()



        # 如果数据集有label，则返回label

        if self.with_label is True:

            # 第0个数据是label

            target = int(temp_data[0])

            # 将图像的数据转换成一个w*h*c的3维数组

            image = temp_data[1:].reshape((28, 28, 1))

            # 对每个数据进行预处理转换

            if self.transform is not None:

                image = self.transform(image)



            return image, target

        else:

            # 将图像的数据转换成一个w*h*c的3维数组

            image = temp_data.reshape((28, 28, 1))

            # 对每个数据进行预处理转换

            if self.transform is not None:

                image = self.transform(image)



            return image
# 对每个数据进行Resize和ToTensor操作

temp_data_loader = DataLoader(dataset=DigitDataset(train_df, True, transforms.Compose([transforms.ToPILImage(),

                                                                                 transforms.Resize((32, 32)),

                                                                                 transforms.ToTensor()])), 

                               batch_size=64, 

                               shuffle=True)

temp_data_iter = iter(temp_data_loader)
images, targets = next(temp_data_iter)
targets
# 这里的数据已经是Tensor格式

images.shape
# 将64张图像拼在一起

grid_image = torchvision.utils.make_grid(images)

grid_image.shape
# plt绘制图像的时候，要求数据shape为[H, W, C](RGB)或者[H, W](BOOL)，所以需要将图像的shape调一下

plt.imshow(np.transpose(grid_image, [1, 2, 0]))

plt.show()
images.shape
plt.imshow(images[0][0])

plt.show()
class LeNet5(nn.Module):

    

    def __init__(self):

        super().__init__()



        # 定义LeNet5的结构

        # 1,32*32 --> 6,28*28 --> 6,14*14

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),

            nn.ReLU(), 

            nn.MaxPool2d(kernel_size=(2, 2))

        )



        # 6,14*14 --> 16,10*10 --> 16,5*5

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),

            nn.ReLU(), 

            nn.MaxPool2d(kernel_size=(2, 2))

        )



        # 16,5*5 --> 120

        self.fc1 = nn.Sequential(

            nn.Linear(in_features=16 * 5 * 5, out_features=120), 

            nn.ReLU()

        )



        # 120 --> 84

        self.fc2 = nn.Sequential(

            nn.Linear(in_features=120, out_features=84),

            nn.ReLU()

        )



        # 84 --> 10

        self.fc3 = nn.Linear(in_features=84, out_features=10)



    def forward(self, x):

        """描述输入的数据在网路内部的前向传递过程"""

        

        x = self.conv1(x)

        x = self.conv2(x)

        # 在输入FC层之前，将数据拉伸为一行

        x = x.view(-1, self.get_flat_features_num(x))

        x = self.fc1(x)

        x = self.fc2(x)

        x = self.fc3(x)



        return x



    def get_flat_features_num(self, x):

        """返回将张量拉伸为行向量时的总的特征数"""

        

        num = 1

        for s in x.size()[1:]:

            num *= s



        return num
# 超参数

BATCH_SIZE = 512

EPOCHS = 50

LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(DEVICE)

# 使用交叉熵作为多分类的损失函数，使用Adam作为损失的优化函数

loss_fun = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# 初始化训练和测试数据的dataloader，将数据Resize为32*32

train_data_loader = DataLoader(dataset=DigitDataset(train_df, 

                                                    True, 

                                                    transforms.Compose([transforms.ToPILImage(),

                                                                        transforms.Resize((32, 32)),

                                                                        transforms.ToTensor(),

                                                                        transforms.Normalize((0.1307,), (0.3081,))])),

                               batch_size=BATCH_SIZE,

                               shuffle=True)



test_data_loader = DataLoader(dataset=DigitDataset(test_df, 

                                                   True, 

                                                   transforms.Compose([transforms.ToPILImage(),

                                                                       transforms.Resize((32, 32)),

                                                                       transforms.ToTensor(),

                                                                       transforms.Normalize((0.1307,), (0.3081,))])),

                              batch_size=BATCH_SIZE,

                              shuffle=True)
%%time

for epoch in range(EPOCHS):

    # Train

    # 开启模型的train模式，启用BatchNormalization和Dropout

    model.train()

    for batch_index, (data, target) in enumerate(train_data_loader):

        # 将数据移动到设备上

        data, target = data.to(DEVICE), target.to(DEVICE)



        # 每次数据数据前，先清空所有参数的梯度信息

        optimizer.zero_grad()

        output = model(data)

        loss = loss_fun(output, target)

        # 误差反向传递，计算每个参数的梯度

        loss.backward()

        # 更新每个参数

        optimizer.step()



        if batch_index % 30 == 0:

            print('Train Epoch: {} [{}/{} {:.2f}%]\t Loss: {:.5f}'.format(

                epoch, batch_index, len(train_data_loader), 100 * batch_index / len(train_data_loader), loss.item()))

            

    # Test

    # 开启模型的evaluate模式，启用BatchNormalization和Dropout

    model.eval()

    test_loss = 0

    test_correct_num = 0

    # 测试时，不计算梯度

    with torch.no_grad():

        for data, target in test_data_loader:

            data, target = data.to(DEVICE), target.to(DEVICE)

            

            output = model(data)

            test_loss += loss_fun(output, target).item()

            # 获得每个数据的预测的类别

            pred = output.max(dim=1, keepdim=True)[1]

            test_correct_num += pred.eq(target.view_as(pred)).sum().item()        

        

        average_loss = test_loss / len(test_data_loader.dataset)

        average_accuracy = test_correct_num / len(test_data_loader.dataset)

            

        print('Test: Average Loss: {:.5f}\t Accuracy: {}/{} ({:.3f})\n'.format(

            average_loss, test_correct_num, len(test_data_loader.dataset), average_accuracy))
torch.save(model, 'LeNet5.pkl')
# 加载训练好的模型

model = torch.load('LeNet5.pkl')
eval_df = pd.read_csv('../input/digit-recognizer/test.csv', dtype='uint8')

eval_df.head()
eval_data_loader = DataLoader(dataset=DigitDataset(eval_df, 

                                                   False, 

                                                   transforms.Compose([transforms.ToPILImage(),

                                                                       transforms.Resize((32, 32)),

                                                                       transforms.ToTensor(),

                                                                       transforms.Normalize((0.1307,), (0.3081,))])),

                              batch_size=BATCH_SIZE)
images = iter(eval_data_loader).next()

grid_image = torchvision.utils.make_grid(images)



plt.imshow(np.transpose(grid_image, [1, 2, 0]))

plt.show()
%%time

# 记录没批数据预测的类别

pred_label = torch.tensor([], dtype=torch.int64).to(DEVICE)

# 不训练时，记得使用no_grad与model.eval()

with torch.no_grad():

    model.eval()

    for batch_index, data in enumerate(eval_data_loader):

        data = data.to(DEVICE)



        output = model(data)

        # 获得该批数据最后输出的最大的分数所对应的类别

        batch_pred_label = output.max(dim=1, keepdim=False)[1]

        pred_label = torch.cat([pred_label, batch_pred_label])



        if batch_index % 5 == 0:

            print('Eval: {}/{} has finished!'.format(batch_index, len(eval_data_loader)))

    print('OK!')
pd.read_csv('../input/digit-recognizer/sample_submission.csv').head()
ImageId = np.arange(1, len(pred_label) + 1)

Label = pred_label.cpu().numpy()
pred_df = pd.DataFrame({'ImageId': ImageId, 'Label': Label})

pred_df.head()
pred_df.to_csv('submission1.csv', index=False)
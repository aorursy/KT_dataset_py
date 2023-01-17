!wget https://labfile.oss.aliyuncs.com/courses/2534/cifar-10-python.tar.gz
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# 定义预处理列表
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 张 32x32 大小的彩色图片，这些图片共分 10 类,每类有 6000 张图像
# root:指定数据集所在位置
# train=True：表示若本地已经存在，无需下载。若不存在，则下载
# transform：预处理列表，这样就可以返回预处理后的数据集合
train_dataset = torchvision.datasets.CIFAR10(root='./', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./', train=False,
                                            download=True, transform=transform)
print("训练集的图像数量为：", len(train_dataset))
print("测试集的图像数量为", len(test_dataset))
batch_size = 256      # 设置批次个数
# shuffle=True:表示加载数据前，会先打乱数据，提高模型的稳健性
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
test_loader, test_loader
import matplotlib.pyplot as plt
%matplotlib inline


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img):
    # 由于加载器产生的图片是归一化后的图片，因此这里需要将图片反归一化
    # 变成归一化前的图像
    img = img / 2 + 0.5
    # 将图像从 Tensor 转为 Numpy
    npimg = img.numpy()
    #产生的数据为 C×W×H 而 plt 展示的图像一般都是 W×H×C
    #因此，这里会有一个维度的变换
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获得一些训练图像
dataiter = iter(train_loader)
images, labels = dataiter.next()

# 将这些图像进行展示
imshow(torchvision.utils.make_grid(images))
import torch.nn.functional as F
import torch.nn as nn

#网络模型的建立
'''定义网络模型'''
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x) 
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# 定义当前设备是否支持 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16().to(device)
model
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion,optimizer
import datetime

starttime = datetime.datetime.now()
num_epochs = 20
# 定义数据长度
n_total_steps = len(train_loader)
print("Start training....")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 原始数据集的大小，每个批次的大小为: [4, 3, 32, 32] 
        # 将数据转为模型支持的环境类型。
        images = images.to(device)
        labels = labels.to(device)

        # 模型的正向传播，得到数据数据的预测值
        outputs = model(images)
        # 根据预测值计算损失
        loss = criterion(outputs, labels)

        # 固定步骤：梯度清空、反向传播、参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)
print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
print("The model have been saved！")
new_model = VGG16().to(device)
new_model.load_state_dict(torch.load(PATH))
with torch.no_grad():
    # 统计预测正确的图像数量和进行了预测的图像数量
    n_correct = 0
    n_samples = 0
    # 统计每类图像中，预测正确的图像数量和该类图像的实际数量
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = new_model(images)
        # 利用 max 函数返回 10 个类别中概率最大的下标，即预测的类别
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        # 通过判断预测值和真实标签是否相同，来统计预测正确的样本数
        n_correct += (predicted == labels).sum().item()
        # 计算每种种类的预测正确数
        if len(labels)<batch_size:
            batch_size=len(labels)
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    # 输出总的模型准确率
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    # 输出每个类别的模型准确率
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

!wget https://labfile.oss.aliyuncs.com/courses/2534/cat2.jpg
## 图片的加载函数
from PIL import Image

infer_path='cat2.jpg'
img = Image.open(infer_path)
plt.imshow(img)   
plt.show()  
def load_image(file):
    im = Image.open(file)
    # 将大小修改为 32*32 符合模型输出
    im = im.resize((32,32),Image.ANTIALIAS)
    # 建立图片矩阵
    im = np.array(im).astype(np.float32)
    ## WHC -> CHW
    im = im.transpose((2,0,1))
    im = im / 255.0
    # 转为 batch,c,w,h
    im = np.expand_dims(im,axis=0)

    print("im_shape 的维度",im.shape)
    return im
#加载如何模型输入的图像
img = load_image(infer_path)


img = load_image(infer_path)
img = torch.from_numpy(img)
prediction = model(img.to(device))
print("The picture is a ", classes[np.argmax(prediction.cpu().detach().numpy()[0])])
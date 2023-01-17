# 导入所需模块

import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy
# 定义基本参数

data_dir = '../input/chest-xray-pneumonia/chest_xray'

model_name = 'vgg'

num_classes = 2

batch_size = 16

num_epochs = 10

input_size = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 进行一系列数据增强，然后生成训练、验证、和测试数据集

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(input_size),                            # 随机长宽比裁剪，最后将图片resize到设定好的input_size

        transforms.RandomHorizontalFlip(),                                   # 随机水平翻转

        transforms.ToTensor(),                                               # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # 对tensor image按通道进行标准化，即先减均值，再除以标准差

    ]),

    'val': transforms.Compose([

        transforms.Resize(input_size),

        transforms.CenterCrop(input_size),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test': transforms.Compose([

        transforms.Resize(input_size),

        transforms.CenterCrop(input_size),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

}



print("Initializing Datasets and Dataloaders...")



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}



dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}



classes = image_datasets['test'].classes
# 定义一个查看图片和标签的函数

def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)



imgs, labels = next(iter(dataloaders_dict['train']))



out = torchvision.utils.make_grid(imgs[:8])



imshow(out, title=[classes[x] for x in labels[:8]])
# 建立VGG16迁移学习模型

# 若pretrained=True，则会下载预训练权重，需要耐心等待一段时间

model = torchvision.models.vgg16(pretrained=True)

model
# 将模型参数设置为不可更新

for param in model.parameters():

    param.requires_grad = False



# 再更改最后一层的输出，至此网络只能更新该层参数

model.classifier[6] = nn.Linear(4096, num_classes)
def train_model(model, dataloaders, criterion, optimizer, mun_epochs=25):

    since = time.time()

    

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs-1))

        print('-' * 10)

        

        for phase in ['train', 'val']:

            

            if phase == 'train':

                model.train()

            else:

                model.eval()

            

            running_loss = 0.0

            running_corrects = 0.0

            

            for inputs, labels in dataloaders[phase]:

                inputs, labels = inputs.to(device), labels.to(device)

                

                optimizer.zero_grad()

                

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)



                    _, preds = torch.max(outputs, 1)



                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            

            print('{} loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

    

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:.4f}'.format(best_acc))

    

    model.load_state_dict(best_model_wts)

    return model
# 定义优化器和损失函数

model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()
# 开始训练

model_ft = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs)
# 测试集评估

imgs, labels = next(iter(dataloaders_dict['test']))

imgs, labels = imgs.to(device), labels.to(device)

outputs = model_ft(imgs)

_, preds = torch.max(outputs, 1)

print('real：' + ' '.join('%9s' % classes[labels[j]] for j in range(10)))

print('pred：' + ' '.join('%9s' % classes[preds[j]] for j in range(10)))
# 全部测试集中的准确率

correct = 0.0

for imgs, labels in dataloaders_dict['test']:

    imgs, labels = imgs.to(device), labels.to(device)

    output = model_ft(imgs)

    _, preds = torch.max(output, 1)

    correct += (preds == labels).sum().item()

print('test accuracy:{:.2f}%'.format(100 * correct / len(dataloaders_dict['test'].dataset)))
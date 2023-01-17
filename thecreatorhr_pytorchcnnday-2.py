%matplotlib inline
import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import warnings



warnings.filterwarnings('ignore')
img = Image.open('../input/image1/img.jpg').convert('L')

plt.imshow(img,'gray')
# 2D kernel

weight = torch.tensor([[[[1,0,-1],[1,0,-1],[1,0,-1]]]]).float()
# 3D kernel

weight = torch.tensor([[[[1,0,-1],[1,0,-1],[1,0,-1]], 

                        [[1,0,-1],[1,0,-1],[1,0,-1]], 

                        [[1,0,-1],[1,0,-1],[1,0,-1]]]]).float()
weight = torch.tensor([[[[1,0,-1],[2,0,-2],[1,0,-1]]]]).float()
weight = torch.tensor([[[[1,2,1],[0,0,0],[-1,-2,-1]]]]).float()
tf = transforms.ToTensor()

img = tf(img)

img = img.unsqueeze_(0)

print(img.shape)
op = F.conv2d(img, weight, bias=None, stride=1, padding=0, dilation=1)
op = op.squeeze_(0)

tf_ = transforms.ToPILImage()

op = tf_(op).convert('L')

plt.imshow(op,'gray')
op = F.conv2d(img, weight, bias=None, stride=1, padding=1000, dilation=1)
op = op.squeeze_(0)

tf_ = transforms.ToPILImage()

op = tf_(op).convert('L')

plt.imshow(op,'gray')
op = F.conv2d(img, weight, bias=None, stride=2, padding=0, dilation=1)
op = op.squeeze_(0)

tf_ = transforms.ToPILImage()

op = tf_(op).convert('L')

plt.imshow(op,'gray')
img = Image.open('../input/image1/img.jpg')

plt.imshow(img)
im = np.asarray(img)

im
im.shape
plt.imshow(im[:,:,0], 'gray')
plt.imshow(im[:,:,0], 'Reds')
plt.imshow(im[:,:,1], 'Greens')
plt.imshow(im[:,:,2], 'Blues')
im_r = im[:,:,0]

im_g = im[:,:,1]

im_b = im[:,:,2] * 2

im_r = np.reshape(im_r, (2322,4128,1))

im_g = np.reshape(im_g, (2322,4128,1))

im_b = np.reshape(im_b, (2322,4128,1))

img = np.concatenate((im_r,im_g,im_b),axis=2)

plt.imshow(img)
inp = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=np.float32)

inp
inp = torch.from_numpy(inp)

op_ = F.max_pool2d(inp.unsqueeze_(0), kernel_size=2,stride=2)

op_ = op_.squeeze_(0)

print(np.asarray(op_))
op1 = F.max_pool2d(tf(op).unsqueeze_(0), kernel_size=3,stride=2)

op1 = op1.squeeze_(0)

op1 = tf_(op1).convert('L')

plt.imshow(op1,'gray')
op_ = F.avg_pool2d(inp.unsqueeze_(0), kernel_size=2,stride=2)

op_ = op_.squeeze_(0)

print(np.asarray(op_))
op2 = F.avg_pool2d(tf(op).unsqueeze_(0), kernel_size=3,stride=2)

op2 = op2.squeeze_(0)

op2 = tf_(op2).convert('L')

plt.imshow(op2,'gray')
# CUDA for PyTorch

from torch.backends import cudnn



use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

cudnn.benchmark = True
trainset = torchvision.datasets.FashionMNIST('./data', train=True, transform=transforms.ToTensor(), download=True)

testset = torchvision.datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
#loading the training data from trainset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle = True)

#loading the test data from testset

testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
class_labels = ['T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

fig = plt.figure(figsize=(15,8));

columns = 5;

rows = 3;

for i in range(1, columns*rows +1):

    index = np.random.randint(len(trainset))

    img = trainset[index][0][0, :, :]

    fig.add_subplot(rows, columns, i)

    plt.title(class_labels[trainset[index][1]])

    plt.axis('off')

    plt.imshow(img, cmap='gray')

plt.show()
class Basic_conv(nn.Module):

    def __init__(self):

        super(Basic_conv, self).__init__()

        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride=1,padding=0)

        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=0)

        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=0)

        self.fc1 = nn.Linear(15488, 64, bias=True)

        self.fc2 = nn.Linear(64, 10, bias=True)

        self.drop = nn.Dropout(0.5)

        # or nn.ReLU()

        

    def forward(self, x):

        x = self.conv1(x)

        x = torch.relu(x)

        x = self.pool(x)

        x = torch.relu(x)

        x = self.conv2(x)

        x = torch.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.drop(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x
model = Basic_conv()

if use_cuda:

    model.to(device)

print(model)



cross_entropy_loss = nn.CrossEntropyLoss()



adam_optim = torch.optim.Adam(model.parameters(), lr=0.005)

print(adam_optim)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



def eval_perfomance(model, dataloader, print_results=False):

    actual, preds = [], []

    #keeping the network in evaluation mode  

    model.eval() 

    for data in dataloader:

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        actual +=[i.item() for i in labels]

        

        #moving the inputs and labels to gpu

        outputs = model(inputs)

        _, pred = torch.max(outputs.data, 1)

        preds += [i.item() for i in pred]

    acc = accuracy_score(actual, preds)

    cm = confusion_matrix(actual, preds)

    cr = classification_report(actual, preds)

    if(print_results):

        print(f'Total accuracy = {acc*100}%')

        print('\n\nConfusion matrix:\n')

        print(cm)

        print('\n\nClassification Report:\n')

        print(cr)

    

    return acc
intial_acc = eval_perfomance(model, testloader, True)
# %%time

loss_arr = []

loss_epoch_arr = []

max_epochs = 5

iter_list = []

train_acc_arr = []

test_acc_arr = []

ctr = 0

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader, 0):

        

        model.train()

        images, labels = data

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)      

        

        loss = cross_entropy_loss(outputs, labels)    

        adam_optim.zero_grad()     



        loss.backward()     

        adam_optim.step()     

        loss_arr.append(loss.item())

        

        ctr+=1

        iter_list+=[ctr]

    

        

    train_acc = eval_perfomance(model, trainloader)

    train_acc_arr+=[train_acc.item()]

        

    test_acc = eval_perfomance(model, testloader)

    test_acc_arr+=[test_acc.item()]

        

    if((epoch+1)%1==0):

        print(f"Iteration: {epoch+1}, Loss: {loss.item()}, Train Acc:{train_acc}, Val Acc: {test_acc}")



    loss_epoch_arr.append(loss.item()) 
plt.plot([i for i in range(len(loss_epoch_arr))], loss_epoch_arr)

plt.xlabel("No. of Iteration")

plt.ylabel("Loss")

plt.title("Loss vs Iterations")

plt.show()
plt.plot([i for i in range(len(test_acc_arr))], test_acc_arr)

plt.xlabel("No. of Iteration")

plt.ylabel("Test Accuracy")

plt.title("Test Accuracy")

plt.show()
class_correct = [0. for _ in range(10)]

total_correct = [0. for _ in range(10)]



with torch.no_grad():

    for images, act_labels in testloader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        predicted = torch.max(outputs, 1)[1]

        if use_cuda:

            c = (predicted == act_labels.cuda()).squeeze()

        else:

            c = (predicted == act_labels).squeeze()

        

        for i in range(4):

            label = act_labels[i]

            class_correct[label] += c[i].item()

            total_correct[label] += 1

        

for i in range(10):

    print("Accuracy of {}: {:.2f}%".format(class_labels[i], class_correct[i] * 100 / total_correct[i]))
final_acc_test = eval_perfomance(model, testloader, print_results=True)
def get_n_params(model):

    pp=0

    for p in list(model.parameters()):

        nn=1

        for s in list(p.size()):

            nn = nn*s

        pp += nn

    return pp

print(get_n_params(model))
resnet = torchvision.models.resnet34(pretrained=True, progress = True)

resnet
for name, param in resnet.named_parameters():

    print(name, param.requires_grad)
for name, param in resnet.named_parameters():

    if name=='fc.weight'or name == 'fc.bias':

        break

    param.requires_grad = False
resnet.fc = nn.Linear(in_features=512, out_features=10, bias=True)

resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
for name, param in resnet.named_parameters():

    print(name, param.requires_grad)
if use_cuda:

    resnet.to(device)



cross_entropy_loss = nn.CrossEntropyLoss()



adam_optim = torch.optim.Adam(resnet.parameters(), lr=0.005)

print(adam_optim)
intial_acc = eval_perfomance(resnet, testloader, True)
# %%time

loss_arr = []

loss_epoch_arr = []

max_epochs = 3

iter_list = []

train_acc_arr = []

test_acc_arr = []

ctr = 0

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader, 0):

        

        model.train()

        images, labels = data

        images, labels = images.to(device), labels.to(device)

        outputs = resnet(images)      

        

        loss = cross_entropy_loss(outputs, labels)    

        adam_optim.zero_grad()     



        loss.backward()     

        adam_optim.step()     

        loss_arr.append(loss.item())

        

        ctr+=1

        iter_list+=[ctr]

    

        

    train_acc = eval_perfomance(resnet, trainloader)

    train_acc_arr+=[train_acc.item()]

        

    test_acc = eval_perfomance(resnet, testloader)

    test_acc_arr+=[test_acc.item()]

        

    if((epoch+1)%1==0):

        print(f"Iteration: {epoch+1}, Loss: {loss.item()}, Train Acc:{train_acc}, Val Acc: {test_acc}")



    loss_epoch_arr.append(loss.item()) 
intial_acc = eval_perfomance(resnet, testloader, True)
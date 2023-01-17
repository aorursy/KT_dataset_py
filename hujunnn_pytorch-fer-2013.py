# import os

# os.listdir('../input/')
import time

import matplotlib.pyplot as plt

import cv2 as cv

from math import sqrt 

import pandas as pd

import numpy as np

from torchvision import transforms as tfs

import torch

from PIL import Image
time_start = time.time()
datas = pd.read_csv("../input/fer2013/fer2013.csv")

datas
lab = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neural']

labels_num = datas.emotion.value_counts()

la = [0,1,2,3,4,5,6]

la_num = [labels_num[i] for i in range(len(labels_num))]

print(labels_num)

plt.bar(range(len(la_num)), la_num,color='rgbc',tick_label=lab)  #plt.barh则是把该图变成横向的  #3fa4ff

for a,b in zip(la,la_num):  

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=10)  

plt.show() 

 
sets = datas.Usage.value_counts()

da = [sets[i] for i in range(len(sets))]

set_la = ['Training','PublicTest','PrivateTest']

print(sets)

plt.axes(aspect=1)

plt.title('Size of Training,PublicTest,PrivateTest sets in the image dataset')

plt.pie(x = da,labels = set_la,autopct='%3.1f %%', shadow=True)

plt.show()
print('图片长度:',len(datas.pixels[1].split()))
time_1 = time.time()

print('数据读取耗时：',round((time_1 - time_start),2),'s')
train_set = datas[(datas.Usage == 'Training')] 

val_set = datas[(datas.Usage == 'PublicTest')]

test_set = datas[(datas.Usage == 'PrivateTest')] 

X_train = np.array(list(map(str.split, train_set.pixels)), np.float32) #, np.float32

X_val = np.array(list(map(str.split, val_set.pixels)), np.float32) 

X_test = np.array(list(map(str.split, test_set.pixels)), np.float32) 

X_train = X_train.reshape(X_train.shape[0], 48, 48) 

X_val = X_val.reshape(X_val.shape[0],48,48) 

X_test = X_test.reshape(X_test.shape[0],48, 48) 
y_train = list(train_set.emotion) 

y_val = list(val_set.emotion)

y_test = list(test_set.emotion )
fig = plt.figure(figsize = (10,8))

for i in range(len(X_train[:35])):

    if(y_train[i] == 0 ):

        str_la = 'Angry'

        img = Image.fromarray(np.uint8(X_train[i]))

    elif(y_train[i] == 1):

        str_la = 'Disgust'

        img = Image.fromarray(np.uint8(X_train[i]))

    elif(y_train[i] == 2):

        str_la = 'Fear'

        img = Image.fromarray(np.uint8(X_train[i]))

    elif(y_train[i] == 3):

        str_la = 'Happy'

        img = Image.fromarray(np.uint8(X_train[i]))

    elif(y_train[i] == 4):

        str_la = 'Sad'

        img = Image.fromarray(np.uint8(X_train[i]))

    elif(y_train[i] == 5):

        str_la = 'Surprise'

        img = Image.fromarray(np.uint8(X_train[i]))

    elif(y_train[i] == 6):

        str_la = 'Neural'

        img = Image.fromarray(np.uint8(X_train[i]))

    y = fig.add_subplot(5,7,i+1)

    y.imshow(img,cmap='gray')

    plt.title(str_la)

    y.axes.get_xaxis().set_visible(False)

    y.axes.get_yaxis().set_visible(False)

plt.show()
train_preprocess = tfs.Compose([

    tfs.ToPILImage(),#少了这一行就会报'int' object is not iterable

    tfs.RandomCrop(44),

    tfs.RandomHorizontalFlip(),

    tfs.ToTensor(),

])







val_preprocess = tfs.Compose([

    tfs.ToPILImage(),

    tfs.TenCrop(44),

    tfs.Lambda(lambda crops: torch.stack([tfs.ToTensor()(crop) for crop in crops])),

])



time_2 = time.time()

print('数据处理耗时：',round((time_2 - time_1),2),'s')
import torch

from torch.utils import data

import numpy as np

import torch.optim as optim

import torch.nn as nn

import pandas as pd

import torch.nn.functional as F

import torchvision.models as models
class Train_Dataset(data.Dataset):#括号里一定要写成data.Dataset,否则会报错

#       初始化 

    def __init__(self,X_train,labels):

        super(Train_Dataset,self).__init__()

        img = []

        label = []

        label = labels

        a = [train_preprocess(X_train[i])  for i in range(X_train.shape[0])]

        img = a

        self.img = img

        self.label=labels

      

            

    def __getitem__(self, index):

        

        imgs = self.img[index]

        labels = self.label[index]

        imgs_tensors =  imgs.type('torch.cuda.FloatTensor')

        return imgs_tensors, labels

        

    

    def __len__(self):

        return len(self.img)
class Val_Dataset(data.Dataset):#括号里一定要写成data.Dataset,否则会报错

#       初始化 

    def __init__(self,X_val,labels):

        super(Val_Dataset,self).__init__()

        img = []

        label = []

        label = labels

        b = [val_preprocess(X_val[i])  for i in range(X_val.shape[0])]

        img = b

        self.img = img

        self.label=labels

      

             

    def __getitem__(self, index):

        

        imgs = self.img[index]

        labels = self.label[index]

        imgs_tensors =  imgs.type('torch.cuda.FloatTensor')

        return imgs_tensors, labels

        

    

    def __len__(self):

        return len(self.img)
def validate_train(model,dataset,batch_size):

    val_loader = data.DataLoader(dataset,batch_size,shuffle=True)

    result,num = 0.0, 0

    for images,labels in val_loader:#ctrl + /多行注释

        images = images.cuda()

        pre = model.forward(images)

        pre = pre.cpu()#要从gpu中换回cpu上

        pre = np.argmax(pre.data.numpy(),axis = 1)

        labels = labels.data.numpy()

        result += np.sum((pre == labels))

        num += len(images)

    acc = result / num

    return acc
def validate_val(model,dataset,batch_size):

    val_loader = data.DataLoader(dataset,batch_size,shuffle=True)

    result,num = 0.0, 0

    for images,labels in val_loader:#ctrl + /多行注释

        for i in range(len(images)):

            images[i] = images[i].cuda()

            pre = model.forward(images[i])

            pre =pre.cpu()#要从gpu中换回cpu上

            pre = np.argmax(pre.data.numpy().mean(0))

            if pre == labels[i] :

                result = result + 1

        num += len(images)

    acc = result / num

    return acc
train_dataset = X_train

train_labels = y_train

Val_dataset = Val_Dataset( X_val,y_val)

Test_dataset = Val_Dataset(X_test,y_test)



batch_size= 128

learning_rate = 0.001

epochs= 20



#resnet18

resnet_historyloss = []

resnet_historyacc = []

resnet_historytrac = []

resnet_historytestac = []



#vgg19

vgg19_historyloss = []

vgg19_historyacc = []

vgg19_historytrac = []

vgg19_historytestac = []



#multiple

multiple_historyloss = []

multiple_historyacc = []

multiple_historytrac = []

multiple_historytestac = []
# def train_resnet18(train_dataset,train_labels,Val_dataset,Test_dataset,batch_size,epochs,learning_rate,momen_tum,wt_decay):



    

# #     构建模型

#     resnet18 = models.resnet18()

#     resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#     resnet18.fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)

    

#     model = resnet18.cuda()#放入gpu

# #     损失函数

#     loss_function = nn.CrossEntropyLoss()

#     loss_function =  loss_function.cuda()#放入gpu

    

# #     优化器

#     optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momen_tum,weight_decay=wt_decay)



# #     逐轮训练

#     print("Resnet18模型开始训练！")

#     for epoch in range(epochs):

# #         每一轮输入的图片都做一些变化，减缓过拟合的速度

#         Train_dataset = Train_Dataset(train_dataset,train_labels)

#         train_loader = data.DataLoader(Train_dataset,batch_size,shuffle=True)

#         loss_rate = 0

#         model.train()#模型训练

#         for images,labels in train_loader:#DataLoader的工作机制？？？？？

# #             梯度清零

            

#             images = images.cuda()#放入gpu

#             labels = labels.cuda()#放入gpu



#             optimizer.zero_grad()

# #             向前传播

#             output = model.forward(images)

# #             计算误差

#             loss_rate = loss_function(output,labels)

# #             误差的反向传播

#             loss_rate.backward()

# #             更新参数

#             optimizer.step()

#         resnet_historyloss.append(loss_rate.item())



# #         打印每轮的损失

        

       

            

#         model.eval()

#         #评估

#         acc_train = validate_train(model, Train_dataset, batch_size)

#         resnet_historytrac.append(acc_train)



#         acc_val = validate_val(model,Val_dataset,batch_size)

#         resnet_historyacc.append(acc_val)



#         acc_test = validate_val(model,Test_dataset,batch_size)

#         resnet_historytestac.append(acc_test)

        

#         if( (epoch+1) == epochs):

#             print("Resnet18模型最终测试结果：")

#             print('The acc_train is :',acc_train)

#             print('The acc_val is :',acc_val)

#             print('The acc_test is :',acc_test)

#             print('\n')



     

#     print("Resnet18模型训练完成！")        

#     return model
# def train_vgg19(train_dataset,train_labels,Val_dataset,Test_dataset,batch_size,epochs,learning_rate,momen_tum,wt_decay):



    

# #     构建模型

#     vgg19 = models.vgg19()

#     vgg19.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

#     vgg19.classifier[6] = torch.nn.Linear(in_features=4096, out_features=7, bias=True)

   

#     model = vgg19.cuda()#放入gpu

# #     损失函数

#     loss_function = nn.CrossEntropyLoss()

#     loss_function =  loss_function.cuda()#放入gpu

    

# #     优化器

#     optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momen_tum,weight_decay=wt_decay)



# #     逐轮训练

#     print("VGG19模型开始训练！")

#     for epoch in range(epochs):

# #         每一轮输入的图片都做一些变化，减缓过拟合的速度

#         Train_dataset = Train_Dataset(train_dataset,train_labels)

#         train_loader = data.DataLoader(Train_dataset,batch_size,shuffle=True)

#         loss_rate = 0

#         model.train()#模型训练

#         for images,labels in train_loader:

            

#             images = images.cuda()#放入gpu

#             labels = labels.cuda()#放入gpu

            

# #             梯度清零

#             optimizer.zero_grad()

# #             向前传播

#             output = model.forward(images)

# #             计算误差

#             loss_rate = loss_function(output,labels)

# #             误差的反向传播

#             loss_rate.backward()

# #             更新参数

#             optimizer.step()

# #         打印每轮的损失    

#         vgg19_historyloss.append(loss_rate.item())





#         model.eval()

#         #评估

#         acc_train = validate_train(model, Train_dataset, batch_size)

#         vgg19_historytrac.append(acc_train)



#         acc_val = validate_val(model,Val_dataset,batch_size)

#         vgg19_historyacc.append(acc_val)



#         acc_test = validate_val(model,Test_dataset,batch_size)

#         vgg19_historytestac.append(acc_test)





#         if((epoch+1) == epochs):

            

#             print("VGG19模型最终测试结果：")

#             print('The acc_train is :',acc_train)

#             print('The acc_val is :',acc_val)

#             print('The acc_test is :',acc_test)

#             print('\n')



#     print("VGG19模型训练完成！")       

#     return model
# resnet18 = train_resnet18(train_dataset,train_labels,Val_dataset,Test_dataset,batch_size,epochs,learning_rate ,momen_tum=0.9,wt_decay = 5e-4)

# torch.save(resnet18,'fer2013_resnet18_model.pkl')



# vgg19 = train_vgg19(train_dataset,train_labels,Val_dataset,Test_dataset,batch_size,epochs,learning_rate ,momen_tum=0.9,wt_decay = 5e-4)

# torch.save(vgg19,'fer2013_vgg19_model.pkl')
resnet = torch.load("../input/fer2013-resnet18-model/fer2013_resnet18_model.pkl")


vgg = torch.load("../input/fer2013-vgg19-modelpkl/fer2013_vgg19_model.pkl")
class Multiple(nn.Module):

    def __init__(self):

        super(Multiple,self).__init__()        

        

        self.fc = nn.Sequential(

             nn.Linear(in_features = 14,out_features = 7),

        )

        

    def forward(self,x):

        

        #经过基模型预处理

        result_1 = vgg(x)

        result_2 = resnet(x)

        

        #拼接基模型处理后的特征

        result_1 = result_1.view(result_1.shape[0],-1)

        result_2 = result_2.view(result_2.shape[0],-1)

        result = torch.cat((result_1,result_2),1)

        

        #将基模型处理后的特征输入融合模型中

        y = self.fc(result)

        

        return y
def multiple_train(train_dataset,train_labels,Val_dataset,Test_dataset,batch_size,epochs,learning_rate,momen_tum,wt_decay):



    

#     构建模型

    model = Multiple()

    model = model.cuda()#放入gpu

#     损失函数

    loss_function = nn.CrossEntropyLoss()

    loss_function =  loss_function.cuda()#放入gpu

    

#     优化器

    optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momen_tum,weight_decay=wt_decay)



#     逐轮训练

    print("融合模型开始训练！")

    for epoch in range(epochs):

#         每一轮输入的图片都做一些变化，减缓过拟合的速度

        Train_dataset = Train_Dataset(train_dataset,train_labels)

        train_loader = data.DataLoader(Train_dataset,batch_size,shuffle=True)

        loss_rate = 0

        model.train()#模型训练

        for images,labels in train_loader:#DataLoader的工作机制？？？？？

#             梯度清零

            

            images = images.cuda()#放入gpu

            labels = labels.cuda()#放入gpu



            optimizer.zero_grad()

#             向前传播

            output = model(images)

#             计算误差

            loss_rate = loss_function(output,labels)

#             误差的反向传播

            loss_rate.backward()

#             更新参数

            optimizer.step()

        multiple_historyloss.append(loss_rate.item())

#         打印每轮的损失

        

        

        model.eval()

        #评估

        acc_train = validate_train(model, Train_dataset, batch_size)

        multiple_historytrac.append(acc_train)

        

        acc_val = validate_val(model,Val_dataset,batch_size)

        multiple_historyacc.append(acc_val)

        

        acc_test = validate_val(model,Test_dataset,batch_size)

        multiple_historytestac.append(acc_test)



        

        print('After {} epochs : '.format(epoch+1))

        print('The loss_rate is :',loss_rate.item())

        print('The acc_train is :',acc_train)

        print('The acc_val is :',acc_val)

        print('The acc_test is :',acc_test)

        print('\n')

    

    print("融合模型训练结束！")   

    

    return model
model = multiple_train(train_dataset,train_labels,Val_dataset,Test_dataset,batch_size,epochs,learning_rate ,momen_tum=0.9,wt_decay = 5e-4)

torch.save(model,'fer2013_multiple_model.pkl')
# print("所有模型训练结束！")
time_3 = time.time()

print('模型训练耗时：',round((time_3 - time_2) / 3600,2),'h')
mul = torch.load('fer2013_multiple_model.pkl')

# vgg = torch.load('fer2013_vgg19_model.pkl')

# resnet = torch.load('fer2013_resnet18_model.pkl')
def plots(historyloss,historyacc,historytrac,historytestac):

    

    epochs = range(len(historyacc))



    plt.plot(epochs,historyloss,'r', label='train_loss')

    plt.plot(epochs,historyacc,'b', label='acc_val')

    plt.plot(epochs,historytrac,'g', label='acc_train')

    plt.plot(epochs,historytestac,'y', label='acc_test')



    plt.title('epoch and acc and loss_rate')

    plt.xlabel('epoch')

    plt.ylabel('acc and loss')

    plt.legend()

    plt.figure()

    
# print("resnet18模型")

# plots(resnet_historyloss,resnet_historyacc,resnet_historytrac,resnet_historytestac)
# print("vgg19模型")

# plots(vgg19_historyloss,vgg19_historyacc,vgg19_historytrac,vgg19_historytestac)
print("融合模型")

plots(multiple_historyloss,multiple_historyacc,multiple_historytrac,multiple_historytestac)
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import numpy as np
def plot_confusion_matrix(model,dataset,batch_size):

    y_true = []

    y_pred = []

    label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neural']#里面是中文会显示不出来，不知道为什么？

    tick_marks = np.array(range(len(label))) + 0.5

    

    val_loader = data.DataLoader(dataset,batch_size,shuffle=True)

    for images,labels in val_loader:#ctrl + /多行注释

        for i in range(len(images)):

            images[i] = images[i].cuda()

            pre = model.forward(images[i])

            pre =pre.cpu()#要从gpu中换回cpu上

            pre = np.argmax(pre.data.numpy().mean(0))

            y_true.append(labels[i])

            y_pred.append(pre)

    

    cm = confusion_matrix(y_true, y_pred)

    np.set_printoptions(precision=2)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6), dpi=80)



    ind_array = np.arange(len(label))

    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):

        c = cm_normalized[y_val][x_val]

        if c > 0.01:

            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')

    plt.gca().set_xticks(tick_marks, minor=True)

    plt.gca().set_yticks(tick_marks, minor=True)

    plt.gca().xaxis.set_ticks_position('none')

    plt.gca().yaxis.set_ticks_position('none')

    plt.grid(True, which='minor', linestyle='-')

    plt.gcf().subplots_adjust(bottom=0.15)



   

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion Matrix')

    plt.colorbar()

    xlocations = np.array(range(len(label)))

    plt.xticks(xlocations, label, rotation=70)#调整底下标签的旋转角度

    plt.yticks(xlocations, label)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    plt.show()
y_val = list(val_set.emotion)

dataset = Val_Dataset( X_val,y_val)

batch_size = 128
# print("vgg19模型  Confusion Matrix")

# plot_confusion_matrix(vgg,dataset,batch_size)



# print("resnet18模型 Confusion Matrix")

# plot_confusion_matrix(resnet,dataset,batch_size)



print("融合模型 Confusion Matrix")

plot_confusion_matrix(mul,dataset,batch_size)
model = torch.load('fer2013_multiple_model.pkl')
y_val = list(val_set.emotion)

y_test = list(test_set.emotion )

#y_val会变空，不知道为什么，所以这里重新生成标签信息
Va_dataset = Val_Dataset( X_val,y_val)

acc_val = validate_val(model,Va_dataset,128)

print('准确率：',acc_val)
def validate(model,dataset,batch_size):

    val_loader = data.DataLoader(dataset,batch_size,shuffle=True)

    result,num = 0.0, 0

    y_pred = []

    

    for images,labels in val_loader:#ctrl + /多行注释

        for i in range(len(images)):

            images[i] = images[i].cuda()

            pre = model.forward(images[i])

            pre =pre.cpu()#要从gpu中换回cpu上

            pre = np.argmax(pre.data.numpy().mean(0))

            y_pred.append(pre)

    return y_pred
i = 1000

j = 1100

img_val = X_val[i:j]

label_val = y_val[i:j]

im_0= []

im_1= []

im_2= []

im_3= []

im_4= []

im_5= []

im_6= []

la_0 = []

la_1 = []

la_2 = []

la_3 = []

la_4 = []

la_5 = []

la_6 = []

for k in range(len(img_val)):

    

    if(label_val[k] == 0):

        

        im_0.append(img_val[k])

        la_0.append(0)

                

    elif(label_val[k] == 1):

        

        im_1.append(img_val[k])

        la_1.append(1)

        

    elif(label_val[k] ==2):

        

        im_2.append(img_val[k])

        la_2.append(2)

        

    elif(label_val[k] ==3):

        

        im_3.append(img_val[k])

        la_3.append(3)

        

    elif(label_val[k] ==4):

        

        im_4.append(img_val[k])

        la_4.append(4)

        

    elif(label_val[k] ==5):

        

        im_5.append(img_val[k])

        la_5.append(5)

        

    elif(label_val[k] ==6):

        

        im_6.append(img_val[k])

        la_6.append(6)

         
x_lis = []

y_lis = []

x_lis.append(np.array(im_0))

x_lis.append(np.array(im_1))

x_lis.append(np.array(im_2))

x_lis.append(np.array(im_3))

x_lis.append(np.array(im_4))

x_lis.append(np.array(im_5))

x_lis.append(np.array(im_6))

y_lis.append(la_0)

y_lis.append(la_1)

y_lis.append(la_2)

y_lis.append(la_3)

y_lis.append(la_4)

y_lis.append(la_5)

y_lis.append(la_6)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neural']
for i in range(len(x_lis)):

    Va_dataset = Val_Dataset( x_lis[i],y_lis[i])

    pre = validate(model,Va_dataset,1)

    print(labels[y_lis[i][0]])

    print('实际结果：\t',y_lis[i])

    print('预测结果为：\t',pre)
time_end = time.time()

time_c= time_end - time_start 

print('总耗时:',round(time_c / 3600,2) , 'h')
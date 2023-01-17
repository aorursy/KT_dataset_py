import os 
import numpy as np
import cv2
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from focalloss import *
import torchvision.models as models
train_label_path='/media/baek/새 볼륨/train.csv'
sub_label_path='/media/baek/새 볼륨/sample_submission.csv'
train_image_path='/media/baek/새 볼륨/train/'
test_image_path='/media/baek/새 볼륨/test/'
retrain_image_path='/media/baek/새 볼륨/retrain/'
retest_image_path='/media/baek/새 볼륨/retest/'
def get_data_label(path):
    data=pd.read_csv(path)
    name=list(data['Id'])
#    label=[[int(i) for i in s.split()]for s in data['Target'][18000:]]
#    label = np.eye(28,dtype=np.float)[label].sum(axis=0)
    return name

def get_image_data(path,name):
    
    img=cv2.imread(path+name+'.png',1)
    
    img=np.array(img,dtype='f')
    img=np.divide(img,255)
    img=transforms.ToTensor()(img)
    return img
#train_name,train_label=get_data_label(train_label_path)
test_name=get_data_label(sub_label_path)

#train_labels=[]
#for i in train_label:
#    train_labels.append(np.eye(28,dtype=np.float32)[i].sum(axis=0))  

img=[]

for name in test_name:
    
#    img.append(get_image_data(retrain_image_path,name))
    img.append(get_image_data(retest_image_path,name))
#test_labels=[(test_labels[i]/test_labels[i].sum())for i in range(len(train_labels))]

print(len(img))

#train_data=[(i,j) for i,j in zip(img,train_labels)]

from torch.utils.data import DataLoader
#trainloader=DataLoader(train_data,batch_size=32,
#                        shuffle=True,num_workers=4,drop_last=True)
testloader=DataLoader(img,batch_size=32,
                        shuffle=False,num_workers=4,drop_last=False)



#print(train_labels[0])
#img.clear()
#test_labels.clear()
#train_data.clear()

from MobileNetV2 import MobileNetV2
net = MobileNetV2(n_class=28)

#checkpoint = torch.load('./train_.pth', map_location="cuda:0")
device = torch.device("cuda")
#net.load_state_dict(checkpoint['model_state_dict'])
#net.to(device)
net.load_state_dict(torch.load('./train_final.pth', map_location="cuda:0")) 


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.eval()
#criterion = FocalLoss(gamma=0.3).to(device)
#criterion = nn.BCELoss(size_average=True)
#criterion = nn.BCEWithLogitsLoss(size_average=True)
#optimizer = optim.Adam(net.parameters(), lr=0.0001)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


epochs=8


for epoch in range(epochs):
#    print('\n===> epoch %d' % epoch)
    
    
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        # get the inputs
        inputs, labels = data
        inputs =inputs.to(device=device)
        labels = labels.to(device=device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        
#        outputs=F.sigmoid(outputs)
        
#        outputs=torch.round(outputs)
        
#        print(outputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
           
            
            
          
            running_loss = 0.0
#torch.save({
 #           'epoch': epoch,
#            'model_state_dict': net.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            
#            }, './train_mo1.pth')


torch.save(net.state_dict(), './train_final.pth')
c=[]
with torch.no_grad():
    for data in testloader:
        images = data
        
        images=images.to(device)
        outputs = net(images)
#        print(outputs)
#        a=torch.round(outputs)
        a=outputs.cpu()
        b=Variable(a).numpy()
        
        for i in b:
            
            b=np.where(i>0.2)
            label=b[0].tolist()
            str_predict_label = ' '.join(str(l) for l in label)
            c.append(str_predict_label)
data=pd.read_csv(sub_label_path)
data.head()

data['Predicted']=c
data.head()

data.to_csv('./submission.csv',index=False)

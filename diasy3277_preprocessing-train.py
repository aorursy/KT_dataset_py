import librosa
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd
import scipy as sklearn

import os
from google.colab import drive
drive.mount('/content/gdrive/')
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader
from matplotlib.pyplot import imshow
%matplotlib inline
import os
#train_data = torchvision.datasets.ImageFolder(root='/content/gdrive/My Drive/music_genres/genres_original', transform=trans)
filename=np.array(pd.read_csv('/content/filename.csv')) #전체 데이터 이름만을 담은 csv
answer = np.array(pd.read_csv('/content/label.csv')) #전체 데이터 라벨링한것
num=0
size_ = 2.0

# 오디오 데이터를  MFCC로 전처리 하여 각 파일에 저장
# blues:0
#classical:1
#country:2
#disco:3
#hiphop:4
#jazz:5
#metal:6
#pop:7
#reggae:8
#rock:9
for step in range(10001):
            audio_path='/content/gdrive/My Drive/music_genres/genres_original/all/'+filename[step] 
            #파일이름에 해당하는 데이터를 MFCC로 시각화
            #print(audio_path)
            #print(answer[num])
                
            
            y, sr = librosa.load(audio_path[0])
            mfccs=librosa.feature.mfcc(y=y,sr=sr)
            plt.figure(figsize=(0.78*size_,0.78*size_)) #size 56x56
            librosa.display.specshow(mfccs)
            #plt.colorbar()
            #plt.title(audio_path[0])
            print(audio_path[0])
            plt.tight_layout()
            temp = ''

              # 라벨링한 csv를 통해 내가원하는 폴더에 이미지들을 저장.
            if answer[num] == 0:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/blues/' + str(step)+'.png'   
            
            elif  answer[num] == 1:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/classical/' + str(step)+'.png'   
            elif  answer[num] == 2:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/country/' + str(step)+'.png'
            elif  answer[num] == 3:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/disco/' + str(step)+'.png'
            elif  answer[num] == 4:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/hiphop/' + str(step)+'.png'
            elif  answer[num] == 5:
            
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/jazz/' + str(step)+'.png'   
            elif  answer[num] == 6:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/metal/' + str(step)+'.png'
            elif  answer[num] == 7:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/pop/' + str(step)+'.png'
            elif  answer[num] == 8:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/reggae/' + str(step)+'.png'
            elif  answer[num] == 9:
                temp = '/content/gdrive/My Drive/music_genres/images_original/MFCC/rock/' + str(step)+'.png'
              
            plt.savefig(temp)
               # f_path = os.path.join(save_path,file)
            
            num=num-1

print('Finished!')
          
trans = transforms.Compose([
    transforms.Resize((28,28))
])
#이미지 폴더를 사용하여 train_data만들기
train_data = torchvision.datasets.ImageFolder(root='/content/gdrive/My Drive/music_genres/images_original/MFCC/train_data', transform=trans)

for num, value in enumerate(train_data):#train_data를 내가 원하는 폴더에 저장
    data, label = value
    print(num, data, label)
    
    if(label == 0):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/blues/%d_%d.jpeg'%(num, label))
    elif(label ==1):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/classical/%d_%d.jpeg'%(num, label))
    elif(label ==2):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/country/%d_%d.jpeg'%(num, label))
    elif(label ==3):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/disco/%d_%d.jpeg'%(num, label))
    elif(label ==4):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/hiphop/%d_%d.jpeg'%(num, label))
    elif(label ==5):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/jazz/%d_%d.jpeg'%(num, label))
    elif(label ==6):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/metal/%d_%d.jpeg'%(num, label))
    elif(label ==7):
        data.save('/content/gdrive/My Drive/music_genres/images_original//MFCC/train/pop/%d_%d.jpeg'%(num, label))
    elif(label ==8):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/reggae/%d_%d.jpeg'%(num, label))
    elif(label ==9):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/train/rock/%d_%d.jpeg'%(num, label))
        
trans = transforms.Compose([
    transforms.Resize((28,28)) #28*28 사이즈로 조절
])
#이미지 폴더를 이용해서 train set 만들기
test_data = torchvision.datasets.ImageFolder(root='/content/gdrive/My Drive/music_genres/images_original/MFCC/test_data', transform=trans)

for num, value in enumerate(test_data): #test_data를 내가 원하는 폴더에 저장
    data, label = value
    print(num, data, label)
    
    if(label == 0):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/blues/%d_%d.jpeg'%(num, label))
    elif(label ==1):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/classical/%d_%d.jpeg'%(num, label))
    elif(label ==2):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/country/%d_%d.jpeg'%(num, label))
    elif(label ==3):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/disco/%d_%d.jpeg'%(num, label))
    elif(label ==4):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/hiphop/%d_%d.jpeg'%(num, label))
    elif(label ==5):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/jazz/%d_%d.jpeg'%(num, label))
    elif(label ==6):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/metal/%d_%d.jpeg'%(num, label))
    elif(label ==7):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/pop/%d_%d.jpeg'%(num, label))
    elif(label ==8):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/reggae/%d_%d.jpeg'%(num, label))
    elif(label ==9):
        data.save('/content/gdrive/My Drive/music_genres/images_original/MFCC/test/rock/%d_%d.jpeg'%(num, label))
    else:
       break;

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)
trans = transforms.Compose([
    transforms.ToTensor() #학습을 위해 dataset들을 텐서로 전환
])

train_data = torchvision.datasets.ImageFolder(root='/content/gdrive/My Drive/music_genres/images_original/MFCC/train',transform=trans)
test_data = torchvision.datasets.ImageFolder(root='/content/gdrive/My Drive/music_genres/images_original/MFCC/test', transform=trans)

batch_size = 7
data_loader = DataLoader(dataset = train_data, 
                         batch_size = batch_size, 
                         shuffle = True, 
                         num_workers=2, # 데이터를 빨리 읽어오기 위한것
                         drop_last=True)
test_set = DataLoader(dataset = test_data, 
                      batch_size = len(test_data),
                      shuffle=False)
linear1 = torch.nn.Linear(3*784,784,bias=True) #input 3*28*28
linear2 = torch.nn.Linear(784,512,bias=True)
linear3 = torch.nn.Linear(512,512,bias=True)
linear4 = torch.nn.Linear(512,512,bias=True) #output 10 class
linear5 = torch.nn.Linear(512,10,bias=True)
#linear6 = torch.nn.Linear(512,10,bias=True)
relu = torch.nn.ReLU() 
dropout = torch.nn.Dropout(p=0.3)
torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
#torch.nn.init.xavier_uniform_(linear6.weight)
model = torch.nn.Sequential(linear1,relu,dropout,
                            linear2,relu,dropout,
                            linear3, relu,dropout,
                            linear4,relu,dropout,
                            linear5
                            #linear6
                            ).to(device)
loss = torch.nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.Adamax(model.parameters(), lr=0.1) 
total_batch = len(data_loader)
#model.train()
for epoch in range(350):
    avg_cost = 0

    for X, Y in data_loader:

       
        #print(X.shape) #텐서의 크기를 줄여줌
        X = X.view(-1, 3*28 * 28).to(device)
        # one-hot encoding되어 있지 않음
        Y = Y.to(device)
      
        optimizer.zero_grad()
        hypothesis = model(X) #forward 계산
        cost = loss(hypothesis, Y) #cost
        cost.backward() #backpropagation
        optimizer.step() #갱신

        avg_cost += cost / total_batch #평균 error
    if epoch % 50==0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
with torch.no_grad():
  
    for num, data in enumerate(test_set):
        imgs, label = data
        imgs = imgs.view(-1,3* 28 * 28).to(device)
        label = label.to(device)
        
        prediction = model(imgs)
        
        correct_prediction = torch.argmax(prediction, 1) == label
        
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
prediction= prediction.cpu().data.numpy().reshape(-1,1) #텐서를 cpu로 바꾼다음 array로 저장
submit=pd.read_csv('submition_form.csv')
submit
for i in range(len(prediction)):
  submit['label'][i]=int(prediction[i])
submit['label']=submit['label'].astype(int)  

submit.to_csv('baseline.csv',index=False,header=True)

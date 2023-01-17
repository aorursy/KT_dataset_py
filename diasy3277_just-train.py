!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
!kaggle competitions download -c music-genres-classification
!unzip music-genres-classification.zip
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

train_data = torchvision.datasets.ImageFolder(root='/content/train',transform=trans)
test_data = torchvision.datasets.ImageFolder(root='/content/test', transform=trans)
batch_size = 7
data_loader = DataLoader(dataset = train_data, 
                         batch_size = batch_size, 
                         shuffle = True, 
                         num_workers=2, # 데이터를 빨리 읽어오기 위한것
                         drop_last=True)
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
for epoch in range(100):
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
test_set = DataLoader(dataset = test_data, 
                      batch_size = len(test_data),
                      shuffle=False)
with torch.no_grad():
  
    for num, data in enumerate(test_set):
        imgs, label = data
        imgs = imgs.view(-1,3* 28 * 28).to(device)
        label = label.to(device)
        
        prediction = model(imgs)
        
        correct_prediction = torch.argmax(prediction, 1) == label
        
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
prediction= prediction.cpu().data.numpy().reshape(-1,1)
submit=pd.read_csv('submition_form.csv')

for i in range(len(prediction)):
  submit['label'][i]=int(prediction[i])
submit['label']=submit['label'].astype(int)  

submit.to_csv('baseline.csv',index=False,header=True)

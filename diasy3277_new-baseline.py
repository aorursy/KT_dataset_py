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
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)
trans = transforms.Compose([
    transforms.ToTensor(), #학습을 위해 dataset들을 텐서로 전환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.ImageFolder(root='/content/train', transform=trans)
test_data = torchvision.datasets.ImageFolder(root='/content/test', transform=trans)
batch_size = 200
data_loader = DataLoader(dataset = train_data, 
                         batch_size = batch_size, 
                         shuffle = True, 
                         num_workers=2, # 데이터를 빨리 읽어오기 위한것
                         drop_last=True)
test_set = DataLoader(dataset = test_data, 
                      batch_size = len(test_data),
                      shuffle=False)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU()
        
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        

        self.fc = nn.Sequential(
            nn.Linear(128*4*4,256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True)
        )    
        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        torch.nn.init.xavier_uniform_(self.fc[2].weight)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc(out)
        return out
model = MyModel().to(device)
loss = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
total_batch = len(data_loader)
model.train()
for epoch in range(150):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        # one-hot encoding되어 있지 않음
        Y = Y.to(device)
      
        optimizer.zero_grad()
        hypothesis = model(X) #forward 계산
        cost = loss(hypothesis, Y) #cost
        cost.backward() #backpropagation
        optimizer.step() #갱신

        avg_cost += cost / total_batch #평균 error
    if epoch % 10==0:
        print('Epoch:', '%04d' % (epoch), 'cost =', '{:.9f}'.format(avg_cost,))
            

print('Learning finished')
with torch.no_grad():
    model.eval()
    for _, data in enumerate(test_set):
        imgs, label = data
#                 print(len(imgs), len(label))
        imgs = imgs.to(device)
        label = label.to(device)

        prediction = model(imgs)

        correct_prediction = torch.argmax(prediction, 1) == label

        accuracy = correct_prediction.float().mean()
        print(accuracy.item())
import pandas as pd

submit=pd.read_csv('submition_form.csv')
submit
submit['label'] = torch.argmax(prediction, 1).cpu().numpy()
submit.to_csv('new_baseline.csv',index=False, header=True)
!kaggle competitions submit -c music-genres-classification -f new_baseline.csv -m "submit"

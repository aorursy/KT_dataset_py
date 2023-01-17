import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL.ImageOps
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
import torchvision.utils
from itertools import combinations
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text,
            style='italic',
            fontweight='bold',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10}
        )
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
class ContrastiveLoss(th.nn.Module):
    "Contrastive loss function"
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = th.mean(
            (1 - label) * th.pow(euclidean_distance, 2) +
            label * th.pow(th.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
training_dir_list = os.listdir('../input/dac-yapikredi-project/train (1)/train')
training_data = pd.DataFrame(
    list(combinations(training_dir_list, 2)),
    columns=['img1no', 'img2no']
)
training_data = training_data[
    training_data.img1no.str[4:7] == training_data.img2no.str[4:7]
]
training_data = training_data.sort_values(["img1no", "img2no"])
training_data = training_data.reset_index(drop=True)
training_data['label'] = 0
training_data.to_csv('./train_data.csv', index=False)
training_data
testing_dir_list = os.listdir('../input/dac-yapikredi-project/test/test')
test_data = pd.DataFrame(
    list(combinations(testing_dir_list, 2)), columns=['img1no', 'img2no']
)
test_data = test_data[test_data.img1no.str[4:7] == test_data.img2no.str[4:7]]
test_data = test_data.sort_values(["img1no", "img2no"])
test_data = test_data.reset_index(drop=True)
test_data['label'] = 0
test_data.to_csv('./test_data.csv', index=False)
test_data
idinfo = pd.read_csv("../input/dac-yapikredi-project/idinfo.csv")
idinfo
submission = pd.read_csv("../input/dac-yapikredi-project/samplesub2c.csv")
idinfo1 = pd.DataFrame(
    'NFI-' +
    idinfo['person1'].astype(str).str.zfill(3) +
    idinfo['imgno1'].astype(str).str.zfill(2) +
    idinfo['person1'].astype(str).str.zfill(3) +
    '.png',
    columns=['img1no']    
)
idinfo1['img2no'] = (
        'NFI-' +
        idinfo['person2'].astype(str).str.zfill(3) +
        idinfo['imgno2'].astype(str).str.zfill(2) +
        idinfo['person2'].astype(str).str.zfill(3) +
        '.png'   
)
idinfo1['label'] = np.where(idinfo1.img1no.str[4:7] == idinfo1.img2no.str[4:7], 0, 1)
idinfo1.to_csv('./testing.csv', index=False)
idinfo1
# dac-yapikredi-project veri setini yükleme 
training_dir = "../input/dac-yapikredi-project/train (1)/train"
training_csv = "../input/training-data-label/training_data_label.csv"

testing_dir = "../input/dac-yapikredi-project/test/test"
testing_csv = "../input/test-data-label/test_data_label.csv"

batch_size = 64  # 32
epochs = 40  # 256
# veri setini ön işleme ve resimlerin yeniden boyutlandırılması
class SiameseDataset():

    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # resimlerin etiketleri ve dizinleri
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        # resimlerin dizinleri
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        # resimlerin yüklenmesi
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # resimlerin yeniden boyutlandırılması
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, th.from_numpy(
            np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
        )

    def __len__(self):
        return len(self.train_df)
# Eğitim veri setinin yüklenmesi
siamese_dataset = SiameseDataset(
    training_csv,
    training_dir,
    transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])    
)
# Örnek imza resimleri
vis_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=8)
dataiter = iter(vis_dataloader)

example_batch = next(dataiter)
concatenated = th.cat((example_batch[0], example_batch[1]), 0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
# Standart CNN
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Ardışık CNN katmanlarının ayarlanması
        self.cnn1 = nn.Sequential(
            
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )
        
        # Bağlantılı katmanların tanımlanması
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2)
        )        
  
  
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
# Dataloader'i kullanarak veri setini pytorch tensors'a yükleme
train_dataloader = DataLoader(
    siamese_dataset,
    shuffle=True,
    num_workers=8,
    batch_size=batch_size
)   
# Siamese Network
net = SiameseNetwork().cuda()
# Loss Function
criterion = ContrastiveLoss()
# Optimizer
optimizer = th.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
# Modelin eğitilmesi fonksiyonu
def train():
    loss=[] 
    counter=[]
    iteration_number = 0

    for epoch in range(1,epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1 ,output2, label)
            loss_contrastive.backward()
            optimizer.step()
            
        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    show_plot(counter, loss)   
    return net
# Ekran kartının CUDA mimarisini kullanması için ayarlanması
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Modelin eğitilmesi
model = train()
th.save(model.state_dict(), "model.pt")
print("<Model saved successfully>")
testing_dir = "../input/dac-yapikredi-project/test/test"
testing_csv = "./testing.csv"

batch_size = 64  # 32
epochs = 40  # 256
# Test veri setinin yüklenmesi
test_dataset = SiameseDataset(
    training_csv=testing_csv,
    training_dir=testing_dir,
    transform=transforms.Compose([transforms.Resize((105, 105)),transforms.ToTensor()])
)
test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)
# Ekran kartının CUDA mimarisini kullanması için ayarlanması
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Eğitilen modelin yüklenmesi
model = SiameseNetwork().to(device)
model.load_state_dict(th.load("./model.pt"))
counter = 0
for i, data in enumerate(test_dataloader, 0):
    x0, x1, label = data
    concatenated = th.cat((x0, x1), 0)
    output1, output2 = model(x0.to(device), x1.to(device))
    eucledian_distance = F.pairwise_distance(output1, output2)
    submission['score'][i] = eucledian_distance
    counter = counter + 1
    if counter == len(submission):
        break
eucledian_distance_min = submission['score'].min()
eucledian_distance_max = submission['score'].max()
# Öklid uzaklığının, benzerlik skoruna dönüştürülmesi
submission['score'] = (
        (submission['score'] - submission['score'].min()) /
        (submission['score'].max() - submission['score'].min())
)

# Aynı müşterinin aynı imza resmi için 1.00 benzerlik
indices = np.where(idinfo1.img1no == idinfo1.img2no)
for i in indices:
    submission['score'][i] = 0

submission['score'] = 1 - submission['score']
submission
submission.to_csv('./submission.csv', index=False)
counter = 0
list_0 = th.FloatTensor([[0]])
list_1 = th.FloatTensor([[1]])
for i, data in enumerate(test_dataloader, 0): 
  x0, x1, label = data
  concatenated = th.cat((x0, x1), 0)
  output1, output2 = model(x0.to(device), x1.to(device))
  eucledian_distance = F.pairwise_distance(output1, output2)
  
  if label == list_0:
    label = "Aynı Müşteri"
  else:
    label = "Farklı Müşteri"
      
  imshow(
      torchvision.utils.make_grid(concatenated),
      'Benzerlik Skoru: {:.2f} Label: {}'.format(
          (eucledian_distance.item() - eucledian_distance_min) /
          (eucledian_distance_max - eucledian_distance_min),
          label          
      )   
  )
  counter = counter + 1  
  if counter == 10:
     break
"""test_dataloader = DataLoader(test_dataset,num_workers=6,batch_size=1,shuffle=True)
accuracy=0
counter=0
correct=0
for i, data in enumerate(test_dataloader,0): 
  x0, x1 , label = data
  # onehsot applies in the output of 128 dense vectors which is then converted to 2 dense vectors
  output1,output2 = model(x0.to(device),x1.to(device))
  res=th.abs(output1.cuda() - output2.cuda())
  label=label[0].tolist()
  label=int(label[0])
  result=th.max(res,1)[1][0][0][0].data[0].tolist()
  if label == result:
        correct=correct+1
        counter=counter+1
#   if counter ==20:
#      break
    
accuracy=(correct/len(test_dataloader))*100
print("Accuracy:{}%".format(accuracy))"""
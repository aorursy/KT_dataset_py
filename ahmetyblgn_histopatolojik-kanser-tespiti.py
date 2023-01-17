import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

base_dir = '../input/'

print(os.listdir(base_dir))



# Matplotlib for visualization

import matplotlib.pyplot as plt

plt.style.use("ggplot")



# OpenCV Image Library

import cv2



# Import PyTorch / Derin öğrenme için  framework 

import torchvision.transforms as transforms # yaygın görüntü dönüşümleri

from torch.utils.data.sampler import SubsetRandomSampler  # veri yükleme yardımcı 

import torch

import torch.nn as nn  # sinir ağı modülleri için temel sınıf

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision

import torch.optim as optim # çeşitli optimizasyon algoritmaları uygulayan bir paket



# Import useful sklearn functions

import sklearn

from sklearn.metrics import roc_auc_score, accuracy_score

from PIL import Image
full_train_df = pd.read_csv("../input/train_labels.csv")

full_train_df.head()
print("Train Size: {}".format(len(os.listdir('../input/train/'))))

print("Test Size: {}".format(len(os.listdir('../input/test/'))))
labels_count = full_train_df.label.value_counts()



%matplotlib inline

plt.pie(labels_count, labels=['No Cancer', 'Cancer'], startangle=180, 

        autopct='%1.1f', colors=['#00ff99','#FF96A7'], shadow=True)

plt.figure(figsize=(16,16))

plt.show()
fig = plt.figure(figsize=(30, 6))

# display 20 images

train_imgs = os.listdir(base_dir+"train")

for idx, img in enumerate(np.random.choice(train_imgs, 20)):

    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])

    im = Image.open(base_dir+"train/" + img)

    plt.imshow(im)

    lab = full_train_df.loc[full_train_df['id'] == img.split('.')[0], 'label'].values[0]

    ax.set_title('Label: %s'%lab)
# Her sınıftaki örnek sayısı

SAMPLE_SIZE = 80000



# Data paths

train_path = '../input/train/'

test_path = '../input/test/'



# 80000 olumlu ve olumsuz örnek kullanın

df_negatives = full_train_df[full_train_df['label'] == 0].sample(SAMPLE_SIZE, random_state=42)

df_positives = full_train_df[full_train_df['label'] == 1].sample(SAMPLE_SIZE, random_state=42)



# İki dfs'yi birleştirme ve karıştırma işlemi

train_df = sklearn.utils.shuffle(pd.concat([df_positives, df_negatives], axis=0).reset_index(drop=True))

train_df.shape
# Veri kümeleri için kendi özel sınıfımız

class CreateDataset(Dataset):

    def __init__(self, df_data, data_dir = './', transform=None):

        super().__init__()

        self.df = df_data.values

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_dir, img_name+'.tif')

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label
transforms_train = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(p=0.4),

    transforms.RandomVerticalFlip(p=0.4),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    #  tüm görüntülerin kanalları için aşağıdaki ortalama ve std olsun (normalizasyon)

    #transforms.Normalize((0.70244707, 0.54624322, 0.69645334), (0.23889325, 0.28209431, 0.21625058))

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



train_data = CreateDataset(df_data=train_df, data_dir=train_path, transform=transforms_train)
# Parti Boyutunu Ayarla

batch_size = 128



# Doğrulama olarak kullanılacak eğitimin yüzdesi

valid_size = 0.1



# validasyon için kullanılacak eğitim endeksleri elde etmek

num_train = len(train_data)

indices = list(range(num_train))

# np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



# Örneklenmiş veri çek

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



# veri yükleyicileri hazırlar (veri seti ve örnekleyiciyi birleştirir)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
transforms_test = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    #transforms.Normalize((0.70244707, 0.54624322, 0.69645334), (0.23889325, 0.28209431, 0.21625058))

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



# test verileri oluşturma

sample_sub = pd.read_csv("../input/sample_submission.csv")

test_data = CreateDataset(df_data=sample_sub, data_dir=test_path, transform=transforms_test)



# test yükleyicisini hazırla

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        # Evrişim ve Havuzlama Katmanları

        self.conv1=nn.Sequential(

                nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0),

                nn.BatchNorm2d(32),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        self.conv2=nn.Sequential(

                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1,padding=1),

                nn.BatchNorm2d(64),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        self.conv3=nn.Sequential(

                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),

                nn.BatchNorm2d(128),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        self.conv4=nn.Sequential(

                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),

                nn.BatchNorm2d(256),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        self.conv5=nn.Sequential(

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),

                nn.BatchNorm2d(512),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(2,2))

        

        self.dropout2d = nn.Dropout2d()

        self.fc=nn.Sequential(

                nn.Linear(512*3*3,1024),

                nn.ReLU(inplace=True),

                nn.Dropout(0.4),

                nn.Linear(1024,512),

                nn.Dropout(0.4),

                nn.Linear(512, 1),

                nn.Sigmoid())

        

    def forward(self,x):

        x=self.conv1(x)

        x=self.conv2(x)

        x=self.conv3(x)

        x=self.conv4(x)

        x=self.conv5(x)

        x=x.view(x.shape[0],-1)

        x=self.fc(x)

        return x
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# tam CNN oluştur

model = CNN()

print(model)



# Move model to GPU if available

if train_on_gpu: model.cuda()
# Eğitilebilir parametre

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Eğitilebilir parametre sayısı: \n{}".format(pytorch_total_params))
# kayıp(loss) fonksiyonunu belirle (kategorik çapraz entropi kayıp)

criterion = nn.BCELoss()



# optizasyon için

optimizer = optim.Adam(model.parameters(), lr=0.00015)
# modeli eğitecek dönem sayısı

n_epochs = 20



valid_loss_min = np.Inf



# kayıp oranını görmek için

train_losses = []

valid_losses = []

val_auc = []

test_accuracies = []

valid_accuracies = []

auc_epoch = []



for epoch in range(1, n_epochs+1):



    # eğitim ve doğrulama kaybı takibi

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for data, target in train_loader:

        # CUDA varsa tensörleri GPU'ya taşı

        if train_on_gpu:

            data, target = data.cuda(), target.cuda().float()

        target = target.view(-1, 1)

        # optimize edilmiş tüm değişkenlerin gradyanlarını temizle

        optimizer.zero_grad()

        # ileri geçiş: girişleri modele geçirerek tahmini çıkışları hesapla

        output = model(data)

        # parti kaybını hesaplamak

        loss = criterion(output, target)

        # backward pass: model parametrelerine göre kaybın gradyanını hesaplar

        loss.backward()

        # tek bir optimizasyon adımı gerçekleştirme (parametre güncelleme)

        optimizer.step()

        # Egitim kaybını ve dogruluklarını güncelleme

        train_loss += loss.item()*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval()

    for data, target in valid_loader:

        # CUDA varsa tensörleri GPU'ya taşıyın

        if train_on_gpu:

            data, target = data.cuda(), target.cuda().float()

        # forward pass: Girdileri modele geçirerek tahmini çıktıları hesaplar

        target = target.view(-1, 1)

        output = model(data)

        # parti kaybını hesaplamak

        loss = criterion(output, target)

        # ortalama doğrulama kaybını güncelle

        valid_loss += loss.item()*data.size(0)

        #output = output.topk()

        y_actual = target.data.cpu().numpy()

        y_pred = output[:,-1].detach().cpu().numpy()

        val_auc.append(roc_auc_score(y_actual, y_pred))        

    

    # ortalama loss hesapla

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

    valid_auc = np.mean(val_auc)

    auc_epoch.append(np.mean(val_auc))

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

        

    # print training/validation statistics 

    print('Epoch: {} | Training Loss: {:.6f} | Validation Loss: {:.6f} | Validation AUC: {:.4f}'.format(

        epoch, train_loss, valid_loss, valid_auc))

    

    ##################

    # Early Stopping # En iyi modeli elde etmek için devirler için kaybı en az olanı yakalıyoruz

    ##################

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'best_model.pt')

        valid_loss_min = valid_loss
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



plt.plot(train_losses, label='Training loss')

plt.plot(valid_losses, label='Validation loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(frameon=False)
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



plt.plot(auc_epoch, label='Validation AUC/Epochs')

plt.legend("")

plt.xlabel("Epochs")

plt.ylabel("Area Under the Curve")

plt.legend(frameon=False)
# Daha sonra tahminlerde bulunmak için eğitimden öğrenilen en iyi parametreleri modelimize yüklemek

model.load_state_dict(torch.load('best_model.pt'))
# Degradeleri kapatma

model.eval()



preds = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.cuda(), target.cuda()

    output = model(data)

    pr = output.detach().cpu().numpy()

    for i in pr:

        preds.append(i)



# Gönderme dosyası oluştur    

sample_sub['label'] = preds
for i in range(len(sample_sub)):

    sample_sub.label[i] = np.float(sample_sub.label[i]) 
sample_sub.to_csv('submission.csv', index=False)

sample_sub.head()
def imshow(img):

    '''Görüntünün normalleştirilmesi ve görüntülenmesi için yardımcı işlev'''

    # unnormalize

    img = img / 2 + 0.5

    # Tensor görüntü ve görüntüden dönüştürme

    plt.imshow(np.transpose(img, (1, 2, 0)))
# bir grup test görüntüsü elde etmek için

dataiter = iter(test_loader)

images, labels = dataiter.next()

images = images.numpy() # görüntüleri gösterim için numpy'ye dönüştür



# ilgili etiketlerle birlikte resimleri topluca göstermek için

fig = plt.figure(figsize=(25, 4))

# görüntülenecek foto adedini yaz

for idx in np.arange(10):

    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    prob = "Cancer" if(sample_sub.label[idx] >= 0.5) else "Normal"

    ax.set_title('{}'.format(prob))
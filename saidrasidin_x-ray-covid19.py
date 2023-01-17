import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms, models
from torchvision.utils import make_grid
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import time
import os
from PIL import Image

! pip install jovian
import jovian
METADATA_COVID = '../input/covid-chest-xray/metadata.csv'
COVID_ROOT = '../input/covid-chest-xray/images'

PNEUMONIA_ROOT = '../input/chest-xray-pneumonia/chest_xray'
PNEUMONIA_TRAIN_ALL = PNEUMONIA_ROOT + '/train'
# PNEUMONIA_TRAIN = PNEUMONIA_ROOT+'/train/PNEUMONIA'
# NORMAL_TRAIN = PNEUMONIA_ROOT+'/train/NORMAL'
# PNEUMONIA_TEST = PNEUMONIA_ROOT+'/test/PNEUMONIA'
# NORMAL_TEST = PNEUMONIA_ROOT+'/test/NORMAL'

#target label
TARGET_LABEL = {0: 'NORMAL',
               1: 'PNEUMONIA',
               2: 'COVID19'}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_name = 'chest X-ray'
pneumonia_data = []
for dirname, _, filenames in os.walk(PNEUMONIA_TRAIN_ALL):
    for filename in filenames:
        if filename.endswith(".jpeg"):
            pneumonia_data.append(os.path.join(dirname, filename))

image = []
label = []
for i in range(len(pneumonia_data)):
    image.append(pneumonia_data[i].split('/')[-1])
    label.append(pneumonia_data[i].split('/')[-2])
# pneoumonia and normal data
df_pneumonia = pd.DataFrame({"label": label, "image_file": image})
df_pneumonia.head()
sns.countplot(df_pneumonia['label'])
plt.title('Pneumonia train dataset');
#covid19 data
df = pd.read_csv(METADATA_COVID)
df_pa = df.drop(df[df.view != 'PA'].index) #only take PA(from back to front film closer to chest) View
covid19 = df_pa[df_pa['finding']=='COVID-19'] #only take covid-19 label
covid19 = covid19[['finding', 'filename']] #take its label and image file
covid19.columns = (['label', 'image_file']) #change columns name same to pneumonia
#covid19[covid19['image_file'].str.endswith('.gz')]
covid19.reset_index(drop=True, inplace=True)
print('Data size:' , len(covid19))
covid19.head()
#takes normal and pneumonia only 300 images
normal = df_pneumonia[df_pneumonia['label']=='NORMAL']
normal = normal.sample(frac=1, axis=0, random_state=7).reset_index(drop=True) #suffle rows
normal = normal[:141] #same with covid19 data

pneumonia = df_pneumonia[df_pneumonia['label']=='PNEUMONIA']
pneumonia = pneumonia.sample(frac=1, axis=0, random_state=7).reset_index(drop=True)
pnuemonia = pneumonia[:141] #same with covid19 data

#concat all data (covid, pneumonia and normal)
all_data = pd.concat([normal, pnuemonia, covid19], ignore_index=True)
all_data = all_data.sample(frac=1, axis=0, random_state=7).reset_index(drop=True)
all_data.head(10)
sns.countplot(all_data['label'])
plt.title('All Datasets');
#split dataset
X_trainval, X_test, y_trainval, y_test = train_test_split(all_data['image_file'].values,
                                                      all_data['label'].values, test_size=0.05,
                                                      stratify=all_data['label'].values, random_state=7)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, stratify=y_trainval, test_size=0.1,
                                                  random_state=7)

len(X_train), len(X_val), len(X_test)
class Xray_split(Dataset):
    def __init__(self, root_dir_pnue, root_dir_covid, X, y, transform=None):
        self.pnue_root = root_dir_pnue
        self.covid_root = root_dir_covid
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image, label = self.X[idx], self.y[idx]
        if self.y[idx] == 'COVID-19':
            label = 2
            img_fname = str(self.covid_root) + "/" + str(image)
            img = Image.open(img_fname).convert("L")           
            if self.transform:
                img = self.transform(img)
        
        if self.y[idx] == 'NORMAL':
            label = 0
            img_fname = str(self.pnue_root) + "/NORMAL/" + str(image)
            img = Image.open(img_fname)
          
 
            if self.transform:
                img = self.transform(img)
        
        if self.y[idx] == 'PNEUMONIA':
            label = 1
            img_fname = str(self.pnue_root) + "/PNEUMONIA/" + str(image)
            img = Image.open(img_fname)
            
            if self.transform:
                img = self.transform(img)
                
        return img, int(label)
# mean = [0.4947]
# std = [0.2226]
mean = [0.0960, 0.0960, 0.0960]
std = [0.9341, 0.9341, 0.9341]

train_transform = transforms.Compose([transforms.Resize((512, 512)),
                                      transforms.Grayscale(3), #output 3 channel grayscale
                                      transforms.RandomResizedCrop((224, 224)),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                     ])

val_transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.Grayscale(3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                   ])

test_transform = transforms.Compose([transforms.Resize((512, 512)),
                                     transforms.Grayscale(3),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std),
                                   ])


train_set = Xray_split(PNEUMONIA_TRAIN_ALL, COVID_ROOT, X_train, y_train, train_transform)
val_set = Xray_split(PNEUMONIA_TRAIN_ALL, COVID_ROOT, X_val, y_val, val_transform)
test_set = Xray_split(PNEUMONIA_TRAIN_ALL, COVID_ROOT, X_test, y_test, test_transform)
#look the training data (already transformed)
fig = plt.figure(figsize=(20, 5))

for i in range(30):
    image, label = train_set[i]
    ax = fig.add_subplot(3, 10, i+1, xticks=[], yticks = [])
    ax.imshow(image[0], cmap='gray')
    ax.set_title(TARGET_LABEL[label], color=("green" if label == 0 else 'red'))
#find the mean and std

# nimages = 0
# mean = 0.
# std = 0.
# for batch, _ in train_loader:
#     # Rearrange batch to be the shape of [B, C, W * H]
#     batch = batch.view(batch.size(0), batch.size(1), -1)
#     # Update total number of images
#     nimages += batch.size(0)
#     # Compute mean and std here
#     mean += batch.mean(2).sum(0) 
#     std += batch.std(2).sum(0)

# # Final step
# mean /= nimages
# std /= nimages

# print(mean)
# print(std)
batch_size = 32 #have used 64 and 128 but 32 works better

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size*2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(20, 25))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
show_batch(train_loader)
#for get learning rate parameter
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#training loop
def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    torch.cuda.empty_cache()
    
    #save variabel
    train_losses = []
    test_losses = []
    train_scores = []
    val_score = []
    lrs = []

    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        train_score = 0
        
        #training loop#
        for image, label in train_loader:
            #training phase
            model.train()
            
            image = image.to(device); label = label.to(device);
            
            output = model(image)
            #accuracy calulcation
            ps = torch.exp(output)
            _, top_class = ps.topk(1, dim=1)
            correct = top_class == label.view(*top_class.shape)
            train_score += torch.mean(correct.type(torch.FloatTensor))
            #loss
            loss = criterion(output, label)
            #backward pass
            loss.backward()
            #update weight
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step() 
            lrs.append(get_lr(optimizer))
            running_loss += loss.item()
            
        else:
            model.eval()
            test_loss = 0
            scores = 0
            #validation loop#
            with torch.no_grad():
                for image, label in val_loader:
                    image = image.to(device); label = label.to(device);

                    output = model(image)

                    #accuracy calulcation
                    ps = torch.exp(output)
                    _, top_class = ps.topk(1, dim=1)
                    correct = top_class == label.view(*top_class.shape)
                    scores += torch.mean(correct.type(torch.FloatTensor))
                    #loss
                    loss = criterion(output, label)                                  
                    test_loss += loss.item()
            
            #calculation mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))
            train_scores.append(train_score/len(train_loader))
            val_score.append(scores/len(val_loader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Train Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}.. ".format(test_loss/len(val_loader)),
                  "Train acc Score: {:.3f}.. ".format(train_score/len(train_loader)),
                  "Val acc : {:.3f}.. ".format(scores/len(val_loader)),
                  "Lr: {:.4f} ".format(get_lr(optimizer)),
                  "Time: {:.2f}s" .format(time.time()-since)
                 )
        
    history = {'train_loss' : train_losses, 'val_loss': test_losses, 
               'train_acc': train_scores, 'val_acc':val_score, 'lrs': lrs}
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history

def plot_loss(history, n_epoch):
    epoch = [x for x in range(1, n_epoch+1)]
    plt.plot(epoch, history['train_loss'], label='Train_loss')
    plt.plot(epoch, history['val_loss'], label='val_loss')
    plt.title('Loss per epoch')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(); 
    plt.show()

def plot_score(history, n_epoch):
    epoch = [x for x in range(1, n_epoch+1)]
    plt.plot(epoch, history['train_acc'], label='Train_acc')
    plt.plot(epoch, history['val_acc'], label='val_acc')
    plt.title('Accuracy per epoch')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(); 
    plt.show()

def plot_lr(history):
    plt.plot(history['lrs'], label='learning rate')
    plt.title('One Cycle Learning Rate')
    plt.ylabel('Learning Rate')
    plt.xlabel('steps')
    plt.legend(); 
    plt.show()
output_label = 3

model_mobile = models.mobilenet_v2(pretrained=True)

model_mobile.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=output_label))

model_mobile.to(device);
model_mobile
max_lr = 0.0001
epoch = 20
weight_decay = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_mobile.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, 
                                            steps_per_epoch=len(train_loader))

history_mobile = fit(epoch, model_mobile, train_loader, val_loader, criterion, optimizer, sched)
torch.save(model_mobile.state_dict(),'mobilenet.pth')
plot_score(history_mobile, epoch)
plot_loss(history_mobile, epoch)
plot_lr(history_mobile)
jovian.reset()
jovian.log_hyperparams(arch='mobile_net', 
                       epochs=epoch, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay,
                       opt='Adam')

jovian.log_metrics(val_loss=history_mobile['val_loss'][-1], 
                   val_acc=history_mobile['val_acc'][-1].item(),
                   train_loss=history_mobile['train_loss'][-1],
                   time='7.79m')
model_resnet18 = models.resnet18(pretrained=True)
model_resnet18.fc = nn.Linear(512, output_label)

model_resnet18.to(device)
model_resnet18
max_lr = 0.0001
epoch = 20
weight_decay = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnet18.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, 
                                            steps_per_epoch=len(train_loader))

history_re18 = fit(epoch, model_resnet18, train_loader, val_loader, criterion, optimizer, sched)
torch.save(model_resnet18.state_dict(),'resnet18.pth')
plot_score(history_re18, epoch)
plot_loss(history_re18, epoch)
plot_lr(history_re18)
jovian.log_hyperparams(arch='resnet18', 
                       epochs=epoch, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       opt='Adam')

jovian.log_metrics(val_loss=history_re18['val_loss'][-1], 
                   val_acc=history_re18['val_acc'][-1].item(),
                   train_loss=history_re18['train_loss'][-1],
                   time='7.58m')
model_vgg16 = models.vgg16(pretrained=True)
model_vgg16.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=30, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5, inplace=False),
                                        nn.Linear(in_features=30, out_features=3, bias=True)
                                       )
model_vgg16.to(device)
model_vgg16
optimizer = optim.Adam(model_vgg16.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch, 
                                            steps_per_epoch=len(train_loader))

history_vgg16 = fit(epoch, model_vgg16, train_loader, val_loader, criterion, optimizer, sched)
torch.save(model_vgg16.state_dict(),'vgg16.pth')
plot_score(history_vgg16, epoch)
plot_loss(history_vgg16, epoch)
plot_lr(history_vgg16)
jovian.log_hyperparams(arch='VGG16', 
                       epochs=epoch, 
                       lr=max_lr, 
                       scheduler='one-cycle', 
                       weight_decay=weight_decay, 
                       opt='Adam')

jovian.log_metrics(val_loss=history_vgg16['val_loss'][-1], 
                   val_acc=history_vgg16['val_acc'][-1].item(),
                   train_loss=history_vgg16['train_loss'][-1],
                   time='8.44m')
def predict_dataset(dataset, model):
    model.eval()
    model.to(device)
    torch.cuda.empty_cache()
    predict = []
    y_true = []
    for image, label in dataset:
        #image = image.to(device); label= label.to(device)
        image = image.unsqueeze(0)
        image = image.to(device);
        
        output = model(image)
        ps = torch.exp(output)
        _, top_class = ps.topk(1, dim=1)
        
        predic = np.squeeze(top_class.cpu().numpy())
        predict.append(predic)
        y_true.append(label)
    return list(y_true), list(np.array(predict).reshape(1,-1).squeeze(0))

def report(y_true, y_predict, title='MODEL OVER TEST SET'):
    print(classification_report(y_true, y_predict))
    sns.heatmap(confusion_matrix(y_true, y_predict), annot=True)
    plt.yticks(np.arange(0.5, len(TARGET_LABEL)), labels=list(TARGET_LABEL.values()), rotation=0);
    plt.xticks(np.arange(0.5, len(TARGET_LABEL)), labels=list(TARGET_LABEL.values()), rotation=45)
    plt.title(title)
    plt.show()
    
def plot_predict(test_set, y_predict):
    """it takes longer time to plot, if you want it faster
    comment or delete tight_layout
    """
    fig = plt.figure(figsize=(20, 20))

    for i in range(len(test_set)):
        image, label = test_set[i]
        ax = fig.add_subplot(4, 6, i+1, xticks=[], yticks = [])
        ax.imshow(image[0], cmap='gray')
        ax.set_title("{}({})" .format(TARGET_LABEL[y_predict[i]], TARGET_LABEL[label]), 
                      color=("green" if y_predict[i] == label else 'red'), fontsize=12)

    plt.tight_layout() #want faster comment or delete this
    plt.show()
y_true, y_predict = predict_dataset(test_set, model_mobile)
report(y_true, y_predict, title='Mobilenet_v2 Over Test Set')
plot_predict(test_set, y_predict)
y_true, y_predict = predict_dataset(test_set, model_resnet18)
report(y_true, y_predict, 'Resnet18')
plot_predict(test_set, y_predict)
y_true, y_predict = predict_dataset(test_set, model_vgg16)
report(y_true, y_predict, 'VGG16')
plot_predict(test_set, y_predict)
jovian.commit(project=project_name, environment=None)
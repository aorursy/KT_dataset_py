import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import gc
import os
import cv2
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms,models
from tqdm import tqdm
# common variables
oc_path="../input/ocular-disease-recognition-odir5k/full_df.csv"
oc_img_path="../input/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images"
cat_normal="../input/cataractdataset/dataset/1_normal"
cat_cat="../input/cataractdataset/dataset/2_cataract"

device='cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
IMG_SIZE=256
BATCH=64
EPOCHS=5
print("Length of cataract images in cat dataset ", len(os.listdir(cat_cat)))
print("Length of normal images in cat dataset ", len(os.listdir(cat_normal)))
cat_df=pd.DataFrame(columns=["Path","cataract"])
for cat_imgs in glob.glob(cat_cat+"/*"):
    cat_df=cat_df.append({"Path":cat_imgs,"cataract":1},ignore_index=True)

for nor_imgs in glob.glob(cat_normal+"/*"):
    cat_df=cat_df.append({"Path":nor_imgs,"cataract":0},ignore_index=True)
cat_df=cat_df.sample(frac=1).reset_index(drop=True)
cat_df
df=pd.read_csv(oc_path)
df.head()
def cataract_or_not(txt):
    if "cataract" in txt:
        return 1
    else:
        return 0

def prepare_dataset(df_path,imgs_path):
    df=pd.read_csv(df_path)
    df['left_eye_cataract']=df["Left-Diagnostic Keywords"].apply(lambda x:cataract_or_not(x))
    df['right_eye_cataract']=df["Right-Diagnostic Keywords"].apply(lambda x:cataract_or_not(x))
    left_df=df.loc[:,['Left-Fundus','left_eye_cataract']].rename(columns={'left_eye_cataract':'cataract'})
    left_df['Path']=imgs_path+"/"+left_df['Left-Fundus']
    left_df=left_df.drop(['Left-Fundus'],1)

    right_df=df.loc[:,['Right-Fundus','right_eye_cataract']].rename(columns={'right_eye_cataract':'cataract'})
    right_df['Path']=imgs_path+"/"+right_df['Right-Fundus']
    right_df=right_df.drop(['Right-Fundus'],1)
    print('Number of left eye images')
    print(left_df['cataract'].value_counts())
    print('\nNumber of right eye images')
    print(right_df['cataract'].value_counts())
    train_df=pd.concat([right_df,left_df])
    return train_df
df['left_eye_cataract']=df["Left-Diagnostic Keywords"].apply(lambda x:cataract_or_not(x))
df['right_eye_cataract']=df["Right-Diagnostic Keywords"].apply(lambda x:cataract_or_not(x))
df.tail()
left_df=df.loc[:,['Left-Fundus','left_eye_cataract']].rename(columns={'left_eye_cataract':'cataract'})
left_df['Path']=oc_img_path+"/"+left_df['Left-Fundus']
left_df=left_df.drop(['Left-Fundus'],1)

right_df=df.loc[:,['Right-Fundus','right_eye_cataract']].rename(columns={'right_eye_cataract':'cataract'})
right_df['Path']=oc_img_path+"/"+right_df['Right-Fundus']
right_df=right_df.drop(['Right-Fundus'],1)

left_df.head()
right_df.head()
print('Number of left eye images')
print(left_df['cataract'].value_counts())
print('\nNumber of right eye images')
print(right_df['cataract'].value_counts())
def downsample(df):
    df = pd.concat([
        df.query('cataract==1'),
        df.query('cataract==0').sample(sum(df['cataract']), 
                                       random_state=42)
    ])
    return df

left_df = downsample(left_df)
right_df = downsample(right_df)

print('Number of left eye images')
print(left_df['cataract'].value_counts())
print('\nNumber of right eye images')
print(right_df['cataract'].value_counts())

ocu_df = pd.concat([left_df, right_df])
ocu_df.head()
train_df = pd.concat([cat_df, ocu_df], ignore_index=True)
len(df)
train_df=train_df.sample(frac=1.0)
train_df.head()
print("length of train_df ",len(train_df))
del left_df,right_df,cat_df,ocu_df
gc.collect()
train_df.cataract.value_counts()
# class_count_samples=list(train_df.cataract.value_counts())
# class_count_samples
train_df,test_df=train_test_split(train_df,test_size=0.12,shuffle=True,stratify=train_df.cataract)
train_df.cataract.value_counts()
test_df.cataract.value_counts()
class cat_dataset(torch.utils.data.Dataset):
    def __init__(self,df,transforms=None):
        self.df=df
        self.transforms=transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        
        img=cv2.imread(self.df.Path.iloc[idx])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.transforms:
            img=self.transforms(img)
        label=self.df.cataract.iloc[idx]
        return (img,label)

train_set=cat_dataset(train_df,transforms=transforms.Compose([
     transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
     transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
test_set=cat_dataset(test_df,transforms=transforms.Compose([
     transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
     transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))


# class_weights = 1./torch.Tensor(class_count_samples)
# train_target=list(train_df.cataract)
# train_samples_weight = [class_weights[class_id] for class_id in train_target]
# test_target=list(test_df.cataract)
# test_samples_weight = [class_weights[class_id] for class_id in test_target]


# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(train_df))
# test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_samples_weight, len(test_df))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH,shuffle=True)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH,shuffle=True)




# def get_freq(sampler):
#     vc={0:0,1:0}
#     for i in sampler:
#         vc[train_df.iloc[i].cataract]+=1
#     return vc
# print("freq labels in train loader is ",get_freq(train_sampler))
# print("freq labels in test loader is ",get_freq(test_sampler))
def plot_me(loader):
    """ Batch size must be more than 25 """
    imgs,lab=next(iter(loader))
    imgs=imgs[:25]
    lab=lab[:25]
    plt.figure(figsize=(15,10))
    for i in range(1,26):
        img=imgs[i-1].numpy().transpose(1,2,0)
        img=img*[0.5,0.5,0.5]+[0.5,0.5,0.5]
        labs=lab[i-1].numpy()
        labs="normal" if labs==0 else "cataract"
        plt.subplot(5,5,i,)
        plt.imshow(img)
        plt.title(labs)
        plt.axis('off')
    #plt.tight_layout()
plot_me(train_loader)
plot_me(val_loader)
model=models.densenet121(pretrained=True)
model.classifier=nn.Sequential(nn.Linear(1024,1),nn.Sigmoid())
model=model.to(device)
# img,lab=next(iter(train_loader))
# densenet_model(img.cuda())
# lab
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
crit=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=1, gamma=0.90)
def train(model, epochs, optimizer, train_loader, criterion,test_loader,sch=None):
    for epoch in range(1,epochs+1):
        # train
        total_loss = 0

        model.train()
        epoch_acc=0
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            acc = binary_acc(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            if sch:
                sch.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader),
            100. * batch_idx / len(train_loader), total_loss /len(train_loader)))
        #print('Train Accuracy for epoch {} is {} \n'.format(epoch,100. *correct/len(train_loader.dataset)))
        print(' Acc', epoch_acc/len(train_loader))

        # test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                

        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

train(model, 20, optimizer, train_loader, crit,val_loader,scheduler)
model.eval()
correct=0
total=0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        output = model(data)
        pred=(output>0.5).float()
        correct+=(pred==target).float().sum()
        total+=target.size(0)
    print(100* correct//total)
output
target

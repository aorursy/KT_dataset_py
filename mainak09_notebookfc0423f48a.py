!  pip install --upgrade efficientnet-pytorch
!pip install torch-lr-finder
import sys, os
import pandas as pd
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations import *
from tqdm.notebook import tnrange, tqdm
from torchvision import models
from efficientnet_pytorch import EfficientNet
import random
from sklearn.preprocessing import MinMaxScaler
import albumentations as A
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
train_images_path = '../input/jpeg-melanoma-256x256/train/'
test_images_path = '../input/siim-isic-melanoma-classification/jpeg/test/'
train_csv_path = '../input/jpeg-melanoma-256x256/train.csv'
test_csv_path = '../input/siim-isic-melanoma-classification/test.csv'  
train_df_0 = pd.read_csv(train_csv_path)
train_df_0.head()
def balance_df(df, class_name = ''):   
    max_size = df[class_name].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby(class_name):
        lst.append(group.sample(max_size-len(group), replace=True))
    frame_new = pd.concat(lst)
    df_balanced=frame_new
    return(df_balanced)

train_df = balance_df(train_df_0, 'target')
train_df['target'].value_counts()
def fill_missing_vals_w_mean(df, list_of_att):
    for att in list_of_att:
        df[att]=df[att].fillna(df[att].mean())
        
    return(df)
        
def drop_columns(df, list_of_columns):
    for column_ in list_of_columns:
        df.drop([column_], axis=1, inplace=True)
    return(df)

cols_drop = ['diagnosis', 
             'patient_id']

train_df = drop_columns(train_df, cols_drop)
train_df.head()
def scale_columns(df, list_of_cols):
    
    for x in list_of_cols:
        float_array = df[x].values.astype(float)
        min_max_scaler = MinMaxScaler()
        scaled_array = min_max_scaler.fit_transform(float_array.reshape(-1, 1))
        df[x] = scaled_array
    return(df)
    

#conv categorical in train_df
def convert_categorical(category, df):
    s = str(category)
    df = pd.get_dummies(df, columns=[s])
    return df


def encode_df(df):
    
    df_encoded = convert_categorical('sex', df)
    df_encoded = convert_categorical('anatom_site_general_challenge', df_encoded)
    df_encoded1 = convert_categorical('benign_malignant', df_encoded)
    return(df_encoded1)


df_encoded = encode_df(train_df)
df_encoded = drop_columns(df_encoded, ['sex_female', 
                                       'benign_malignant_benign'])
df_encoded = fill_missing_vals_w_mean(df_encoded, ['age_approx'])

df_scaled = scale_columns(df_encoded, ['age_approx', 
                                       'target', 
                                       'tfrecord', 
                                       'width', 
                                       'height',
                                       'sex_male', 
                                       'anatom_site_general_challenge_head/neck',
                                       'anatom_site_general_challenge_lower extremity',
                                       'anatom_site_general_challenge_oral/genital',
                                       'anatom_site_general_challenge_palms/soles',
                                       'anatom_site_general_challenge_torso',
                                       'anatom_site_general_challenge_upper extremity',
                                       'benign_malignant_malignant'])

df_scaled.to_csv('./train_cleaned.csv')
df_scaled['target'].value_counts()
df_scaled.head()
other_df = drop_columns(df_encoded, ['target', 'image_name'])
other_df.to_csv('./other_df.csv')
other_df.head()
def image_to_nparray(path):
    i = plt.imread(path)    
    return(np.array(i))
class AdvancedHairAugmentation:
    def __init__(self, hairs: int = 4, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img
    
    
class Microscope:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'
class MyDataset(Dataset):
    
    '''
    csv_file_path : path to csv
    images_path : path to images
    transform (callable, optional): Optional transform to be applied on a sample.
    '''
    
    def __init__(self, csv_file_path ,other_csv_path, images_path, augmentation=None, transform=None):

        self.transform = transform
        self.augmentation = augmentation
        
        self.images_path = images_path
        self.d = pd.read_csv(csv_file_path)[0:]
        print(self.d.head())
        self.other_df = pd.read_csv(other_csv_path)[0:]
        self.smol_img_paths =self.d[self.d.columns[1]]
        
        
   
        
    def __getitem__(self, index): 
        
        full_img_path = self.images_path+str(self.d.iloc[index]['image_name']) + '.jpg'
        #print(full_img_path)
        img_arr = image_to_nparray(full_img_path)
        
        class_name2 = self.d.iloc[index]['target']
        other_part2=np.array(self.other_df.iloc[index][0:].values, dtype=np.float32)

        if self.augmentation is not None:
            img  = self.augmentation(image = img_arr)
            img2 = img["image"]

        if self.transform is not None:
            img_ret = self.transform(img2)
        
        return(torch.tensor(int(class_name2)), ((img_ret), torch.tensor([other_part2])) )
                                                                                   
    def __len__(self):
        return (len(self.d))
data = MyDataset('./train_cleaned.csv', 
                 './other_df.csv',
                 train_images_path,
                 augmentation = Compose([ 
                                        ]),
                transform = transforms.Compose([AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs'),
                                                Microscope(p=0.7),
                                                transforms.ToPILImage(),
                                                transforms.Resize((224,224),interpolation = Image.NEAREST),
                                                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    
                                                ]))
                

train_data, val_data = torch.utils.data.random_split(data, [len(data)-500, 500])
train_loader = torch.utils.data.DataLoader(
    data
    ,batch_size=10
    ,shuffle=False
    
)

val_loader = torch.utils.data.DataLoader(
    val_data
    ,batch_size=1000
    ,shuffle=False
)
for label, x in (train_loader):
    #print(label)
    images,features = x
    i=np.transpose(images[0], (1, 2, 0)) 
    #print(i)
    plt.imshow(i, interpolation='sinc')
    plt.show()
    print(label[0])
    break
    
    #x = ((I,I,I,I), (F,F,F,F))
fig, axs = plt.subplots(2, 2, figsize=(10,10))
fig = plt.figure(figsize=(10,3))
trans = transforms.ToPILImage()
class_list = ["no", "yes"]

for i in range(2):
    for j in range(2):
        for labels,x in (train_loader):

            images,features = x
            img=np.transpose(images[0], (1, 2, 0)) 
            #print(i)
            class_number = labels[0].item()
            #print(class_number)
            im_label=class_list[class_number]
            axs[j,i].imshow(img, interpolation='sinc')
            #axs[j,i].title.set_text(str(class_number))
            
            break
plt.show()
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
arch = EfficientNet.from_pretrained('efficientnet-b1')
class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = arch
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
            
        self.lower_nn = nn.Sequential(nn.Linear(n_meta_features,50))
                             
        
        self.combined_layer = nn.Linear(500+50, 1)

    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        #print(meta)
        x = x.to(device)
        meta=meta.to(device)
        cnn_features = self.arch(x)
        meta_features = self.lower_nn(meta)
        features = torch.cat((cnn_features.squeeze().detach(), meta_features.squeeze().detach()), dim=1)
        output = (self.combined_layer(features))
        return output


arch = EfficientNet.from_pretrained('efficientnet-b1')

network = Net(arch=arch, n_meta_features=13)
network = network.to(device)
print(network)

optimizer = optim.Adam(network.parameters(), lr = 0.01) 
criterion = nn.BCEWithLogitsLoss()
scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                           mode='max',
                           patience=2, 
                           verbose=True, 
                           factor=0.2)


loss_list=[]
acc_list=[]
val_loss_list=[]
val_acc_list=[]
train_loader_2 = torch.utils.data.DataLoader(
    train_data
    ,batch_size=64
    ,shuffle=True
    ,num_workers=4

)

val_loader_2 = torch.utils.data.DataLoader(
    val_data
    ,batch_size=64
    ,shuffle=False
    ,num_workers=4
)
for epoch in tnrange(1): 
    
    total_loss = 0
    total_correct = 0
    total_loss2 = 0
    total_correct2 = 0
    

    for labels, x_train in tqdm(train_loader_2): # Get Batch

        images, feature_tensors = x
        images = images.to(device)
        feature_tensors = feature_tensors.to(device)
        labels=labels.to(device)
        
        optimizer.zero_grad()
        preds = network(x_train)
        preds= preds.to(device)
        preds2 = torch.round(torch.sigmoid(preds))
        
        labels = labels.type_as(preds2)
        loss = criterion(preds2.flatten(), labels)
        loss_list.append(loss)
        num_correct=get_num_correct(preds, labels)
        
        loss.backward()
        optimizer.step() 
        total_correct+=num_correct
    
    acc_list.append(total_correct)
        
    print("training_corr :",total_correct)
    
    with torch.no_grad():
        network.eval()
        for labels, x_val in tqdm(val_loader_2): # Get Batch

            images, feature_tensors = x_val
            images = images.to(device)
            feature_tensors = feature_tensors.to(device)
            labels=labels.to(device)
            preds = network(x_val)
            preds= preds.to(device)
            preds1 = torch.round(torch.sigmoid(preds))

            labels = labels.type_as(preds)
            loss2 = criterion(preds1.flatten(), labels)
            val_loss_list.append(loss2)
            val_num_correct=get_num_correct(preds, labels)
            total_correct2+=val_num_correct

        val_acc_list.append(total_correct2)

        print("val_corr :", total_correct2)
        
        
        #why isnt the accuracy going up?
        
with torch.no_grad():
        network.eval()
        total_correct2=0
        for labels, x_val in tqdm(val_loader_2): # Get Batch

            images, feature_tensors = x_val
            images = images.to(device)
            feature_tensors = feature_tensors.to(device)
            labels=labels.to(device)
            preds = network(x_val)
            preds= preds.to(device)
            preds1 = torch.round(torch.sigmoid(preds))

            labels = labels.type_as(preds)
            loss2 = criterion(preds1.flatten(), labels)
            val_loss_list.append(loss2)
            val_num_correct=get_num_correct(preds, labels)
            total_correct2+=val_num_correct
        print(total_correct2)
plt.plot(loss_list)

# y 13 metafeatures?
# try scaling all values


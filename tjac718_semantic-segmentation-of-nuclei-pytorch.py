import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.manual_seed(42)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import sys
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
seed = 42
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
with zipfile.ZipFile('../input/data-science-bowl-2018/stage1_train.zip') as z:
    z.extractall('stage1_train')

with zipfile.ZipFile('../input/data-science-bowl-2018/stage1_test.zip') as z:
    z.extractall('stage1_test')
    
with zipfile.ZipFile('../input/data-science-bowl-2018/stage1_test.zip') as z:
    z.extractall('stage2_test_final')    
TRAIN_PATH = '/kaggle/working/stage1_train/'
TEST_PATH = '/kaggle/working/stage1_test/'
TEST_PATH_2 = '/kaggle/working/stage2_test_final/'

train_files = next(os.walk(TRAIN_PATH))[1]
test_files = next(os.walk(TEST_PATH))[1]
test_files_2 = next(os.walk(TEST_PATH_2))[1]
test_files_final = test_files + test_files_2
X_train = np.zeros((len(train_files), 128, 128, 3), dtype = np.uint8)
Y_train = np.zeros((len(train_files), 128, 128, 1), dtype = np.bool)
X_test = np.zeros((len(test_files), 128, 128, 3), dtype = np.uint8)
X_test_2 = np.zeros((len(test_files), 128, 128, 3), dtype = np.uint8)

print('Getting training data...')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_files), total = len(train_files)):
    img_path = TRAIN_PATH + id_ + '/images/' + id_ + '.png'
    img = imread(img_path)[:,:,:3]
    img = resize(img, (128, 128), mode='constant', preserve_range=True)
    X_train[n] = img
    
    masks_path = TRAIN_PATH + id_ + '/masks/'
    mask = np.zeros((128, 128, 1))
    mask_images = next(os.walk(masks_path))[2]
    for mask_id in mask_images:
        mask_path = masks_path + mask_id
        mask_ = imread(mask_path)
        mask_ = np.expand_dims(resize(mask_, (128, 128), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

print('Getting testing data for stage 1...')
sys.stdout.flush()

sizes_test = []
for n, id_ in tqdm(enumerate(test_files), total = len(test_files)):
    img_path = TEST_PATH + id_ + '/images/' + id_ + '.png'
    img = imread(img_path)[:,:,:3]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (128, 128), mode='constant', preserve_range=True)
    X_test[n] = img

print('Getting testing data for stage 2...')
sys.stdout.flush()

sizes_test_2 = []
for n, id_ in tqdm(enumerate(test_files_2), total = len(test_files_2)):
    img_path = TEST_PATH + id_ + '/images/' + id_ + '.png'
    img = imread(img_path)[:,:,:3]
    sizes_test_2.append([img.shape[0], img.shape[1]])
    img = resize(img, (128, 128), mode='constant', preserve_range=True)
    X_test_2[n] = img

print('Done!')
X_test_final = np.concatenate((X_test, X_test_2), axis = 0)
sizes_test_final = sizes_test + sizes_test_2
print(X_test_final.shape)
class Nuc_Seg(Dataset):
    def __init__(self, images_np, masks_np):
        self.images_np = images_np
        self.masks_np = masks_np
    
    def transform(self, image_np, mask_np):
        ToPILImage = transforms.ToPILImage()
        image = ToPILImage(image_np)
        mask = ToPILImage(mask_np.astype(np.int32))
        
        image = TF.pad(image, padding = 20, padding_mode = 'reflect')
        mask = TF.pad(mask, padding = 20, padding_mode = 'reflect')
        
        angle = random.uniform(-10, 10)
        width, height = image.size
        max_dx = 0.1 * width
        max_dy = 0.1 * height
        translations = (np.round(random.uniform(-max_dx, max_dx)), np.round(random.uniform(-max_dy, max_dy)))
        scale = random.uniform(0.8, 1.2)
        shear = random.uniform(-0.5, 0.5)
        image = TF.affine(image, angle = angle, translate = translations, scale = scale, shear = shear)
        mask = TF.affine(mask, angle = angle, translate = translations, scale = scale, shear = shear)
        
        image = TF.center_crop(image, (128, 128))
        mask = TF.center_crop(mask, (128, 128))
        
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask
        
    def __len__(self):
        return len(self.images_np)
    
    def __getitem__(self, idx):
        image_np = self.images_np[idx]
        mask_np = self.masks_np[idx]
        image, mask = self.transform(image_np, mask_np)
        
        return image, mask    
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = seed)

train_dataset = Nuc_Seg(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
valid_dataset = Nuc_Seg(X_val, Y_val)
valid_loader = DataLoader(valid_dataset, batch_size = 16, shuffle = True)
fig, axis = plt.subplots(2, 2)
axis[0][0].imshow(X_train[0].astype(np.uint8))
axis[0][1].imshow(np.squeeze(Y_train[0]).astype(np.uint8))
axis[1][0].imshow(X_val[0].astype(np.uint8))
axis[1][1].imshow(np.squeeze(Y_val[0]).astype(np.uint8))
%matplotlib inline

for ex_img, ex_mask in train_loader:
    
    img = np.array(TF.to_pil_image(ex_img[0]))
    mask = np.array(TF.to_pil_image(ex_mask[0]))
    
    fig, (axis_1, axis_2) = plt.subplots(1, 2)
    axis_1.imshow(img.astype(np.uint8))
    axis_2.imshow(mask.astype(np.uint8))
    
    break
def iou(pred, target, n_classes = 2):
    
    iou = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
      pred_inds = pred == cls
      target_inds = target == cls
      intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
      union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    
      if union == 0:
        iou.append(float('nan'))  # If there is no ground truth, do not include in evaluation
      else:
        iou.append(float(intersection) / float(max(union, 1)))
     
    return sum(iou)
def iou_metric(y_pred, y_true, n_classes = 2):
    miou = []
    for i in np.arange(0.5, 1.0, 0.05):
        y_pred_ = (y_pred > i)
        iou_init = iou(y_pred_, y_true, n_classes = n_classes)
        miou.append(iou_init)
    
    return sum(miou)/len(miou)
class UNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.drop1_1 = nn.Dropout2d(0.1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.drop2_1 = nn.Dropout2d(0.1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.drop3_1 = nn.Dropout2d(0.2)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.drop4_1 = nn.Dropout2d(0.2)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        
        self.conv5_1 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.drop5_1 = nn.Dropout2d(0.3)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        
        self.conv_trans6_1 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = (2, 2))
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size = 3, padding = 1)
        self.drop6_1 = nn.Dropout2d(0.2)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        
        self.conv_trans7_1 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = (2, 2))
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size = 3, padding = 1)
        self.drop7_1 = nn.Dropout2d(0.2)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        
        self.conv_trans8_1 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = (2, 2))
        self.conv8_1 = nn.Conv2d(64, 32, kernel_size = 3, padding = 1)
        self.drop8_1 = nn.Dropout2d(0.1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        
        self.conv_trans9_1 = nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = (2, 2))
        self.conv9_1 = nn.Conv2d(32, 16, kernel_size = 3, padding = 1)
        self.drop9_1 = nn.Dropout2d(0.1)
        self.conv9_2 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1)
        
        self.conv10 = nn.Conv2d(16, 1, kernel_size = 3, padding = 1)
    
    def forward(self, s):
        
        c1 = F.elu(self.conv1_1(s))
        c1 = self.drop1_1(c1)
        c1 = F.elu(self.conv1_2(c1))
        p1 = F.max_pool2d(c1, kernel_size = (2, 2), stride = 2)
        
        c2 = F.elu(self.conv2_1(p1))
        c2 = self.drop2_1(c2)
        c2 = F.elu(self.conv2_2(c2))
        p2 = F.max_pool2d(c2, kernel_size = (2, 2), stride = 2)
        
        c3 = F.elu(self.conv3_1(p2))
        c3 = self.drop3_1(c3)
        c3 = F.elu(self.conv3_2(c3))
        p3 = F.max_pool2d(c3, kernel_size = (2, 2), stride = 2)
        
        c4 = F.elu(self.conv4_1(p3))
        c4 = self.drop4_1(c4)
        c4 = F.elu(self.conv4_2(c4))
        p4 = F.max_pool2d(c4, kernel_size = (2, 2), stride = 2)
        
        c5 = F.elu(self.conv5_1(p4))
        c5 = self.drop5_1(c5)
        c5 = F.elu(self.conv5_2(c5))
        
        u6 = self.conv_trans6_1(c5)
        u6 = torch.cat((u6, c4), axis = 1)
        c6 = F.elu(self.conv6_1(u6))
        c6 = self.drop6_1(c6)
        c6 = F.elu(self.conv6_2(c6))
        
        u7 = self.conv_trans7_1(c6)
        u7 = torch.cat((u7, c3), axis = 1)
        c7 = F.elu(self.conv7_1(u7))
        c7 = self.drop7_1(c7)
        c7 = F.elu(self.conv7_2(c7))
    
        u8 = self.conv_trans8_1(c7)
        u8 = torch.cat((u8, c2), axis = 1)
        c8 = F.elu(self.conv8_1(u8))
        c8 = self.drop8_1(c8)
        c8 = F.elu(self.conv8_2(c8))
        
        u9 = self.conv_trans9_1(c8)
        u9 = torch.cat((u9, c1), axis = 1)
        c9 = F.elu(self.conv9_1(u9))
        c9 = self.drop9_1(c9)
        c9 = F.elu(self.conv9_2(c9))
        
        output = torch.sigmoid(self.conv10(c9))
        
        return output        
model = UNet()
model = model.float()
model = model.to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)
opt = optim.Adam(model.parameters(), lr = 0.001)
loss_func = nn.BCELoss()
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience = 3, verbose = 1)
def fit(model, epochs, opt, loss_func, train_loader, valid_loader, alpha):

    for epoch in range(epochs):
        
        #Going into training mode
        model.train()
        
        train_loss = 0 
        iou = 0
        
        for image, mask in train_loader:
            image = image.to(device)   #Passing the input mini-batch to the GPU
            mask = mask.to(device)   #Passing the label mini-batch to the GPU
            opt.zero_grad()      #Setting the grads to zero to avoid accumulation of gradients
            out = model(image.float())
            loss = loss_func(out.float(), mask.float())    
            train_loss += loss
            
            iou += iou_metric(out, mask)
            iou_rev = 16 - iou_metric(out, mask)
            loss += alpha * iou_rev
            
            loss.backward()
            opt.step()
        
        lr_scheduler.step(train_loss/len(train_loader))   #Setting up lr decay  
        
        model.eval()            #Going into eval mode                            
        with torch.no_grad():   #No backprop
            valid_loss = 0
            valid_iou = 0
            
            for image_val, mask_val in valid_loader:
                image_val = image_val.to(device)  
                mask_val = mask_val.to(device)
                out_val = model(image_val.float())
                valid_loss += loss_func(out_val.float(), mask_val.float())
                
                valid_iou += iou_metric(out_val, mask_val)
        
        print("Epoch ", epoch + 1, " Training Loss: ", train_loss/len(train_loader), "CV Loss: ", valid_loss/len(valid_loader))
        print("Training IoU: ", iou/len(train_loader), "CV IoU: ", valid_iou/len(valid_loader))
fit(model, 30, opt, loss_func, train_loader, valid_loader, 5)
%matplotlib inline

for ex_img, ex_mask in train_loader:
    
    img = ex_img[1].to(device)
    img.unsqueeze_(0)
    mask_pred = model(img.float())
    mask_pred = mask_pred.cpu()
    mask_pred = (mask_pred > 0.75)
    mask_true = ex_mask[1]
    
    img = TF.to_pil_image(mask_pred.float().squeeze(0))
    mask = TF.to_pil_image(mask_true)
    
    img = np.array(img)
    mask = np.array(mask)
    
    fig, (axis_1, axis_2) = plt.subplots(1, 2)
    axis_1.imshow(img.astype(np.uint8))
    axis_2.imshow(mask.astype(np.uint8))
    
    break
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
with torch.no_grad():
    
    test = torch.from_numpy(X_test)
    test = test/255.0
    test = test.permute(0, 3, 1, 2)
    test = test.to(device)
    preds = model(test.float())
    preds = preds.permute(0, 2, 3, 1) 
    print(preds.shape)

    preds = preds*255.0
    preds = preds.cpu().numpy()

print(preds.shape)
preds_t = (preds > 0.5).astype(np.uint8)

preds_upsampled = []
for i in range(len(preds)):
    preds_upsampled.append(resize(np.squeeze(preds[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))    
new_test_ids = []
rles = []
for n, id_ in enumerate(test_files):
    rle = list(prob_to_rles(preds_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)
with torch.no_grad():
    
    test = torch.from_numpy(X_test_2)
    test = test/255.0
    test = test.permute(0, 3, 1, 2)
    test = test.to(device)
    preds_2 = model(test.float())
    preds_2 = preds_2.permute(0, 2, 3, 1) 
    print(preds_2.shape)

    preds_2 = preds_2*255.0
    preds_2 = preds_2.cpu().numpy()

print(preds_2.shape)
preds_t_2 = (preds_2 > 0.5).astype(np.uint8)

preds_upsampled_2 = []
for i in range(len(preds_2)):
    preds_upsampled_2.append(resize(np.squeeze(preds_2[i]), 
                                       (sizes_test_2[i][0], sizes_test_2[i][1]), 
                                       mode='constant', preserve_range=True))    
new_test_ids_2 = []
rles_2 = []
for n, id_ in enumerate(test_files_2):
    rle = list(prob_to_rles(preds_upsampled_2[n]))
    rles_2.extend(rle)
    new_test_ids_2.extend([id_] * len(rle))
sub_2 = pd.DataFrame()
sub_2['ImageId'] = new_test_ids_2
sub_2['EncodedPixels'] = pd.Series(rles_2).apply(lambda x: ' '.join(str(y) for y in x))
sub_2.to_csv('sub-dsbowl2018-2.csv', index=False)
with torch.no_grad():
    
    test = torch.from_numpy(X_test_final)
    test = test/255.0
    test = test.permute(0, 3, 1, 2)
    test = test.to(device)
    preds_final = model(test.float())
    preds_final = preds_final.permute(0, 2, 3, 1) 
    print(preds_final.shape)

    preds_final = preds_final*255.0
    preds_final = preds_final.cpu().numpy()

print(preds_final.shape)
preds_t_final = (preds_final > 0.5).astype(np.uint8)

preds_upsampled_final = []
for i in range(len(preds_final)):
    preds_upsampled_final.append(resize(np.squeeze(preds_final[i]), 
                                       (sizes_test_final[i][0], sizes_test_final[i][1]), 
                                       mode='constant', preserve_range=True))    
new_test_ids_final = []
rles_final = []
for n, id_ in enumerate(test_files_final):
    rle = list(prob_to_rles(preds_upsampled_final[n]))
    rles_final.extend(rle)
    new_test_ids_final.extend([id_] * len(rle))
sub_final = pd.DataFrame()
sub_final['ImageId'] = new_test_ids_2
sub_final['EncodedPixels'] = pd.Series(rles_final).apply(lambda x: ' '.join(str(y) for y in x))
sub_final.to_csv('sub-dsbowl2018-final.csv', index=False)
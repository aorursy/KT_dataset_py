#requirements 

!pip install segmentation_models_pytorch
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

import segmentation_models_pytorch as smp

from sklearn.model_selection import train_test_split

from PIL import Image

import time

from skimage.morphology import binary_opening, disk, label
#create dataset

class ShipDatabaseSegmation(torch.utils.data.Dataset):

    def __init__(self,in_df,root_path,transforms=None):

        imagesIds = in_df['ImageId'].tolist()

        self.image_ids =  list(set(imagesIds))

        self.in_df = in_df

        self.root_path = root_path

        self.transforms = transforms

        

    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self,idx):

        ImageId = self.image_ids[idx]

        img = Image.open(self.root_path + "/"+ ImageId)

        img_masks = self.in_df.loc[self.in_df['ImageId'] == ImageId, 'EncodedPixels'].tolist()

        all_masks = np.zeros((768, 768))

        for mask in img_masks:

            all_masks += rle_decode(mask)

        

        #all_masks = np.expand_dims(all_masks,axis=0)

        if self.transforms is not None:

            img = self.transforms(img)

            all_masks = self.transforms(all_masks)



        return img,all_masks
def multi_rle_encode(img, **kwargs):

    '''

    Encode connected regions as separated masks

    '''

    labels = label(img)

    if img.ndim > 2:

        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]

    else:

        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]



# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    if np.max(img) < min_max_threshold:

        return '' ## no need to encode if it's all zeros

    if max_mean_threshold and np.mean(img) > max_mean_threshold:

        return '' ## ignore overfilled mask

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    if np.max(img) < min_max_threshold:

        return '' ## no need to encode if it's all zeros

    if max_mean_threshold and np.mean(img) > max_mean_threshold:

        return '' ## ignore overfilled mask

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_decode(mask_rle, shape=(768, 768)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros([shape[0]*shape[1],1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction
# helper function for data visualization

def visualize(**images):

    """PLot images in one row."""

    n = len(images)

    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):

        plt.subplot(1, n, i + 1)

        plt.xticks([])

        plt.yticks([])

        plt.title(' '.join(name.split('_')).title())

        plt.imshow(image)

    plt.show()
submission = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')

submission = submission.dropna()

submission.head()
unique_img_ids = submission.groupby('ImageId').size().reset_index(name='counts')

train_ids, valid_ids = train_test_split(unique_img_ids, 

                 test_size = 0.05, 

                 stratify = unique_img_ids['counts'],

                 random_state=42

                )

train_df = pd.merge(submission, train_ids)

valid_df = pd.merge(submission, valid_ids)

print(train_df.shape[0], 'training masks')

print(valid_df.shape[0], 'validation masks')
# #take only 1000 ( maybe with less examples we can get same results)

train_df = train_df[:1000]

valid_df = valid_df[:1000]
transforms = torchvision.transforms.Compose([

    torchvision.transforms.ToTensor() 

])

ship_dataset_train = ShipDatabaseSegmation(train_df,"../input/airbus-ship-detection/train_v2",transforms=transforms)

ship_dataset_valid = ShipDatabaseSegmation(valid_df,"../input/airbus-ship-detection/train_v2",transforms=transforms)



train_loader = torch.utils.data.DataLoader(ship_dataset_train, batch_size=5, shuffle=True, num_workers=8)

valid_loader = torch.utils.data.DataLoader(ship_dataset_valid, batch_size=1, shuffle=False, num_workers=4)
examples = iter(valid_loader)

for i in range(10):

    img,mask = next(examples)

    visualize(

    image=img.squeeze().permute(1,2,0),

    mask = mask.squeeze(),

    )
loss = nn.BCEWithLogitsLoss()

loss.__name__ = "bceWithLogitLoss"



device = "cuda"

metrics = [

    smp.utils.metrics.IoU(threshold=0.5),

]

def train(model,number_epoches,model_name,device='cuda',lr=0.0001):

    optimizer = torch.optim.Adam([ 

    dict(params=model.parameters(), lr=lr),])



    # create epoch runners 

    # it is a simple loop of iterating over dataloader`s samples

    train_epoch = smp.utils.train.TrainEpoch(

        model, 

        loss=loss, 

        metrics=metrics, 

        optimizer=optimizer,

        device=device,

        verbose=True,

    )



    valid_epoch = smp.utils.train.ValidEpoch(

        model, 

        loss=loss, 

        metrics=metrics, 

        device=device,

        verbose=True,

    )

    print("Start train model : ",model_name)

    

    max_score = 0

    for i in range(0, number_epoches):

        print('\nEpoch: {}'.format(i))

        train_logs = train_epoch.run(train_loader)

        valid_logs = valid_epoch.run(valid_loader)



        # do something (save model, change lr, etc.)

        if max_score < valid_logs['iou_score']:

            max_score = valid_logs['iou_score']

            torch.save(model, './'+model_name+'_best_model.pth')

            print('Model saved!')



        if i == 25:

            optimizer.param_groups[0]['lr'] = 1e-5

            print('Decrease decoder learning rate to 1e-5!')

        if i % 5 == 0:

            torch.save(model, './'+model_name+'_epoch_'+str(i)+'_'+str(time.time())+'.pth')

            print("save model !")

    

    
encoder = 'resnet50'

encoder_weights ='imagenet'

number_epoches = 20

device = 'cuda'

lr = 0.0001

Unet = smp.Unet(encoder,encoder_weights=encoder_weights)

FPN = smp.FPN(encoder,encoder_weights=encoder_weights)





# for model_name,model in zip(["Unet","FPN"],[Unet,FPN]):

#     train(model,number_epoches,model_name,device=device,lr=lr) # train take 1 hour and half each model so in commands for submit

FPN = torch.load("../input/ship-detection-unet-fpn/FPN_best_model.pth")

Unet = torch.load("../input/ship-detection-unet-fpn/Unet_best_model.pth")
test_epoch_FPN = smp.utils.train.ValidEpoch(

        FPN, 

        loss=loss, 

        metrics=metrics, 

        device=device,

        verbose=True,

    )

test_epoch_Unet = smp.utils.train.ValidEpoch(

        Unet, 

        loss=loss, 

        metrics=metrics, 

        device=device,

        verbose=True,

    )
valid_logs = test_epoch_FPN.run(valid_loader) #  on data validation

train_logs = test_epoch_FPN.run(train_loader) #  on data train
valid_logs = test_epoch_Unet.run(valid_loader) # on data validation

train_logs = test_epoch_Unet.run(train_loader) # on data train
valid_loader = torch.utils.data.DataLoader(ship_dataset_valid, batch_size=1, shuffle=False, num_workers=4)

test = iter(valid_loader)



for i in range(30):

    img,mask = next(test)

    pred = Unet(img.cuda())

    pred = pred.detach().cpu().double()

    visualize(

    image=img.squeeze().permute(1,2,0),

    mask = mask.squeeze(),

    Predict=pred.squeeze(),

    )
def pred_image_from_path_encode(root_path,image_names,model,device):

    out_pred_rows = []

    for img_name in image_names:

        c_img = Image.open(root_path+"/"+img_name)

        covnertTensor = torchvision.transforms.transforms.ToTensor()

        c_img = covnertTensor(c_img)

        c_img = c_img.unsqueeze(0)

        if device == 'cuda':

            c_img = c_img.cuda()

        cur_seg = model(c_img)

        cur_seg = cur_seg.squeeze(0).squeeze(0).detach().cpu().numpy()

        cur_seg[cur_seg < 0.5] = 0 

        cur_seg[cur_seg >= 0.5] = 1

        cur_rles = multi_rle_encode(cur_seg,max_mean_threshold=1.0)

        if len(cur_rles)>0:

            for c_rle in cur_rles:

                out_pred_rows += [{'ImageId': img_name, 'EncodedPixels': c_rle}]

        else:

            out_pred_rows += [{'ImageId': img_name, 'EncodedPixels': None}]

    

    return out_pred_rows

    



test_image_dir = "../input/airbus-ship-detection/test_v2"

images_names = np.array(os.listdir(test_image_dir))

print(images_names)

print(len(images_names), 'test images found')

out_pred_rows = pred_image_from_path_encode(test_image_dir,images_names[:200],Unet,device)

sub = pd.DataFrame(out_pred_rows)

sub.columns = ['ImageId', 'EncodedPixels']

sub = sub[sub.EncodedPixels.notnull()]

sub.head(5)
sub.to_csv('submission.csv', index=False)
# validation submission.csv
submission = pd.read_csv('./submission.csv')

submission = submission.dropna()

submission.head()




dataset_test = ShipDatabaseSegmation(submission,"../input/airbus-ship-detection/test_v2",transforms=transforms)

test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=8)

examples = iter(test_loader)



for img,mask in examples:

    visualize(

    image=img.squeeze().permute(1,2,0),

    predict = mask.squeeze(),

    )
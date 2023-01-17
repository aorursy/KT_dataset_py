import os

print(os.listdir("../input"))



import numpy as np

import pandas as pd



import cv2

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

print('Dimensions:', train_df.shape, test_df.shape, sub.shape)
test_df['diagnosis'] = 0

test_df.head()
labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

explode = (0.1, 0, 0, 0, 0.1)



fig, ax = plt.subplots(figsize=(7,7))

ax.pie(train_df['diagnosis'].value_counts(), explode=explode, labels=labels,

      autopct='%1.1f%%',shadow=True, startangle=90);



ax.set_title('Distibution the presence of diabetic retinopathu in each image on a scale of 0 to 4',

            fontdict={

                'fontsize':15

            });
fig = plt.figure(figsize=(15, 10))

for label in sorted(train_df['diagnosis'].unique()):

    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == label].sample(5).iterrows()):

        ax = fig.add_subplot(5, 5, label * 5 + i + 1, xticks=[], yticks=[])

        img = cv2.imread(f"../input/aptos2019-blindness-detection/train_images/{row['id_code']}.png")

        plt.imshow(img[...,[2,1,0]])

        ax.set_title(f'Label: {label}')
# code from https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping

# Image processing

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

    

    

def load_ben_color(path, sigmaX=10 ):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (512, 512))

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        

    return image
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader 



import torchvision

from torchvision.transforms import transforms
# Dataset

class RetinaDataset(Dataset):

    def __init__(self, df, img_dir, transforms):

        self.df = df

        self.img_dir = img_dir

        self.transforms = transforms

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        img_name = os.path.join(self.img_dir,

                               self.df.iloc[idx, 0] + '.png')

        image = load_ben_color(img_name)

        image = self.transforms(image)

        label = self.df.iloc[idx, 1]

        return image, label
# Augmentations for train/test data

train_aug = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(100),

    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225]),

])

test_aug = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])

])



# train/test dataset & dataloader

train_img_dir = '../input/aptos2019-blindness-detection/train_images/'

test_img_dir =  '../input/aptos2019-blindness-detection/test_images/'



train_dataset = RetinaDataset(df=train_df, img_dir=train_img_dir, transforms=train_aug)

test_dataset = RetinaDataset(df=test_df, img_dir=test_img_dir, transforms=test_aug)



train_loader = DataLoader(dataset=train_dataset, batch_size=16,shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
i, l = next(iter(train_loader))

i.shape, l.shape
# Checking aug for tta

def show_aug(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.figure(figsize=(20,15))

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated



    

# Get a batch of training data

inputs, _ = next(iter(train_loader))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs,4)  



show_aug(out)
# Hyper parameters

num_epochs = 8

num_classes = 5

lr = 0.001
# Model initialization

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

f = '../input/inceptionv3-6epoch/inception3_6epoch.pt'



model = torchvision.models.inception_v3(pretrained=False, aux_logits = False)

model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(f, map_location='cuda:0'))

model = model.to(device)
# Loss and Optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
# Training model

total_step = len(train_loader)

for epoch in range(num_epochs):

    for batch_i, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        

        # Forward pass

        outputs = model(data)

        loss = criterion(outputs, target)

        

        # Backward and optimizer

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (batch_i+1) % 100 == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'

                 .format(epoch+1, num_epochs, batch_i+1, total_step, loss.item()))

torch.save(model.state_dict(), 'model.pt')            
# Augmentation data generators



aug1 = transforms.Compose([

       transforms.ToPILImage(),

       transforms.ToTensor(),

       transforms.Normalize(mean=[0.485, 0.456, 0.406],

                            std=[0.229, 0.224, 0.225])

])





aug2 = transforms.Compose([

        transforms.ToPILImage(),

        transforms.RandomChoice([

            transforms.RandomRotation((0,0)),

            transforms.RandomHorizontalFlip(p=1),

            transforms.RandomVerticalFlip(p=1),

            transforms.RandomRotation((90,90)),

            transforms.RandomRotation((180,180)),

            transforms.RandomRotation((270,270)),

        ]),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

])





aug3 = transforms.Compose([

        transforms.ToPILImage(),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(100),

        transforms.RandomVerticalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])

])



# Augmentation dataset & data loaders

test_dataset1 = RetinaDataset(df=test_df, img_dir=test_img_dir, transforms=aug1)

test_dataset2 = RetinaDataset(df=test_df, img_dir=test_img_dir, transforms=aug2)

test_dataset3 = RetinaDataset(df=test_df, img_dir=test_img_dir, transforms=aug3)



tl1 = DataLoader(dataset=test_dataset1, batch_size=16, shuffle=False)

tl2 = DataLoader(dataset=test_dataset2, batch_size=16, shuffle=False)

tl3 = DataLoader(dataset=test_dataset3, batch_size=16, shuffle=False)

tta_loaders = [tl1, tl2, tl3]



t1,t2,t3 = [], [], []

preds = [t1, t2, t3]

for i in range(len(tta_loaders)):

    with torch.no_grad():

        model.eval()

        for data, target in tta_loaders[i]:

            data = data.to(device)

            target = target.to(device)

            outputs = model(data)

            for probs in outputs:

                #print(prob)

                preds[i].append(probs.detach().cpu().numpy())

                

end = [(a+b+c) / 3 for a,b,c in zip(t1, t2, t3)]



predictions = []

for prob in end:   

    idx = np.argmax(prob)

    #pred = rectification(prob[idx])

    predictions.append(idx)
sub['diagnosis'] =  predictions

sub.to_csv('submission.csv', index=False)

sub['diagnosis'].value_counts()
# Extract  pretrained activations

class SaveFeatures():

    """ Extract pretrained activations"""

    features=None

    def __init__(self, m):

        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.features = ((output.cpu()).data).numpy()

    def remove(self):

        self.hook.remove()

        

final_layer = model._modules.get('Mixed_7c')

activated_features = SaveFeatures(final_layer)
## Probabilities & labels for each images

output = model(data[:8])# conver to cuda for softmax

probabilities = F.softmax(output,dim=1).data.squeeze()

pred_idx = np.argmax(probabilities.cpu().detach().numpy(),axis=1)

labels = pred_idx

activated_features.remove()

print('Probabilities classes: %s \n Prediction indices %s \n Labels: %s' % (probabilities, pred_idx, labels))
def getCAM(feature_conv, weight_fc, class_idx):

    _, nc, h, w = feature_conv.shape

    cam = weight_fc[class_idx].dot(feature_conv[0,:, :, ].reshape((nc, h*w)))

    cam = cam.reshape(h, w)

    cam = cam - np.min(cam)

    cam_img = cam / np.max(cam)

    return cam_img



weight_softmax_params = list(model._modules.get('fc').parameters())

weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

weight_softmax_params
## Current images & their heatmaps

cur_images = data.cpu().numpy().transpose((0, 2, 3, 1))

heatmaps = []

for i in pred_idx:

    img = getCAM(activated_features.features, weight_softmax, i)

    heatmaps.append(img)

    

print(cur_images.shape, len(heatmaps))
# Probability for each images

proba = []

for i in probabilities.cpu().detach().numpy():

    idx = np.argmax(i)

    proba.append((str(np.round(i[idx]*100,2)))+'%')

print(proba)
fig=plt.figure(figsize=(20,15))

for i in range(0, len(cur_images[:8])):

    img = cur_images[i]

    mask = heatmaps[i]

    ax = fig.add_subplot(4, 4,i +1,xticks=[], yticks=[])

    plt.imshow(img)

    plt.imshow(cv2.resize(mask, (512,512), interpolation=cv2.INTER_LINEAR), alpha=0.5, cmap='jet');

    ax.set_title('Label %d with %s probability' % (labels[i], proba[i]),fontsize=14)

    

#cax = fig.add_axes([0.3, 0.42, 0.4, 0.04]) # place where be map

cax = fig.add_axes([0.32, 0.42, 0.4, 0.03]) # place where be map

clb = plt.colorbar(cax=cax, orientation='horizontal',ticks=[0, 0.5, 1])

clb.ax.set_title('Level of "attention" NN in making prediction',fontsize=20)

clb.ax.set_xticklabels(['low', 'medium', 'high'],fontsize=18)





plt.show()
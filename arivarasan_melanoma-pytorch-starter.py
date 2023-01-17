# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))'''



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! pip install pretrainedmodels

! pip install wtfml
! pip install efficientnet_pytorch torchtoolbox

! curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

! python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import gc

import torch

import torchvision

import albumentations

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from tqdm import tqdm_notebook, tqdm

import pretrainedmodels

from wtfml.utils import EarlyStopping

from efficientnet_pytorch import EfficientNet

#device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#print("Imported required packages. Using device: {}".format(device))

import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp
import cv2

import matplotlib.pyplot as plt

import pandas as pd
BASE_PATH = '../input/siim-isic-melanoma-classification'

hair_images =['ISIC_0078712','ISIC_0080817','ISIC_0082348','ISIC_0109869','ISIC_0155012','ISIC_0159568','ISIC_0164145','ISIC_0194550','ISIC_0194914','ISIC_0202023']

without_hair_images = ['ISIC_0015719','ISIC_0074268','ISIC_0075914','ISIC_0084395','ISIC_0085718','ISIC_0081956']
'''fig = plt.figure(figsize=(20,30))

l = len(hair_images)

# Plot different stages of transformation

for i, image_name in enumerate(hair_images):

    image = cv2.imread(BASE_PATH+'/jpeg/train/'+image_name+'.jpg')

    resized_img = cv2.resize(image, (512,512))

    #original image

    plt.subplot(l, 5, (i*5)+1)

    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.title('Original Image')

    

    # gray image

    plt.subplot(l, 5, (i*5)+2)

    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray_image)

    plt.axis('off')

    plt.title('Gray Image')

    

    # blackhat

    kernel = cv2.getStructuringElement(1, (17,17))

    plt.subplot(l, 5, (i*5)+3)

    black_hat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    plt.imshow(black_hat)

    plt.axis('off')

    plt.title('Blackhat Image')

    

    # Intensify the hair contours

    plt.subplot(l, 5, (i*5)+4)

    retval, intense_hair = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)

    plt.imshow(intense_hair)

    plt.axis('off')

    plt.title('Intense hair Image')

    

    # Inpaint the hair region with neighbouring pixels

    plt.subplot(l, 5, (i*5)+5)

    hair_removed = cv2.inpaint(resized_img, intense_hair, 1, cv2.INPAINT_TELEA)

    plt.imshow(cv2.cvtColor(hair_removed, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.title('Hair removed image')'''
def remove_hair(image):

    #fig = plt.figure(figsize=(20,30))

    #l = len(hair_images)

    # Plot different stages of transformation

    #transformed_images = []

    #image = cv2.imread(BASE_PATH+'/jpeg/train/'+image+'.jpg')

    #resized_img = cv2.resize(image, (128,128))



    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    kernel = cv2.getStructuringElement(1, (17,17))

    black_hat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)



    # Intensify the hair contours

    retval, intense_hair = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)



    # Inpaint the hair region with neighbouring pixels

    hair_removed = cv2.inpaint(image, intense_hair, 1, cv2.INPAINT_TELEA)

    #transformed_images.append(hair_removed)

    return hair_removed
'''%%time

hair_image = 'ISIC_0078712'

hair_removed_images = remove_hair(hair_image)'''
print(hair_removed_images[0].shape)
# To use TPU

try:

    import torch_xla.core.xla_model as xm 

    import torch_xla.distributed.parallel_loader as pl

    _xla_available = True

except ImportError:

    _xla_available = False

print('TPU available: ',_xla_available)

#print("Imported required packages. Using device: {}".format(device))
'''def reduce_fn(vals):

    return sum(vals) / len(vals)'''
warnings.simplefilter('ignore')

torch.manual_seed(42)

np.random.seed(42)
BASE_DIR = "../input/siim-isic-melanoma-classification/"
!ls -lrt "../input/siimisic-melanoma-resized-images"
npy_data = np.load("../input/siimisic-melanoma-resized-images/x_train_96.npy")
class MelanomaDataLoader(Dataset):

    '''Dataloader class'''

    def __init__(self, npy_data, targets, augmentations=None):

        self.npy_data = npy_data

        self.targets = targets

        self.augmentations = augmentations

        

    def __len__(self):

        return len(self.npy_data)

    

    def __getitem__(self, idx):

        

        np_img = self.npy_data[idx]

        np_img = remove_hair(np_img)

        target = self.targets[idx]

        if self.augmentations:

            augmented = self.augmentations(image=np_img)

            image_data = augmented['image']

        else:

            image_data = torch.from_numpy(np_img)

        image_data = np.transpose(image_data, (2,0,1)).astype(np.float32)

        return {

            'images': torch.tensor(image_data, dtype=torch.float),

            'targets': torch.tensor(target, dtype=torch.long)

        }
class SEResnext50_32x4d(nn.Module):

    '''This is network class'''

    def __init__(self, pretrained='imagenet', wp = None):

        super(SEResnext50_32x4d, self).__init__()

        

        self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)

        #print(self.base_model)

        if pretrained is not None:

            self.base_model.load_state_dict(

            torch.load('../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth')

            )

        '''for params in self.base_model.parameters():

            params.requires_grad = False'''

            

        self.l0 = nn.Linear(2048, 1)

        if wp is not None:

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=wp)

        else:

            self.criterion = nn.BCEWithLogitsLoss()

        

    def forward(self, images, targets):

        batch_size = images.shape[0]

        

        x = self.base_model.features(images)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        yhat = self.l0(x)

        #loss = nn.BCEWithLogitsLoss(pos_weight=wp)(yhat, targets.view(-1, 1).type_as(x))

        loss = self.criterion(yhat, targets.view(-1, 1).type_as(x))

        return yhat, loss
class EfNet(nn.Module):

    '''This is network class'''

    def __init__(self, pretrained='imagenet', wp = None):

        super(EfNet, self).__init__()

        

        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        self.base_model._fc = nn.Linear(1280, 1, bias=True)

        

        '''self.meta = nn.Sequential(

                        nn.BatchNorm1d(500),

                        nn.ReLU(),

                        nn.Dropout(0.4),

                        nn.Linear(500,100, bias=True),

                        nn.BatchNorm1d(100),

                        nn.ReLU(),

                        nn.Dropout(0.4),

                        nn.Linear(100,1, bias=True))'''

        if wp is not None:

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=wp)

        else:

            self.criterion = nn.BCEWithLogitsLoss()

        

    def forward(self, images, targets):

        batch_size = images.shape[0]

        

        yhat = self.base_model(images)

        #x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        #yhat = self.l0(x)

        #loss = nn.BCEWithLogitsLoss(pos_weight=wp)(yhat, targets.view(-1, 1).type_as(x))

        #yhat = self.meta(x)

        loss = self.criterion(yhat, targets.view(-1, 1).type_as(yhat))

        return yhat, loss
efnet = EfNet(pretrained='Imagenet',wp=torch.tensor(0))

#efnet._fc = nn.Linear(1280, 1)

print(efnet)
for param in efnet.parameters():

    if param.requires_grad:

        print(param.shape)
!ls -lrt ../input/siim-isic-melanoma-classification/
# create folds

df = pd.read_csv(BASE_DIR+'train.csv')

df['fold'] = -1

#df = df.sample(frac=1).reset_index(drop=True)

y = df.target.values



kfolds = StratifiedKFold(n_splits=5)



for fold, (t_, v_) in enumerate(kfolds.split(X=df, y=y)):

    df.loc[v_, 'fold'] = fold

    

df.to_csv('./train_new.csv', index=False)

print(df.head())
train_new = pd.read_csv('./train_new.csv')

print(train_new.head())

train_new.fold.value_counts()
fold = 2

train_indices = train_new[train_new.fold != fold].index.to_numpy()

val_indices = train_new[train_new.fold == fold].index.to_numpy()

print(len(train_indices), len(val_indices))
print(train_indices[:5])
print(len(train_indices)+len(val_indices))

print(len(npy_data))
print(set(train_indices).intersection(val_indices))
fold=1

train_npy = npy_data[train_indices]

val_npy = npy_data[val_indices]

train_targets = train_new[train_new.fold != fold]['target'].to_numpy()

val_targets = train_new[train_new.fold == fold]['target'].to_numpy()

print(sum(train_targets==0))

print(sum(train_targets==1))

print(len(train_targets))
#fold=0

train_unique, train_counts = np.unique(train_targets, return_counts=True)

val_unique, val_counts = np.unique(val_targets, return_counts=True)

print(f"Train counts: {train_unique} {train_counts} ********* Val counts: {val_unique} {val_counts}")
#fold=1

train_unique, train_counts = np.unique(train_targets, return_counts=True)

val_unique, val_counts = np.unique(val_targets, return_counts=True)

print(f"Train counts: {train_unique} {train_counts} ********* Val counts: {val_unique} {val_counts}")
print(f"There are {len(train_npy)} train data and {len(train_targets)} train targets. val count: {len(val_npy)}")
#mean = (0.485, 0.456, 0.406)

#std = (0.229, 0.224, 0.225)

mean = (0.5,0.5,0.5)

std = (0.5,0.5,0.5)

train_aug = albumentations.Compose([

    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),

    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),

    albumentations.Flip(p=0.5)

])

valid_aug = albumentations.Compose([

    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)

])
train_data = MelanomaDataLoader(train_npy, train_targets, augmentations=train_aug)

val_data = MelanomaDataLoader(val_npy, val_targets, augmentations=valid_aug)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

val_loader = DataLoader(val_data, batch_size=4, shuffle=True)
print(len(train_loader.dataset))
train_batch = next(iter(train_loader))

print(train_batch['images'].shape, train_batch['targets'])
def imshow(img, title):

    plt.figure(figsize=(10,5))

    np_img = img.numpy() / 2 + 0.5

    plt.axis('off')

    plt.imshow(np.transpose(np_img, (1,2,0)))

    plt.title(title)

    plt.show()
def show_image_batches(data_loader):

    batch = next(iter(data_loader))

    imgs, labels = batch['images'], batch['targets']

    print("img shape: ",batch['images'].shape)

    imgs = torchvision.utils.make_grid(imgs)

    title = labels.numpy().tolist()

    imshow(imgs, title)
show_image_batches(train_loader)
show_image_batches(val_loader)
BS = 16

train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)

val_loader = DataLoader(val_data, batch_size=BS, shuffle=True)
%%time

batch = next(iter(train_loader))

print(batch['images'].shape)
def train(fold, use_tpu=False, net='se_resnext'):

    epochs = 25

    BS = 64

    lr = 0.0001

    device = xm.xla_device()

    if use_tpu:

        device = xm.xla_device()

    #else:

    #    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Dataloader prep steps

    train_indices = train_new[train_new.fold != fold].index.to_numpy()

    val_indices = train_new[train_new.fold == fold].index.to_numpy()

    train_npy = npy_data[train_indices]

    val_npy = npy_data[val_indices]

    train_targets = train_new[train_new.fold != fold]['target'].to_numpy()

    val_targets = train_new[train_new.fold == fold]['target'].to_numpy()

    

    #let's check target distribution in this fold

    train_unique, train_counts = np.unique(train_targets, return_counts=True)

    val_unique, val_counts = np.unique(val_targets, return_counts=True)

    print(f"Train counts: {train_unique} {train_counts} ********* Val counts: {val_unique} {val_counts}")

    

    mean = (0.485, 0.456, 0.406)

    std = (0.229, 0.224, 0.225)

    train_aug = albumentations.Compose([

        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),

        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),

        albumentations.Flip(p=0.5)

    ])

    valid_aug = albumentations.Compose([

        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)

    ])

    train_data = MelanomaDataLoader(train_npy, train_targets, augmentations=train_aug)

    val_data = MelanomaDataLoader(val_npy, val_targets, augmentations=valid_aug)

    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)

    val_loader = DataLoader(val_data, batch_size=BS, shuffle=False)

    

    wp = sum(train_targets==0) / sum(train_targets)

    fold_wp = torch.tensor(wp, dtype=torch.float)

    #modelling

    

    if 'ef' in net:

        model = EfNet(pretrained='imagenet', wp=fold_wp)

    else:

        model = SEResnext50_32x4d(pretrained='imagenet', wp=fold_wp)

    

    model.to(device)

    

    '''for param in model.parameters():

        if param.requires_grad:

            print(param.shape)'''

    

    optimizr = torch.optim.Adam(model.parameters(), lr=lr)

    schedulr = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizr, patience=3, threshold=0.001, mode="max"

        )

    es = EarlyStopping(patience=5, mode='max')

    best_auc = 0

    losses = []

    n_iter = len(train_indices) // BS

    

    for epoch in range(epochs):

        model.train()

        if use_tpu:

            pl_loader = pl.ParallelLoader(train_loader, [device])

            tk0 = tqdm(

                pl_loader.per_device_loader(device),

                total=len(train_loader))

        else:

            tk0 = tqdm(train_loader, total=len(train_loader))

                    

        for i, data in enumerate(tk0, 1):

            images, targets = data['images'], data['targets']

            images, targets = images.to(device), targets.to(device)

            

            optimizr.zero_grad()

            #batch_wp = sum(targets==0) / sum(targets)

            out, loss = model(images, targets)

            

            loss.backward()

            

            if use_tpu:

                xm.optimizer_step(optimizr)

            else:

                optimizr.step()

            

            optimizr.zero_grad()

            #train_unique, train_counts = np.unique(targets.cpu().numpy(), return_counts=True)

            #print(f"Train counts: {train_unique} {train_counts}")

            #print("Loss for batch: {} is {}".format(i+1, loss.item()))

            

            torch.cuda.empty_cache()

            

            #if i%50 == 0:

            #print(f"Batch {i} contains {sum(targets)} positive labels")

            #print("Evaluating model...")

            #print("Epoch: %d ******* Iter: %d/%d ******* Loss: %0.2f VAL_AUC: %0.2f"%(epoch, i, n_iter, loss.item(), val_auc))

            '''if val_auc > best_auc:

                print("Max AUC attained, saving model..")

                torch.save(model.state_dict(), './siimModel_{}.pth'.format(fold))

                best_auc = val_auc'''

            

            del images, targets

            

        

        val_auc = evaluate(val_loader, val_targets, model, device, use_tpu)

        print("Epoch: %d ******* VAL_AUC: %0.2f"%(epoch, val_auc))

        schedulr.step(val_auc)

        es(val_auc, model, model_path=f"./melanoma_fold_{fold}.bin")

        

        '''if val_auc > best_auc:

            print("Max AUC attained, saving model..")

            torch.save(model.state_dict(), './siimModel_{}.pth'.format(fold))

            best_auc = val_auc'''

            

        if es.early_stop:

            print("Early Stopping..")

            break

        gc.collect()
def evaluate(data_loader, val_targets, model, device, use_tpu=False):

    model = model.to(device)

    model.eval()

    final_preds = []

    with torch.no_grad():

        if use_tpu:

            pl_loader = pl.ParallelLoader(data_loader, [device])

            tk0 = tqdm(pl_loader.per_device_loader(device), total = len(data_loader))

        else:

            tk0 = tqdm(data_loader, total=len(data_loader))

        for i, data in enumerate(tk0):

            images, targets = data['images'], data['targets']

            images, targets = images.to(device), targets.to(device)

            batch_wp = sum(targets==0) / sum(targets)

            

            preds, _ = model(images, targets)

            final_preds.append(preds.cpu())

    predictions = np.vstack((final_preds)).ravel()

    print('val_targets: ',val_targets[:5])

    print('predictions: ',predictions[:5])

    

    auc = roc_auc_score(val_targets, predictions)

    return auc
'''def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = train(fold=0, use_tpu=True)'''
'''FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')'''
train(0, use_tpu=True, net='efnet')
train(1, use_tpu=True, net='efnet')

train(2, use_tpu=True, net='efnet')

train(3, use_tpu=True, net='efnet')

train(4, use_tpu=True, net='efnet')
!ls -lrt ../input/melanoma-pytorch-starter/
npy_test = np.load("../input/siimisic-melanoma-resized-images/x_test_64.npy")

print(f"There are {len(npy_test)} images in test set")

print(npy_test.shape)
def predict(fold):

    BS = 4

    mean = (0.485, 0.456, 0.406)

    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose([

        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)

    ])

    #just for the sake

    test_targets = np.zeros(len(npy_test))

    test_wp = torch.tensor(1, dtype=torch.float)

    

    test_data = MelanomaDataLoader(npy_test, test_targets, augmentations=test_aug)

    test_loader = DataLoader(test_data, batch_size=BS, shuffle=False)

    

    if 'ef' in net:

        model = EfNet(pretrained=None, wp=fold_wp)

    else:

        model = SEResnext50_32x4d(pretrained=None, wp=fold_wp)

    

    #model = SEResnext50_32x4d(pretrained=None, wp=test_wp)

    print(f"Loading from model: melanoma_fold_{fold}.bin")

    model.load_state_dict(torch.load(f"../input/melanoma-pytorch-starter/melanoma_fold_{fold}.bin"))

    model = model.to(device)

    model.eval()

    

    test_preds = []

    #tk1 = tqdm(test_loader, total = len(test_loader))

    with torch.no_grad():

        for batch, data in enumerate(test_loader, 1):

            torch.cuda.empty_cache()

            images, targets = data['images'], data['targets']

            images, targets = images.to(device), targets.to(device)

            out, _ = model(images, targets)

            #test_preds.append(out)

            test_preds.append(out.cpu())

            del images, targets

    predictions = np.vstack(test_preds).ravel()

    return predictions        
p1 = predict(1)
print(type(p1))
p0 = predict(0)

p2 = predict(2)

p3 = predict(3)

p4 = predict(4)
#submission

predictions = (p0 + p1 + p2 + p3 + p4) / 5

submission_df = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

submission_df.loc[:, 'target'] = predictions

print(submission_df.head())

submission_df.to_csv('submission_file.csv', index=False)
#Check roc_auc

targets = np.zeros(10)

targets[8] = 1

print(targets)

preds = (np.random.rand(10)*0.1).ravel()

preds[8] = -0.1

print(preds)

auc = roc_auc_score(targets, preds)

print(auc)
# check loss

out = np.array([-2.9445, -3.8510, -8.5114, 3.1692, 1.6949, -5.5680, -9.3456, -6.9603, -5.7006, -9.9718])

#out = (np.random.rand(10)*-10)

out = torch.from_numpy(out)

#out = out.view(-1,1)

print((out))

targets = torch.zeros(10)

targets[9] = 1

#targets = targets.view(-1,1)

print((targets))

print(targets.shape, out.shape)

wp = torch.tensor(9/1, dtype=torch.float)

loss = nn.BCEWithLogitsLoss(pos_weight=wp)(out, targets)

print(loss)
# check loss

out = np.array([-2.9445, -3.8510, -8.5114, -3.1692, -1.6949, -5.5680, -9.3456, -6.9603, -5.7006, -5.9718])

#out = (np.random.rand(10)*-10)

out = torch.from_numpy(out)

#out = out.view(-1,1)

print((out))

targets = torch.zeros(10)

targets[9] = 1

#targets = targets.view(-1,1)

print((targets))

print(targets.shape, out.shape)

loss = nn.BCEWithLogitsLoss()(out, targets)

print(loss)
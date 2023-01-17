!pip install efficientnet_pytorch torchtoolbox
import os

import gc

import math

import time

import random

import datetime

import warnings

import cv2

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchtoolbox.transform as transforms

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, roc_auc_score



import matplotlib.pyplot as plt

import seaborn as sns



from efficientnet_pytorch import EfficientNet

%matplotlib inline
warnings.simplefilter('ignore')

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(47)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
basepath = "../input/jpeg-melanoma-256x256/"

train_csv = basepath + "train.csv"

test_csv = basepath + "test.csv"

train_img_path = '../input/jpeg-melanoma-256x256/train/'

test_img_path = '../input/jpeg-melanoma-256x256/test/'



print(train_csv)
!ls ../input/jpeg-melanoma-256x256
train_df = pd.read_csv(train_csv)

test_df = pd.read_csv(test_csv)



display(train_df.head())

print("Train shape: ", train_df.shape)

print("Test shape: ",test_df.shape)
location = 'anatom_site_general_challenge'

concat = pd.concat([train_df[location], test_df[location]], ignore_index=True)

dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')

train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]].reset_index(drop=True)], axis=1)

test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)



train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})

train_df['sex'] = train_df['sex'].fillna(-1)

test_df['sex'] = test_df['sex'].map({'male': 1, 'female': 0})

test_df['sex'] = test_df['sex'].fillna(-1)



train_df['age_approx'] /= train_df['age_approx'].max()

train_df['age_approx'] = train_df['age_approx'].fillna(0)

test_df['age_approx'] /= test_df['age_approx'].max()

test_df['age_approx'] = test_df['age_approx'].fillna(0)



print("Train shape: ", train_df.shape)

print("Test shape: ", test_df.shape)
meta_features = ['sex', 'age_approx'] + [col for col in train_df.columns if 'site_' in col]

meta_features.remove('anatom_site_general_challenge')

meta_features
class SIIMDataset(Dataset):

    def __init__(self, df: pd.DataFrame, imgfolder: str, train: bool = True, transforms = None, meta_features = None):

        self.df = df

        self.imgfolder = imgfolder

        self.transforms = transforms

        self.train = train

        self.meta_features = meta_features

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        im_path = os.path.join(self.imgfolder, self.df.iloc[idx]['image_name'] + '.jpg')

        img = cv2.imread(im_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        meta = np.array(self.df.iloc[idx][self.meta_features].values, dtype=np.float32)

        meta = torch.from_numpy(meta)

        

        if self.transforms:

            img = self.transforms(img)

        

        if self.train:

            target = torch.tensor(self.df.loc[idx, 'target'], dtype=torch.float)

            return (img, meta), target

        else:

            return (img, meta)
ones = len(train_df.query('target == 1'))

zeros = len(train_df.query('target == 0'))



weightage_fn = {0: 1./zeros, 1: 1./ones}

print(weightage_fn)
def get_sampler(df, idx):

    targets = df['target'][idx].values

    weights = [weightage_fn[x] for x in targets]

    sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    return sampler
class AdvancedHairAugmentation:

    """

    Impose an image of a hair to the target image



    Args:

        hairs (int): maximum number of hairs to impose

        hairs_folder (str): path to the folder with hairs images

    """



    def __init__(self, hairs: int = 10, hairs_folder: str = ""):

        self.hairs = hairs

        self.hairs_folder = hairs_folder



    def __call__(self, img):

        """

        Args:

            img (PIL Image): Image to draw hairs on.



        Returns:

            PIL Image: Image with drawn hairs.

        """

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



            # Creating a mask and inverse mask

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)

            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

            mask_inv = cv2.bitwise_not(mask)



            # Now black-out the area of hair in ROI

            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)



            # Take only region of hair from hair image.

            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)



            # Put hair in ROI and modify the target image

            dst = cv2.add(img_bg, hair_fg)



            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

                

        return img



    def __repr__(self):

        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'
# Transformations



train_transforms = transforms.Compose([

    #transforms.ToPILImage(),

    AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs'),

    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ColorJitter(brightness=24. / 255.,saturation=0.3),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

])



test_transforms = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

])
def show_images(img):

    plt.figure(figsize=(18,15))

    img = img.numpy()

    plt.imshow(np.transpose(img, (1,2,0)))

    plt.show()
dataset = SIIMDataset(df=train_df, imgfolder=train_img_path, train=True, transforms=train_transforms, meta_features=meta_features)

loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

data = iter(loader)

images = data.next()

show_images(torchvision.utils.make_grid(images[0][0]))

del dataset, loader
class SIIMNet(nn.Module):

    def __init__(self, base1, base2, n_meta_features: int):

        super(SIIMNet, self).__init__()

        self.base1 = base1

        self.base1._fc = nn.Linear(in_features=1536, out_features=500, bias=True)

        self.base2 = base2

        self.base2._fc = nn.Linear(in_features=1280, out_features=500, bias=True)

        self.fc1 = nn.Linear(1000, 500)

        self.bn1 = nn.BatchNorm1d(500)

        self.meta_net = nn.Sequential(nn.Linear(n_meta_features, 500),

                                  nn.BatchNorm1d(500),

                                  nn.ReLU(),

                                  nn.Dropout(p=0.2),

                                  nn.Linear(500, 250),

                                  nn.BatchNorm1d(250),

                                  nn.ReLU(),

                                  nn.Dropout(p=0.2))

        self.out = nn.Linear(500+250, 1)

        

    def forward(self, inputs):

        img, meta = inputs

        cnn1_ = self.base1(img)

        cnn2_ = self.base2(img)

        cnn_ = torch.cat((cnn1_, cnn2_), 1)

        cnn_ = self.fc1(cnn_)

        cnn_ = self.bn1(cnn_)

        meta_ = self.meta_net(meta)

        features = torch.cat((cnn_, meta_), dim=1)

        output = self.out(features)

        return output
# dataset = SIIMDataset(df=train_df, imgfolder=train_img_path, train=True, transforms=train_transforms, meta_features=meta_features)

# loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

# data = iter(loader)

# images = data.next()



# base1 = EfficientNet.from_pretrained('efficientnet-b3')

# base2 = EfficientNet.from_pretrained('efficientnet-b1')

# model = SIIMNet(base1=base1, base2=base2, n_meta_features=len(meta_features))



# model(images[0])
# CONFIG 

epochs = 10

model_path = 'model.pth'

es_patience = 3

TTA = 3 # Test Time Augmentation



skf = StratifiedKFold(n_splits=5)
# base = EfficientNet.from_pretrained('efficientnet-b1')
# features = base.extract_features(images[0][0])

# features.shape
def train_fn(model, train_loader, opt, criterion):

    batch = 1

    epoch_loss = 0

    correct = 0

    for x, y in train_loader:

        if(batch % 40 == 0):

            print("=", end="")

        x[0] = x[0].to(device)

        x[1] = x[1].to(device)

        y = y.to(device)



        opt.zero_grad()

        z = model(x)



        loss = criterion(z, y.unsqueeze(1))

        loss.backward()

        opt.step()



        pred = torch.round(torch.sigmoid(z))

        correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()

        epoch_loss += loss.item()



        batch += 1

    return correct, epoch_loss
def validate_fn(model, val_loader, val_idx):

    batch = 1

    val_preds = torch.zeros((len(val_idx), 1), device=device, dtype=torch.float32)

    with torch.no_grad():

        for j, (x_val, y_val) in enumerate(val_loader):

            if(batch % 20 == 0):

                print("=", end="")

            x_val[0] = x_val[0].to(device)

            x_val[1] = x_val[1].to(device)

            y_val = y_val.to(device)



            z_val = model(x_val)

            val_pred = torch.sigmoid(z_val)



            val_preds[j*val_loader.batch_size : j*val_loader.batch_size + x_val[0].shape[0]] = val_pred



            batch += 1



        val_acc = accuracy_score(train_df.iloc[val_idx]['target'].values, torch.round(val_preds.cpu()))

        val_roc = roc_auc_score(train_df.iloc[val_idx]['target'].values, val_preds.cpu())

    return val_acc, val_roc
def engine():

    

    oof = np.zeros((len(train_df), 1))

    preds = torch.zeros((len(test_df), 1), dtype=torch.float32, device=device)

    

    test_ds = SIIMDataset(

        df = test_df,

        imgfolder = test_img_path,

        train = False,

        transforms = train_transforms,

        meta_features = meta_features

    )

    

    for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), start=1):

        print("\n")

        print("="*15, "FOLD ", fold, "="*15)

        

        best_val = None

        patience = es_patience



        base1 = EfficientNet.from_pretrained('efficientnet-b3')

        base2 = EfficientNet.from_pretrained('efficientnet-b1')

        model = SIIMNet(base1=base1, base2=base2, n_meta_features=len(meta_features))

        model.to(device)



        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = ReduceLROnPlateau(optimizer=opt, mode='max', patience=1, factor=0.2, verbose=True)

        criterion = nn.BCEWithLogitsLoss()



        

        train_ds = SIIMDataset(

            df = train_df.iloc[train_idx].reset_index(drop=True),

            imgfolder = train_img_path,

            train = True,

            transforms = train_transforms,

            meta_features = meta_features

        )

        sampler = get_sampler(train_df, train_idx)

        train_loader = DataLoader(dataset=train_ds, batch_size=32, sampler=sampler, num_workers=2)



        val_ds = SIIMDataset(

            df = train_df.iloc[val_idx].reset_index(drop=True),

            imgfolder = train_img_path,

            train = True,

            transforms = test_transforms,

            meta_features = meta_features

        )

        val_loader = DataLoader(dataset=val_ds, batch_size=16, shuffle=False, num_workers=2)



        test_loader = DataLoader(dataset=test_ds, batch_size=16, shuffle=False, num_workers=2)

        

        for epoch in range(epochs):

            start_time = time.time()



            print("Training:")

            model.train()

            correct, epoch_loss = train_fn(model, train_loader, opt, criterion)

            train_acc = correct / len(train_idx)



            print("\nValidating:")

            model.eval()            

            val_acc, val_roc = validate_fn(model, val_loader, val_idx)

            scheduler.step(val_roc)

        

            print('\nEpoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(

            epoch + 1, 

            epoch_loss, 

            train_acc, 

            val_acc, 

            val_roc, 

            str(datetime.timedelta(seconds=time.time() - start_time))[:7]))



            if not best_val:

                best_val = val_roc

                torch.save(model, model_path)

                continue



            if val_roc >= best_val:

                best_val = val_roc

                patience = es_patience

                torch.save(model, model_path)

            else:

                patience -= 1

                if patience == 0:

                    print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))

                    break



        model = torch.load(model_path)

        val_preds = torch.zeros((len(val_idx), 1), device=device, dtype=torch.float32)



        batch = 1



        print("Validating model for FOLD {}:".format(fold))

        model.eval()

        with torch.no_grad():

            for j, (x_val, y_val) in enumerate(val_loader):

                if(batch % 20 == 0):

                    print("=", end="")

                x_val[0] = x_val[0].to(device)

                x_val[1] = x_val[1].to(device)

                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)



                z_val = model(x_val)

                val_pred = torch.sigmoid(z_val)



                val_preds[j*val_loader.batch_size : j*val_loader.batch_size + x_val[0].shape[0]] = val_pred



                batch += 1



            oof[val_idx] = val_preds.cpu().numpy()





            batch = 1



            print("\nTesting:")

            for _ in range(TTA):

                for i, x_test in enumerate(test_loader):

                    if(batch % 60 == 0):

                        print("=", end="")

                    x_test[0] = x_test[0].to(device)

                    x_test[1] = x_test[1].to(device)



                    z_test = model(x_test)

                    z_test = torch.sigmoid(z_test)



                    preds[i*test_loader.batch_size : i*test_loader.batch_size + x_test[0].shape[0]] += z_test



                    batch += 1

            preds /= TTA



        del train_ds, val_ds, train_loader, val_loader, x_val, y_val

        gc.collect()



    preds /= skf.n_splits

    return preds, oof
preds, oof = engine()
print('OOF: {:.3f}'.format(roc_auc_score(train_df['target'], oof)))
pd.Series(oof.reshape(-1,)).to_csv('oof.csv', index=False)
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

sub['target'] = preds.cpu().numpy().reshape(-1,)

sub.to_csv('submission-v16.csv', index=False)
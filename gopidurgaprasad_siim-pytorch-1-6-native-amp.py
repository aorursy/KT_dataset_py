!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html > /dev/null

!pip install -U git+https://github.com/albumentations-team/albumentations > /dev/null

!pip install timm > /dev/null

!pip install catalyst==20.4.2 > /dev/null
import warnings

warnings.filterwarnings('ignore')



import os

import cv2

import sklearn

import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F

import albumentations as A





from timm.models.efficientnet import tf_efficientnet_b5_ns

from albumentations.pytorch.functional import img_to_tensor

from sklearn.model_selection import KFold

from torch.utils.data import Dataset

from torch.nn.modules.dropout import Dropout

from torch.nn.modules.linear import Linear

from torch.nn.modules.pooling import AdaptiveAvgPool2d

from catalyst.data.sampler import DistributedSampler, BalanceClassSampler

from functools import partial

from tqdm import tqdm
class args:

    

    exp_name = "E5_512_300"

    output_dir = "outputs"

    

    folds = 2

    

    train_image_path = "../input/jpeg-melanoma-512x512/train"

    test_image_path = "../input/jpeg-melanoma-512x512/test"

    

    network = "MelanomaClassifier"

    encoder = "tf_efficientnet_b5_ns"

    

    train_csv = "../input/jpeg-melanoma-512x512/train.csv"

    test_csv = "../input/jpeg-melanoma-512x512/test.csv"

    

    label_smoothing = 0.01

    epochs = 50

    size = 300

    batch_size = 30

    learning_rate = 0.00002

    

    

    normalize = {

        "mean": [0.485, 0.456, 0.406],

        "std": [0.229, 0.224, 0.225]

    }

    

    # CUSTOM LEARNING SCHEUDLE

    LR_START = 0.00001

    LR_MAX = 0.00005

    LR_MIN = 0.00001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    
encoder_params = {

    "tf_efficientnet_b5_ns": {

        "features": 2048,

        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)

    }

}





class MelanomaClassifier(nn.Module):

    def __init__(self, encoder, dropout_rate=0.0) -> None:

        super().__init__()

        self.encoder = encoder_params[encoder]["init_op"]()

        self.avg_pool = AdaptiveAvgPool2d((1, 1))

        self.dropout = Dropout(dropout_rate)

        self.fc = Linear(encoder_params[encoder]["features"], 1)



    def forward(self, x):

        x = self.encoder.forward_features(x)

        x = self.avg_pool(x).flatten(1)

        x = self.dropout(x)

        x = self.fc(x)

        return x

class MelanomaClassifierDataset(Dataset):

    def __init__(

        self,

        df,

        label_smoothing,

        normalize,

        mode="train",

        transforms=None,

        data_root=None

    ):

        

        super().__init__()

        self.df = df

        self.mode = mode

        self.label_smoothing = label_smoothing

        self.normalize = normalize

        self.transforms = transforms

        self.data_root = data_root

        

        self.image_name = self.df["image_name"].values

        self.label = self.df["target"].values

        

    def __getitem__(self, index: int):



        image_name, label = self.image_name[index], self.label[index]

        if self.mode == "train":

            label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)

        image = cv2.imread(f"{self.data_root}/{image_name}.jpg", cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        if self.transforms:

            image = self.transforms(image=image)["image"]



        image = img_to_tensor(image, self.normalize)



        return {

            "image_name": image_name,

            "image": image,

            "label": label

        }

    

    def __len__(self):

        return len(self.image_name)

    

    def __get_labels__(self):

        return self.label.tolist()





class MelanomaClassifierDatasetTest(Dataset):

    def __init__(

        self,

        df,

        normalize={"mean": [0.485, 0.456, 0.406],

                    "std": [0.229, 0.224, 0.225]},

        transforms=None,

        data_root=None

    ):

        super().__init__()

        self.df = df

        self.normalize = normalize

        self.transforms = transforms

        self.data_root = data_root



        self.image_name = self.df["image_name"]

    

    def __getitem__(self, index: int):

        

        image = cv2.imread(f"{self.data_root}/{self.image_name[index]}.jpg", cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        if self.transforms:

            image = self.transforms(image=image)["image"]



        image = img_to_tensor(image, self.normalize)



        return {

            "image_name": self.image_name[index],

            "image": image

        }

    

    def __len__(self):

        return len(self.image_name)
class AverageMeter(object):

    """Computes and stores the average and current value"""



    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count



class RocAucMeter(object):

    def __init__(self):

        self.reset()

    

    def reset(self):

        self.y_true = [0,1]

        self.y_pred = [0.5, 0.5]

        self.score = 0

    

    def update(self, y_true, y_pred):

        self.y_true.extend(y_true.cpu().detach().numpy().round().tolist())

        self.y_pred.extend(y_pred.cpu().detach().numpy().reshape(-1).tolist())

        y_pred1 = np.array(self.y_pred)

        y_true1 = np.array(self.y_true)

        y_pred1[np.isnan(y_pred1)]=0.5

        self.score = sklearn.metrics.roc_auc_score(y_true1, y_pred1)

    @property

    def avg(self):

        return self.score
def train_transforms(size=300):

    return A.Compose([

        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),

        A.GaussianBlur(blur_limit=3 , p=0.05),

        A.GaussNoise(p=0.1),

        A.Resize(size, size),

        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),

        A.OneOf([

            A.VerticalFlip(),

            A.HorizontalFlip(),

            A.Flip()

        ], p=0.5),

        A.Transpose(p=0.33),

        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),

        A.RandomRotate90(p=0.11),

        A.ElasticTransform(alpha_affine=60, p=0.33),

        A.Cutout(num_holes=8, max_h_size=size//8, max_w_size=size//8, fill_value=0, p=0.3),

        A.Rotate(limit=80)

    ])



def valid_transforms(size=300):

    return A.Compose([

        A.Resize(size, size),

        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),

    ])



def test_transforms(size=300):

    return A.Compose([

        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),

        A.GaussianBlur(blur_limit=3 , p=0.05),

        A.GaussNoise(p=0.1),

        A.Resize(size, size),

        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),

        A.OneOf([

            A.VerticalFlip(),

            A.HorizontalFlip(),

            A.Flip()

        ], p=0.5),

        A.Transpose(p=0.33),

        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),

        A.RandomRotate90(p=0.11),

        A.ElasticTransform(alpha_affine=60, p=0.33),

        A.Rotate(limit=80)

    ])
def loss_fn(output, target):

    return nn.BCEWithLogitsLoss()(output, target.view(-1, 1))



def lrfn(epoch):

    if epoch < args.LR_RAMPUP_EPOCHS:

        lr = (args.LR_MAX - args.LR_START) / args.LR_RAMPUP_EPOCHS * epoch + args.LR_START

    elif epoch < args.LR_RAMPUP_EPOCHS + args.LR_SUSTAIN_EPOCHS:

        lr = args.LR_MAX

    else:

        lr = (args.LR_MAX - args.LR_MIN) * args.LR_EXP_DECAY**(epoch - args.LR_RAMPUP_EPOCHS - args.LR_SUSTAIN_EPOCHS) + args.LR_MIN

    return lr



def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = lrfn(epoch)

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr 
def train_epoch(args, model, train_loader, optimizer, scheduler, device, epoch):

    losses = AverageMeter()

    scores = RocAucMeter()



    model.train()

    scaler = torch.cuda.amp.GradScaler()



    t = tqdm(train_loader)

    for i, sample in enumerate(t):

        imgs = sample["image"].to(device)

        labels = sample["label"].to(device)



        optimizer.zero_grad()



        # Casts operations to mixed precision

        with torch.cuda.amp.autocast():

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)



        bs = imgs.size(0)

        scores.update(labels, torch.sigmoid(outputs))

        losses.update(loss.item(), bs)



        # Scales the loss, and calls backward()

        # to create scaled gradients

        scaler.scale(loss).backward()



        # Uncales gradients and calls

        # or skips optimizer.step()

        scaler.step(optimizer)



        # Updates the scale for next iteration

        scaler.update()



        t.set_description(f"Train E:{epoch} - Loss:{losses.avg:0.4f} - AUC:{scores.avg:0.4f} ")



    return scores.avg, losses.avg



def valid_epoch(args, model, valid_loader, device, epoch):

    losses = AverageMeter()

    scores = RocAucMeter()



    model.eval()

    with torch.no_grad():

        t = tqdm(valid_loader)

        for i, sample in enumerate(t):

            imgs = sample["image"].to(device)

            labels = sample["label"].to(device)



            outputs = model(imgs)



            bs = imgs.size(0)

            scores.update(labels, torch.sigmoid(outputs))



            t.set_description(f"Valid E:{epoch} - AUC:{scores.avg:0.4f} ")



    return scores.avg



def test_epoch(args, model, test_loader, device):



    probs = []



    model.eval()



    with torch.no_grad():

        t = tqdm(test_loader)

        for i, sample in enumerate(t):

            imgs = sample["image"].to(device)

            img_names = sample["image_name"]



            out = model(imgs)

            preds = torch.sigmoid(out).cpu().numpy().tolist()



            probs.extend(preds)

    

    return probs
def main(fold, idxT, idxV):

    

    device = "cuda"

    model = MelanomaClassifier(args.encoder)

    model = model.cuda()

    

    args.save_path = os.path.join(args.output_dir, args.exp_name)

    os.makedirs(args.save_path, exist_ok=True)

    

    train_df = pd.read_csv(args.train_csv)

    test_df = pd.read_csv(args.test_csv)

    

    train_folds = train_df[train_df.tfrecord.isin(idxT)]

    valid_folds = train_df[train_df.tfrecord.isin(idxV)]

    

    train_dataset = MelanomaClassifierDataset(

        df=train_folds,

        mode="train",

        label_smoothing=args.label_smoothing,

        normalize=args.normalize,

        transforms=train_transforms(size=args.size),

        data_root=args.train_image_path

    )



    valid_dataset = MelanomaClassifierDataset(

        df=valid_folds,

        mode="valid",

        label_smoothing=args.label_smoothing,

        normalize=args.normalize,

        transforms=valid_transforms(size=args.size),

        data_root=args.train_image_path

    )



    test_dataset = MelanomaClassifierDatasetTest(

        df=test_df,

        normalize=args.normalize,

        transforms=test_transforms(size=args.size),

        data_root=args.test_image_path   

    )

    

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.learning_rate

    )

    scheduler = None

    

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        sampler=BalanceClassSampler(labels=train_dataset.__get_labels__(), mode="downsampling"),

        drop_last=True,

        num_workers=4

    )

    

    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=args.batch_size,

        drop_last=False,

        num_workers=4

    )

    

    test_loader = torch.utils.data.DataLoader(

        test_dataset,

        batch_size=args.batch_size,

        drop_last=False,

        num_workers=4

    )

    

    best_auc = 0

    preds_list = []

    

    print("Training started ..... ")

    

    for epoch in range(args.epochs):

        

        adjust_learning_rate(optimizer, epoch)

        

        train_auc, train_loss = train_epoch(

            args,

            model,

            train_loader,

            optimizer,

            scheduler,

            device,

            epoch

        )

        

        if epoch >= 40:

            

            valid_auc = valid_epoch(

                args,

                model,

                valid_loader,

                device,

                epoch

            )

            

            print(f"Epoch : {epoch} - AUC : {valid_auc}")

            

            if valid_auc > best_auc:

                print(f"###***### Model Improved from {best_auc} to {valid_auc}")

                torch.save(model.state_dict(), os.path.join(args.save_path, f"fold-{fold}.bin"))

                best_auc = valid_auc

            

            preds = test_epoch(

                args,

                model,

                test_loader,

                device

            )

            

            preds_list.append(preds)

    

    final_preds = np.mean(preds_list, axis=0)       

    np.save(os.path.join(args.save_path, f"test-pred-fold-{fold}.npy"), final_preds)
# clean up gpu in case you are debugging 

import gc

torch.cuda.empty_cache(); gc.collect()

torch.cuda.empty_cache(); gc.collect()
skf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):

    main(fold, idxT, idxV)

    

    
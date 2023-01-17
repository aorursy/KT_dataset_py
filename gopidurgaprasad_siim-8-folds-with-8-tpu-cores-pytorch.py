!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!pip install pretrainedmodels

!pip install efficientnet_pytorch
import pandas as pd

import numpy as np

from sklearn import model_selection



import os

import random



from sklearn import metrics



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp





import albumentations

from PIL import Image, ImageFile





import pretrainedmodels

from efficientnet_pytorch import EfficientNet



from joblib import Parallel, delayed



import warnings

warnings.filterwarnings('ignore')



from tqdm.autonotebook import tqdm
FOLDS = 8



train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

train_df["kfold"] = -1

train_df = train_df.sample(frac=1).reset_index(drop=True)

y = train_df.target.values

kfold = model_selection.StratifiedKFold(n_splits=FOLDS)



for f, (tra_, val_) in enumerate(kfold.split(X=train_df, y=y)):

    train_df.loc[val_, "kfold"] = f





train_df.kfold.value_counts()

class SIIMDataset:

    def __init__(self, args, df, mode="train", fold=0):



        self.mode = mode



        mean = (0.485, 0.456, 0.406)

        std = (0.229, 0.224, 0.225)



        if self.mode == "train":

            df = df[~df.kfold.isin([fold])].dropna()

            self.image_names = df.image_name.values

            self.targets = df.target.values



            self.aug = albumentations.Compose(

                [

                    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),

                    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),

                    albumentations.Flip(p=0.5)

                ]

            )



        if self.mode == "valid":

            df = df[df.kfold.isin([fold])].dropna()

            self.image_names = df.image_name.values

            self.targets = df.target.values



            self.aug = albumentations.Compose(

                [

                    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)

                ]

            )



    def __len__(self):

        return len(self.image_names)

        

    def __getitem__(self,item):

        

        image_path = f"../input/siic-isic-224x224-images/train/{self.image_names[item]}.png"

        image = Image.open(image_path)

        target = self.targets[item]



        image = np.array(image)

        augmented = self.aug(image=image)

        image = augmented["image"]



        image = np.transpose(image, (2, 0, 1)).astype(np.float32)



        return {

            "image" : torch.tensor(image, dtype=torch.float),

            "target": torch.tensor(target, dtype=torch.long)

        }
class ResNet50(nn.Module):

    def __init__(self, pretrained="imagenet"):

        super(ResNet50, self).__init__()



        self.base_model = pretrainedmodels.__dict__[

            "resnet50"

        ](pretrained=pretrained)



        self.l0 = nn.Linear(2048, 1)



    def forward(self, image):

        bs, _, _, _ = image.shape



        x = self.base_model.features(image)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)



        out = self.l0(x)



        return out



class EfficientNet3(nn.Module):

    def __init__(self, pretrained="imagenet"):

        super(EfficientNet3, self).__init__()



        self.base_model = EfficientNet.from_pretrained("efficientnet-b0")



        self.l0 = nn.Linear(1536, 1)



    def forward(self, image):

        bs, _, _, _ = image.shape



        x = self.base_model.extract_features(image)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)



        out = self.l0(x)



        return out


class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.0001):

        self.patience = patience

        self.counter = 0

        self.mode = mode

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        if self.mode == "min":

            self.val_score = np.Inf

        else:

            self.val_score = -np.Inf



    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":

            score = -1.0 * epoch_score

        else:

            score = np.copy(epoch_score)



        if self.best_score is None:

            self.best_score = score

            #self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            #print(

            #    "EarlyStopping counter: {} out of {}".format(

            #        self.counter, self.patience

            #    )

            #)

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            #self.save_checkpoint(epoch_score, model, model_path)

            self.counter = 0



    def save_checkpoint(self, epoch_score, model, model_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            print(

                "Validation score improved ({} --> {}). Saving model!".format(

                    self.val_score, epoch_score

                )

            )

            torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score
def to_list(tensor):

    return tensor.detach().cpu().tolist()



class AverageMeter(object):

    """Computes and stores the average and current values"""

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



def reduce_fn(vals):

    return sum(vals) / len(vals)



def loss_fn(preds, labels):

    return nn.BCEWithLogitsLoss()(preds, labels.view(-1, 1).type_as(preds))





def train(args, train_loader, model, device, optimizer, epoch):

    total_loss = AverageMeter()



    model.train()



    t = tqdm(train_loader, disable=not xm.is_master_ordinal())

    for step, d in enumerate(t):

        

        image = d["image"].to(device)

        target = d["target"].to(device)



        model.zero_grad()



        logits = model(

            image

        )



        loss = loss_fn(logits, target)

        n_position1 = target.shape[0]

        total_loss.update(loss.item())

        

        loss.backward()

        xm.optimizer_step(optimizer, barrier=True)

        print_loss = total_loss.avg

        

        t.set_description(f"Train E:{epoch+1} - Loss:{print_loss:0.4f}")

    

    return total_loss.avg





def valid(args, valid_loader, model, device, epoch):

    total_loss = AverageMeter()

    final_predictions = []

    final_targets = []



    model.eval()



    with torch.no_grad():

        t = tqdm(valid_loader, disable=not xm.is_master_ordinal())

        for step, d in enumerate(t):

            

            image = d["image"].to(device)

            target = d["target"].to(device)



            model.zero_grad()



            logits = model(

                image

            )



            loss = loss_fn(logits, target)

            n_position1 = target.shape[0]

            total_loss.update(loss.item(), n_position1)



            print_loss = total_loss.avg

            t.set_description(f"Train E:{epoch+1} - Loss:{print_loss:0.4f}")



            predictions = to_list(logits)

            targets = to_list(target)



            final_predictions.append(predictions)

            final_targets.append(targets)



    final_predictions = np.concatenate(final_predictions)

    final_targets = np.concatenate(final_targets)



    auc = metrics.roc_auc_score(final_targets, final_predictions)

    

    return total_loss.avg, auc
    

def run(fold_index):

    

    args.fold_index = fold_index

    

    MX = ResNet50(pretrained=None)

    



    device = xm.xla_device(fold_index+1)

    model = MX.to(device)

    

    args.save_path = os.path.join(args.output_dir, args.exp_name)



    if not os.path.exists(args.save_path):

        os.makedirs(args.save_path)



    # DataLoaders

    train_dataset = SIIMDataset(

        args=args, 

        df=train_df, 

        mode="train", 

        fold=args.fold_index

    )



    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        drop_last=False,

        shuffle=True

    )



    valid_dataset =  SIIMDataset(

        args=args, 

        df=train_df, 

        mode="valid", 

        fold=args.fold_index

    )

    

    valid_loader = DataLoader(

        valid_dataset,

        batch_size=args.batch_size,

        drop_last=False,

        shuffle=True

    )





    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=args.learning_rate

    )



    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        patience=3,

        threshold=0.001,

        mode="max"

    )



    best_auc = 0

    early_stopping = EarlyStopping(patience=3, mode="max")



    for epoch in range(args.epochs):

        

        train_loss = train(

            args, 

            train_loader,

            model,

            device,

            optimizer,

            epoch

        )



        valid_loss, valid_auc = valid(

            args, 

            valid_loader,

            model,

            device,

            epoch,

        )





        auc = valid_auc

        val_loss = valid_loss



        scheduler.step(val_loss)



        print(f"Fold {fold_index} ** Epoch {epoch+1} **==>** AUC = {auc}")

        print(f"Fold {fold_index} ** Epoch {epoch+1} **==>** valid_loss = {val_loss}")



        if auc > best_auc:

            xm.save(model.state_dict(), os.path.join(args.save_path, f"fold_{fold_index}.bin"))

            best_auc = auc



        early_stopping(auc, model, "none")



        if early_stopping.early_stop:

            print("Early stopping")

            break
class args:

    

    learning_rate = 0.00002

    epochs = 5

    batch_size = 64

    output_dir = "resnet50"

    exp_name = "base_model"

    seed = 42
Parallel(n_jobs=8, backend="threading")(delayed(run)(i) for i in range(8))
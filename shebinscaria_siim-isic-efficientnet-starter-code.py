import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import imageio
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from albumentations.pytorch.transforms import ToTensorV2
from sklearn import model_selection
from sklearn import metrics
!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
class stratification:
    def __init__(self,path_dir,num_splits):
        self.input_path = path_dir
        self.n_splits = num_splits
        self.df = pd.read_csv(os.path.join(input_path,"train.csv"))
        
    def create_split(self):
        self.df['kfold'] = -1
        #Shuffling the csv file => to get a new shuffled dataframe
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        #Target value
        y=self.df.target.values
        #Why stratified - because we want the ratio of +ve:-ve samples to be the same
        kf = model_selection.StratifiedKFold(n_splits=self.n_splits)
        
        kfold_df_dict = {}
        
        for fold_, (train_idx, val_idx) in enumerate(kf.split(X=self.df,y=y)):
            df_temp = pd.read_csv(os.path.join(input_path,"train.csv"))
            df_temp['kfold'] = -1
            df_temp['dataset_type'] = 'train'
            df_temp.loc[:,'kfold']=fold_
            df_temp.loc[val_idx,'dataset_type'] = 'val'
            kfold_df_dict[fold_]=df_temp
        
        df_comb_fold = pd.concat(kfold_df_dict[k] for (k,v) in kfold_df_dict.items())
        
        return df_comb_fold
input_path = "/kaggle/input/siim-isic-melanoma-classification"
num_splits = 2
df_actual_train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
df_kfold = stratification(input_path,num_splits).create_split()
class model_efficientnetb7(torch.nn.Module):
    def __init__(self, pretrained=True):
        
        super(model_efficientnetb7,self).__init__()
        
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        ## Changing the last layer
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, 1)
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self,image,targets):
        # Arguments should match your dataloader arguments wrt dataset being passed
        # in this case it is image, targets
        
        
        out = self.model(image)
        
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1,1).type_as(out)
        )
        # shape and datatype
        return out,loss
class ClassificationLoader:
    def __init__(self, image_paths, targets, resize, augmentations):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
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
class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1
    ):
        
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
        tk0 = tqdm(data_loader, total=len(data_loader),disable=False)
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            _, loss = model(**data)
                   
            with torch.set_grad_enabled(True):
                loss.backward()
                if (b_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    if b_idx > 0:
                        optimizer.zero_grad()
            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device):
        losses = AverageMeter()
        final_predictions = []
        model.eval()
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            tk0 = tqdm(data_loader, total=len(data_loader), disable=False)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, loss = model(**data)
                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
        return final_predictions, losses.avg

    @staticmethod
    def predict(data_loader, model, device, use_tpu=False):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, _ = model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions
def train(fold):
    print(f"Starting Training for fold = {fold+1}")
    #Image size requirements per EfficientNet documentation
    training_data_path = "/kaggle/input/siic-isic-224x224-images/train"
    df = df_kfold[df_kfold['kfold']==fold]
    device = 'cuda'
    epochs = 50
    train_bs = 4
    val_bs = 4
    
    df_train = df.loc[df['dataset_type']=='train',list(df_actual_train.columns)]
    df_val = df.loc[df['dataset_type']=='val',list(df_actual_train.columns)]
    # Normalization needed as per EfficientNet documentation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # add any extra augmentations here
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True,p=1.0)
        ]
    )
    val_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True,p=1.0)
        ]
    )
    train_images_list = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path,i + '.png') for i in train_images_list]
    train_targets = df_train.target.values
    
    val_images_list = df_val.image_name.values.tolist()
    val_images = [os.path.join(training_data_path,i + '.png') for i in val_images_list]
    val_targets = df_val.target.values
    
    train_dataset = ClassificationLoader(
        image_paths = train_images,
        targets= train_targets,
        resize = None,
        augmentations = train_aug
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_bs,
        shuffle = False,
        num_workers=0
    )
    
    val_dataset = ClassificationLoader(
        image_paths = val_images,
        targets= val_targets,
        resize = None,
        augmentations = val_aug
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = val_bs,
        shuffle = False,
        num_workers=0
    )
    #Earlier defined class for model
    #model = Model_Inception_v3(pretrained='imagenet')
    model = model_efficientnetb7()
    model.to(device)
    
    #Specify an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #Specify an scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode='max'
    )
    # why mode='max' becauase we will be using the metric of AUC
    
    # we would also need early stopping
    es = EarlyStopping(patience=5, mode='max')
    
    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device
        )
        predictions, val_loss = Engine.evaluate(
            val_loader,
            model,
            device
        )
        
        predictions = np.vstack((predictions)).ravel()
        # Ravel it because we have only one value
        auc = metrics.roc_auc_score(val_targets, predictions)
        # thats why val_loader shuffle was kept false
        
        scheduler.step(auc)
        print(f"epoch={epoch},auc={auc}")
        # Save it with .bin extension
        model_path = f'efficient_model_fold{fold}.bin'
        es(auc, model, model_path)
        if es.early_stop:
            print("Early Stopping")
            break
def predict(fold):
    print(f"Generating Predictions for saved model, fold = {fold+1}")
    test_data_path = "/kaggle/input/siic-isic-224x224-images/test"
    df_test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
    df_test.loc[:,'target'] = 0
    
    #model_path = "f'/kaggle/working/model_fold{fold}'"
    #model_path = '/kaggle/working/model_fold0_epoch0.bin'
    model_path = f'/kaggle/working/efficient_model_fold{fold}.bin'
    
    device = 'cuda'
    
    test_bs = 16
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True,p=1.0)
        ]
    )
    test_images_list = df_test.image_name.values.tolist()
    test_images = [os.path.join(test_data_path,i + '.png') for i in test_images_list]
    test_targets = df_test.target.values
    
    test_dataset = ClassificationLoader(
        image_paths = test_images,
        targets= test_targets,
        resize = None,
        augmentations = test_aug
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = test_bs,
        shuffle = False,
        num_workers=4
    )
    #Earlier defined class for model
    model = model_efficientnetb7()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    predictions_op = Engine.predict(
        test_loader,
        model,
        device
    )
    return np.vstack((predictions_op)).ravel()
#Training
for fold_ in range(num_splits):
    torch.cuda.empty_cache()
    train(fold=fold_)
    list_fold_pred.append(predict(fold_))
#Generating Predictions
list_fold_pred = []
for fold_ in range(num_splits):    
    list_fold_pred.append(predict(fold_))
pred = np.mean(np.vstack(list(list_fold_pred[i] for i in range(1))),axis=0)
sample = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sample.loc[:, "target"] = pred
sample.to_csv("submission.csv", index=False)
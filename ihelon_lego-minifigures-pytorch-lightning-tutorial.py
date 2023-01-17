!pip install -q --upgrade pip

!pip install -q pytorch-lightning
import os

import math

import time

import random

import warnings



import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import seaborn as sn

import albumentations as A

import torch

from torch.utils import data as torch_data

from torch import nn as torch_nn

from torch.nn import functional as torch_F

import torchvision

import pytorch_lightning as pl

from pytorch_lightning import metrics as pl_metrics

from pytorch_lightning import callbacks as pl_callbacks

from pytorch_lightning.core.decorators import auto_move_data

from sklearn import metrics as sk_metrics



warnings.filterwarnings("ignore")
def set_seed(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True





SEED = 42

set_seed(SEED)
# The base dataset directory

BASE_DIR = '../input/lego-minifigures-classification/'



df_metadata = pd.read_csv(os.path.join(BASE_DIR, 'metadata.csv'), index_col=0)

N_CLASSES = df_metadata.shape[0]

print('Number of classes: ', N_CLASSES)
class DataRetriever(torch_data.Dataset):

    def __init__(

        self, 

        paths, 

        targets, 

        image_size=(224, 224),

        transforms=None

    ):

        self.paths = paths

        self.targets = targets

        self.image_size = image_size

        self.transforms = transforms

        self.preprocess = torchvision.transforms.Compose([

            torchvision.transforms.ToTensor(),

            torchvision.transforms.Normalize(

                mean=[0.485, 0.456, 0.406], 

                std=[0.229, 0.224, 0.225]

            ),

        ])

          

    def __len__(self):

        return len(self.targets)

    

    def __getitem__(self, index):

        img = cv2.imread(self.paths[index])

        img = cv2.resize(img, self.image_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:

            img = self.transforms(image=img)['image']

            

        img = self.preprocess(img)

        

        y = torch.tensor(self.targets[index], dtype=torch.long)

            

        return {'X': img, 'y': y}
def get_train_transforms():

    return A.Compose(

        [

            A.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.5),

            A.Cutout(num_holes=8, max_h_size=25, max_w_size=25, fill_value=0, p=0.25),

            A.Cutout(num_holes=8, max_h_size=25, max_w_size=25, fill_value=255, p=0.25),

            A.HorizontalFlip(p=0.5),

            A.RandomContrast(limit=(-0.3, 0.3), p=0.5),

            A.RandomBrightness(limit=(-0.4, 0.4), p=0.5),

            A.Blur(p=0.25),

        ], 

        p=1.0

    )
class LEGOMinifiguresDataModule(pl.LightningDataModule):

    def __init__(

        self, 

        train_batch_size, 

        valid_batch_size, 

        image_size, 

        base_dir,

        train_augmentations=None

    ):

        super().__init__()

        self.train_batch_size = train_batch_size

        self.valid_batch_size = valid_batch_size

        self.image_size = image_size

        self.base_dir = base_dir

        self.train_augmentations=train_augmentations

        

    def prepare_data(self):

        self.df = pd.read_csv(os.path.join(self.base_dir, 'index.csv'), index_col=0)



    def setup(self, stage):

        tmp_train = self.df[self.df['train-valid'] == 'train']

        train_paths = tmp_train['path'].values

        self.train_targets = tmp_train['class_id'].values - 1

        self.train_paths = list(map(lambda x: os.path.join(self.base_dir, x), train_paths))

        

        tmp_valid = self.df[self.df['train-valid'] == 'valid']

        valid_paths = tmp_valid['path'].values

        self.valid_targets = tmp_valid['class_id'].values - 1

        self.valid_paths = list(map(lambda x: os.path.join(self.base_dir, x), valid_paths))

        

    def train_dataloader(self):

        train_data_retriever = DataRetriever(

            self.train_paths, 

            self.train_targets, 

            image_size=self.image_size,

            transforms=self.train_augmentations

        )

        

        train_loader = torch_data.DataLoader(

            train_data_retriever,

            batch_size=self.train_batch_size,

            shuffle=True,

        )

        return train_loader

    

    def val_dataloader(self):

        valid_data_retriever = DataRetriever(

            self.valid_paths, 

            self.valid_targets, 

            image_size=self.image_size,

        )

        

        valid_loader = torch_data.DataLoader(

            valid_data_retriever, 

            batch_size=self.valid_batch_size,

            shuffle=False,

        )

        return valid_loader
class LitModel(pl.LightningModule):

    

    def __init__(self, n_classes):

        super().__init__()

        self.net = torch.hub.load(

            'pytorch/vision:v0.6.0', 

            'mobilenet_v2', 

            pretrained=True

        )

        self.net.classifier = torch_nn.Linear(

            in_features=1280, 

            out_features=n_classes, 

            bias=True

        )

        self.save_hyperparameters()



    @auto_move_data

    def forward(self, x):

        x = self.net(x)

        return x

    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        return optimizer

    

    def training_step(self, batch, batch_idx):

        X, y = batch['X'], batch['y']

        y_hat = self(X)

        train_loss = torch_F.cross_entropy(y_hat, y)

        train_acc = pl_metrics.functional.accuracy(

            y_hat, 

            y, 

            num_classes=self.hparams.n_classes

        )

        

        result = pl.TrainResult(train_loss)

        result.log('train_loss', train_loss, prog_bar=True, on_epoch=True, on_step=False)

        result.log('train_acc', train_acc, prog_bar=True, on_epoch=True, on_step=False)

        return result

    

    def validation_step(self, batch, batch_idx):

        X, y = batch['X'], batch['y']

        y_hat = self(X)

        

        valid_loss = torch_F.cross_entropy(y_hat, y)

        valid_acc = pl_metrics.functional.accuracy(

            y_hat, 

            y, 

            num_classes=self.hparams.n_classes

        )

        

        result = pl.EvalResult(checkpoint_on=valid_loss, early_stop_on=valid_loss)

        result.log('valid_loss', valid_loss, prog_bar=True, on_epoch=True, on_step=False)

        result.log('valid_acc', valid_acc, prog_bar=True, on_epoch=True, on_step=False)

        return result

    
model = LitModel(n_classes=N_CLASSES)



data_module = LEGOMinifiguresDataModule(

    train_batch_size=4, 

    valid_batch_size=1, 

    image_size=(512, 512), 

    base_dir=BASE_DIR,

    train_augmentations=get_train_transforms()

)



callback_early_stopping = pl_callbacks.EarlyStopping(

    'valid_loss', 

    patience=3, 

    mode='min'

)

callback_model_checkpoint = pl_callbacks.ModelCheckpoint(

    '{epoch}-{valid_loss:.3f}', 

    monitor='valid_loss', 

    mode='min'

)



trainer = pl.Trainer(

    gpus=1,

    early_stop_callback=callback_early_stopping,

    checkpoint_callback=callback_model_checkpoint, 

    max_epochs=50

)



trainer.fit(

    model, 

    data_module,

)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



best_model_path = callback_model_checkpoint.best_model_path



model = LitModel.load_from_checkpoint(

    checkpoint_path=best_model_path

)

model = model.to(device)

model.freeze()
# Save the model predictions and true labels

y_pred = []

y_valid = []

for ind, batch in enumerate(data_module.val_dataloader()):

    pred_probs = model(batch['X'])

    y_pred.extend(pred_probs.argmax(axis=-1).cpu().numpy())

    y_valid.extend(batch['y'])

    



# Calculate needed metrics

print(f'Accuracy score on validation data:\t{sk_metrics.accuracy_score(y_valid, y_pred)}')

print(f'Macro F1 score on validation data:\t{sk_metrics.f1_score(y_valid, y_pred, average="macro")}')
# Load metadata to get classes people-friendly names

labels = df_metadata['minifigure_name'].tolist()



# Calculate confusion matrix

confusion_matrix = sk_metrics.confusion_matrix(y_valid, y_pred)

df_confusion_matrix = pd.DataFrame(confusion_matrix, index=labels, columns=labels)



# Show confusion matrix

plt.figure(figsize=(12, 12))

sn.heatmap(df_confusion_matrix, annot=True, cbar=False, cmap='Oranges', linewidths=1, linecolor='black')

plt.xlabel('Predicted labels', fontsize=15)

plt.xticks(fontsize=12)

plt.ylabel('True labels', fontsize=15)

plt.yticks(fontsize=12);
error_images = []

error_label = []

error_pred = []

error_prob = []

for batch in data_module.val_dataloader():

    _X_valid, _y_valid = batch['X'], batch['y']

    pred = torch.softmax(model(_X_valid), axis=-1).cpu().numpy()

    pred_class = pred.argmax(axis=-1)

    if pred_class != _y_valid.cpu().numpy():

        error_images.extend(_X_valid)

        error_label.extend(_y_valid)

        error_pred.extend(pred_class)

        error_prob.extend(pred.max(axis=-1))
def denormalize_image(image):

    return image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]



plt.figure(figsize=(16, 16))

w_size = int(len(error_images) ** 0.5)

h_size = math.ceil(len(error_images) / w_size)

for ind, image in enumerate(error_images):

    plt.subplot(h_size, w_size, ind + 1)

    plt.imshow(denormalize_image(image.permute(1, 2, 0).numpy()))

    pred_label = labels[error_pred[ind]]

    pred_prob = error_prob[ind]

    true_label = labels[error_label[ind]]

    plt.title(f'predict: {pred_label} ({pred_prob:.2f}) true: {true_label}')

    plt.axis('off')
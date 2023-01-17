!pip install pytorch_lightning GPUtil > /dev/null
import os

import zipfile

import GPUtil

import random

import pytorch_lightning as pl

import numpy as np

import pandas as pd

import seaborn as sns

import torch

import torchvision

import matplotlib.pyplot as plt



from torch import nn

from torch.nn import functional as F

from torch.utils.data import (Dataset, DataLoader)

from torchvision.transforms import (

        Resize,

        Compose,

        ToTensor,

        Normalize,

        RandomOrder,

        ColorJitter,

        RandomRotation,

        RandomGrayscale,

        RandomResizedCrop,

        RandomVerticalFlip,

        RandomHorizontalFlip)



from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split

from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.callbacks import ModelCheckpoint



%matplotlib inline
np.__version__, pd.__version__, sns.__version__
torch.__version__, torchvision.__version__, pl.__version__
RNG_SEED = 9527

DATA_ROOT = '/kaggle/input/dogs-vs-cats'

WORK_ROOT = '/kaggle/working'

CKPT_PATH = f'{WORK_ROOT}/checkpoints/best.ckpt'

IMGS_ROOT = f'{WORK_ROOT}/temp_unzip'

SUBMITCSV = f'{WORK_ROOT}/submission.csv'

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'

LABEL_ID_MAP = {'dog': 0, 'cat': 1}

ID_LABEL_MAP = {0: 'dog', 1: 'cat'}



INPUT_SIZE = 224

BATCH_SIZE = 128

NUM_CLASSES = 2



MAX_EPOCHS = 3



DATASET_MEAN = (0.485, 0.456, 0.406)

DATASET_STD = (0.229, 0.224, 0.225)



TEST_SPLIT = 0.3
!ls -l $WORK_ROOT
torch.manual_seed(RNG_SEED)

np.random.seed(RNG_SEED)

random.seed(RNG_SEED)



torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
if not os.path.exists(f'{IMGS_ROOT}/train'):

    with zipfile.ZipFile(f'{DATA_ROOT}/train.zip', 'r') as z:

        z.extractall(f'{IMGS_ROOT}')

        

if not os.path.exists(f'{IMGS_ROOT}/test1'):

    with zipfile.ZipFile(f'{DATA_ROOT}/test1.zip', 'r') as z:

        z.extractall(f'{IMGS_ROOT}')
filenames = os.listdir(f'{IMGS_ROOT}/train')

label_ids = [LABEL_ID_MAP[str(fname)[:3]] for fname in filenames]

train_df = pd.DataFrame({'filename': filenames, 'label': label_ids})

train_df[10:15]
train_df, valid_df = train_test_split(train_df, test_size = TEST_SPLIT)

valid_df[10:15]
filenames = os.listdir(f'{IMGS_ROOT}/test1')

label_ids = [ -1 for x in filenames]

test_df = pd.DataFrame({'filename': filenames, 'label': label_ids})

test_df[10:15]
sns.countplot(x='label',data=train_df).set_title("Train Data Distribution");
sns.countplot(x='label',data=valid_df).set_title("Valid Data Distribution");
def draw_image(filepath, labelname, resize=None, augtrans=None):

    img = Image.open(filepath).convert('RGB')

    if resize is not None:

        img = img.resize((resize, resize))

    if augtrans is not None:

        img = augtrans(img)

        

    font_obj = ImageFont.truetype(FONT_PATH, 48)

    draw_img = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    draw_img.text((0, 0), labelname, font=font_obj, fill=(0, 0, 255))

    return np.array(img)



def grid_image(imgs_list, cols=4):

    images = torch.as_tensor(imgs_list) # [(W, H, C)...] to (B, H, W, C)

    images = images.permute(0, 3, 1, 2) # (B, H, W, C) to (B, C, H, W)

    images = torchvision.utils.make_grid(images, nrow=4) # (C, 2*H, 4*W)

    images = images.permute(1, 2, 0) # (H, W, C)

    return images
plt.figure(figsize=(24, 12))



images_2x4 = [

    draw_image(

        filepath=f'{IMGS_ROOT}/train/{row.filename}',

        labelname=f'{ID_LABEL_MAP[row.label]}',

        resize=INPUT_SIZE

    ) for _, row in train_df[:8].iterrows()

]



plt.imshow(grid_image(images_2x4, cols=4));
aug_trans = RandomOrder([

    RandomResizedCrop((INPUT_SIZE, INPUT_SIZE)),

    RandomRotation(degrees=10),

    RandomVerticalFlip(p=0.3),

    RandomHorizontalFlip(p=0.3),

    ColorJitter(brightness=0.55, contrast=0.3, saturation=0.25, hue=0),

])



img_trans = Compose([

    Resize((INPUT_SIZE, INPUT_SIZE)),

    ToTensor(),

    Normalize(mean=DATASET_MEAN, std=DATASET_STD),

])
plt.figure(figsize=(24, 12))



trans_images_2x4 = [

    draw_image(

        filepath=f'{IMGS_ROOT}/train/{row.filename}',

        labelname=f'{ID_LABEL_MAP[row.label]}',

        resize=INPUT_SIZE,

        augtrans = aug_trans

    ) for _, row in train_df[:8].iterrows()

]



plt.imshow(grid_image(trans_images_2x4, cols=4));
backbone = torchvision.models.resnet50(pretrained=True)
for param in backbone.parameters():

    param.requires_grad = False
extractor = list(backbone.children())[:-2] # avgpool and fc
METRICS = {

    'epoch':[],

    'train_loss':[],

    'train_acc':[],

    'val_acc':[],

    'val_loss':[],

}



class DCDataset(Dataset):

    def __init__(self, root, df, augtrans=None, imgtrans=ToTensor()):

        super().__init__()

        self.data = [(f'{root}/{row.filename}', row.label) for _, row in df.iterrows()]

        self.augtrans = augtrans

        self.imgtrans = imgtrans

    

    def __getitem__(self, index):

        imgpath, label = self.data[index]

        img = Image.open(imgpath).convert('RGB')

        if self.augtrans:

            img = self.augtrans(img)

        img = self.imgtrans(img)

        return img, label, imgpath

    

    def __len__(self):

        return len(self.data)



class DCNet(pl.LightningModule):

    def __init__(self, extractor, num_classes=NUM_CLASSES):

        super().__init__()

        self.features = nn.Sequential(

            *extractor, # 2048, 7, 7

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            nn.BatchNorm2d(num_features=256, momentum=0.1),

            nn.MaxPool2d(kernel_size=5, stride=1, padding=2, ceil_mode=False),

            nn.Dropout(inplace=True, p=0.5)

        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(

            nn.Flatten(start_dim=1, end_dim=-1),

            nn.Linear(in_features=256, out_features=128, bias=True),

            nn.Dropout(inplace=True, p=0.5),

            nn.Linear(in_features=128, out_features=num_classes, bias=True),

        )



  

    def forward(self, x, *args, **kwargs):

        x = self.features(x)

        x = self.avgpool(x)

        x = self.classifier(x)

        return x

        

    def setup(self, stage):

        torch.cuda.empty_cache()



    def teardown(self, stage):

        for idx, gpu in enumerate(GPUtil.getGPUs()):

            allocmem = round(torch.cuda.memory_allocated(idx) / 1024**2, 2)

            allocmax = round(torch.cuda.max_memory_allocated(idx) / 1024**2, 2)

            print(f'({stage})\tGPU-{idx} mem allocated: {allocmem} MB\t maxmem allocated: {allocmax} MB')

            

    @property

    def metrics(self):

        return self.metrics

        

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(

            filter(lambda p: p.requires_grad, model.parameters()),

            lr=0.001,

            weight_decay=0.001

        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode='min',

            factor=0.1,

            patience=3,

            min_lr=1e-6)

        return [optimizer], [scheduler]

    

    def prepare_data(self):

        self.train_dataset = DCDataset(f'{IMGS_ROOT}/train', train_df, aug_trans, img_trans) 

        self.valid_dataset = DCDataset(f'{IMGS_ROOT}/train', valid_df, None, img_trans) 

        self.test_dataset = DCDataset(f'{IMGS_ROOT}/test1', test_df, None, img_trans) 



    def train_dataloader(self):

        return DataLoader(

                self.train_dataset,

                batch_size=BATCH_SIZE,

                num_workers=4,

                drop_last=True,

                shuffle=True)

    

    def training_step(self, batch, batch_idx):

        x, y_true, path = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y_true, reduction='mean')

        acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()

        return {'loss': loss, 'acc': acc}



    def training_epoch_end(self, outputs):

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        acc = torch.stack([x['acc'] for x in outputs]).mean()

        METRICS['epoch'].append(self.current_epoch)

        METRICS['train_loss'].append(loss)

        METRICS['train_acc'].append(acc)

        return {'progress_bar': {'train_loss': loss, 'train_acc': acc}}



    def val_dataloader(self):

        return DataLoader(

            self.valid_dataset,

            batch_size=BATCH_SIZE,

            num_workers=4,

            drop_last=False,

            shuffle=False)

    

    def validation_step(self, batch, batch_idx):

        x, y_true, path = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred, y_true, reduction='mean')

        acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()

        return {'val_loss': loss, 'val_acc': acc}



    def validation_epoch_end(self, outputs):

        loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        METRICS['val_loss'].append(loss)

        METRICS['val_acc'].append(acc)

        return {'progress_bar': {'val_loss': loss, 'val_acc': acc}}

    

    def test_dataloader(self):

        return DataLoader(

            self.test_dataset,

            batch_size=BATCH_SIZE,

            num_workers=4,

            drop_last=False,

            shuffle=False)

    

    def test_step(self, batch, batch_idx):

        x, _, path = batch

        y_pred = torch.argmax(self(x), dim=1).cpu().numpy()

        log = {'imgid': [os.path.basename(x).split('.')[0] for x in path], 'label': y_pred}

        return log



    def test_epoch_end(self, outputs):

        imgid = np.concatenate([x['imgid'] for x in outputs])

        label = np.concatenate([x['label'] for x in outputs])

        return {'id': imgid, 'label': label}

    

class DCTrainer(pl.Trainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        

    def save_checkpoint(self, filepath, weights_only: bool = False):

        return super().save_checkpoint(CKPT_PATH, weights_only)
trainer = DCTrainer(

    max_epochs=MAX_EPOCHS,

    logger=False,

    log_gpu_memory='min_max',

    weights_summary='top',

    num_sanity_val_steps=0,

    progress_bar_refresh_rate=1,

    check_val_every_n_epoch=1,

    default_root_dir=WORK_ROOT,

    resume_from_checkpoint=CKPT_PATH if os.path.exists(CKPT_PATH) else None,

    early_stop_callback=EarlyStopping(monitor='val_loss', patience=7, mode='min'),

    checkpoint_callback=ModelCheckpoint(monitor='val_loss', period=5, mode='min'),

    gpus=[0],

)



model = DCNet(extractor)
trainer.fit(model);
result = trainer.test(model, verbose=False, ckpt_path=CKPT_PATH)
result_df = pd.DataFrame(data=result[0])

sns.countplot(x='label',data=result_df).set_title("Predict Data Distribution");
num_epoch = len(METRICS['epoch'])

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

axs[0].plot(METRICS['epoch'], METRICS['train_acc'])

axs[0].plot(METRICS['epoch'], METRICS['val_acc'])

axs[0].set_title('Accuracy')

axs[0].set_ylabel('Accuracy')

axs[0].set_xlabel('Epoch')

axs[0].legend(['train', 'val'], loc='best')



axs[1].plot(METRICS['epoch'], METRICS['train_loss'])

axs[1].plot(METRICS['epoch'], METRICS['val_loss'])

axs[1].set_title('Loss')

axs[1].set_ylabel('Loss')

axs[1].set_xlabel('Epoch')

axs[1].legend(['train', 'val'], loc='best');
result_df.to_csv(SUBMITCSV, index=False)
!rm -rf $IMGS_ROOT

!ls -l $WORK_ROOT
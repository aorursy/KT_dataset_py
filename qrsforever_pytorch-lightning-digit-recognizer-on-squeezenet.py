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



sns.set(style='white', font_scale=1.2)
np.__version__, pd.__version__, sns.__version__
torch.__version__, torchvision.__version__, pl.__version__
RNG_SEED = 9527

DATA_ROOT = '/kaggle/input/digit-recognizer'

WORK_ROOT = '/kaggle/working/digit-recognizer'

CKPT_PATH = f'{WORK_ROOT}/checkpoints/best.ckpt'

SUBMITCSV = '/kaggle/working/submission.csv'

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'



INPUT_SIZE = 28

BATCH_SIZE = 128

NUM_CLASSES = 10



MAX_EPOCHS = 30



DATASET_MEAN = (0.1307, 0.1307, 0.1307)

DATASET_STD = (0.3081, 0.3081, 0.3081)



TEST_SPLIT = 0.3
!ls -l $DATA_ROOT

!mkdir -p $WORK_ROOT
torch.manual_seed(RNG_SEED)

np.random.seed(RNG_SEED)

random.seed(RNG_SEED)



torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
train_df = pd.read_csv(f'{DATA_ROOT}/train.csv')
train_df, valid_df = train_test_split(train_df, test_size = TEST_SPLIT)

train_df[:2]
def preprocess_data(df):

    labels = df['label'].values

    images = np.uint8(np.array(df.drop(columns=['label'])).reshape(-1, 28, 28))

    return list(zip(labels, images))
train_data = preprocess_data(train_df)

valid_data = preprocess_data(valid_df)
test_df = pd.read_csv(f'{DATA_ROOT}/test.csv')
test_data = np.uint8(np.array(test_df).reshape(-1, 28, 28))
del test_df
print(f'Train Count: {len(train_data)}, Valid Count: {len(valid_data)}, Test Count: {len(test_data)}')
fig, axes = plt.subplots(nrows=3, ncols=5, sharey=True, figsize=(12,4))

C = random.randint(100, 200)

for r in range(3):

    for c in range(5):

        axes[r][c].set_xticks([])

        axes[r][c].set_yticks([])

        axes[r][c].imshow(test_data[(r+1)*(c+1) + C], cmap='gray') # 'gray_r'
sns.countplot(x='label', data=train_df).set_title("Train Label Distribution")

del train_df
sns.countplot(x='label',data=valid_df).set_title("Valid Label Distribution")

del valid_df
def draw_image(imgdata, labelname, resize=None, augtrans=None):

    img = Image.fromarray(imgdata).convert('RGB')

    if resize is not None:

        img = img.resize((resize, resize))

    if augtrans is not None:

        img = augtrans(img)

        

    font_obj = ImageFont.truetype(FONT_PATH, 12)

    draw_img = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    draw_img.text((0, 0), labelname, font=font_obj, fill=(0, 0, 255))

    return np.array(img)



def grid_image(imgs_list, cols=5):

    images = torch.as_tensor(imgs_list) # [(W, H, C)...] to (B, H, W, C)

    images = images.permute(0, 3, 1, 2) # (B, H, W, C) to (B, C, H, W)

    images = torchvision.utils.make_grid(images, nrow=cols) # (C, 2*H, 4*W)

    images = images.permute(1, 2, 0) # (H, W, C)

    return images
plt.figure(figsize=(16, 8))



images_2x5 = [

    draw_image(

        imgdata=imgdata,

        labelname=str(label),

    ) for label, imgdata in train_data[:10]

]



plt.xticks([])

plt.yticks([])

plt.imshow(grid_image(images_2x5, cols=5), cmap='gray');
aug_trans = RandomOrder([

    RandomRotation(degrees=30),

    # RandomVerticalFlip(p=0.3),

    # RandomHorizontalFlip(p=0.3),

    # ColorJitter(brightness=0.55, contrast=0.3, saturation=0.25, hue=0),

])



img_trans = Compose([

    ToTensor(),

    Normalize(mean=DATASET_MEAN, std=DATASET_STD),

])
plt.figure(figsize=(24, 12))



trans_images_2x5 = [

    draw_image(

        imgdata=imgdata,

        labelname=str(label),

        augtrans = aug_trans

    ) for label, imgdata in train_data[:10]

]



plt.xticks([])

plt.yticks([])

plt.imshow(grid_image(trans_images_2x5, cols=5));
backbone = torchvision.models.squeezenet1_1(pretrained=True)
fz_count = 6

for param in backbone.features.parameters():

    pass

    param.requires_grad = False

    fz_count -= 1

    if fz_count < 0:

        break
class DCDataset(Dataset):

    def __init__(self, data, augtrans=None, imgtrans=ToTensor()):

        super().__init__()

        self.data = data

        self.augtrans = augtrans

        self.imgtrans = imgtrans

    

    def __getitem__(self, index):

        label, imgdata = self.data[index]

        img = Image.fromarray(imgdata).convert('RGB')

        if self.augtrans:

            img = self.augtrans(img)

        img = self.imgtrans(img)

        return img, label

    

    def __len__(self):

        return len(self.data)

    

class TDCDataset(DCDataset):

    def __getitem__(self, index):

        imgdata = self.data[index]

        img = Image.fromarray(imgdata).convert('RGB')

        if self.augtrans:

            img = self.augtrans(img)

        img = self.imgtrans(img)

        return img, index+1 # id

        

class DCNet(pl.LightningModule):

    def __init__(self, extractor, num_classes=NUM_CLASSES):

        super().__init__()

        self.features = extractor

        self.classifier = nn.Sequential(

            nn.Dropout(p=0.3),

            nn.Conv2d(512, num_classes, kernel_size=1),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

            nn.Flatten(start_dim=1, end_dim=-1)

        )

  

    def forward(self, x, *args, **kwargs):

        x = self.features(x)

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

        optimizer = torch.optim.SGD(

            filter(lambda p: p.requires_grad, model.parameters()),

            lr=0.001,

        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode='min',

            factor=0.1,

            patience=3,

            min_lr=1e-6)

        return [optimizer], [scheduler]

    

    def prepare_data(self):

        self.train_dataset = DCDataset(train_data, aug_trans, img_trans) 

        self.valid_dataset = DCDataset(valid_data, None, img_trans) 

        self.test_dataset = TDCDataset(test_data, None, img_trans)



    def train_dataloader(self):

        return DataLoader(

                self.train_dataset,

                batch_size=BATCH_SIZE,

                num_workers=4,

                drop_last=True,

                shuffle=True)

    

    def training_step(self, batch, batch_idx):

        x, y_true = batch

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

        x, y_true = batch

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

        x, imgid = batch

        y_pred = torch.argmax(self(x), dim=1).cpu().numpy()

        log = {'imgid': imgid.cpu().numpy(), 'label': y_pred}

        return log



    def test_epoch_end(self, outputs):

        imgid = np.concatenate([x['imgid'] for x in outputs])

        label = np.concatenate([x['label'] for x in outputs])

        return {'ImageId': imgid, 'Label': label}

    

class DCTrainer(pl.Trainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        

    def save_checkpoint(self, filepath, weights_only: bool = False):

        return super().save_checkpoint(CKPT_PATH, weights_only)



    def on_validation_start(self):

        lrs = []

        for scheduler in self.lr_schedulers:

            ss = scheduler['scheduler']

            if isinstance(ss, torch.optim.lr_scheduler.ReduceLROnPlateau):

                for i, param_group in enumerate(ss.optimizer.param_groups):

                    lrs.append(np.float32(param_group['lr']))

            else:

                lrs.extend([np.float32(x) for x in ss.get_lr()])

        self.add_progress_bar_metrics({'lr': lrs})

        return super().on_validation_start()
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



model = DCNet(backbone.features, num_classes=NUM_CLASSES)
METRICS = {

    'epoch':[],

    'train_loss':[],

    'train_acc':[],

    'val_acc':[],

    'val_loss':[],

}

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
!rm -rf $CKPT_PATH

!ls -l 
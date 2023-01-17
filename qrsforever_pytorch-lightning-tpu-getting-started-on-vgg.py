!pip uninstall -y torch torchvision 
file_url = 'https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py'

!curl {file_url} -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version 20200529 --apt-packages libomp5 libopenblas-dev
!pip install pytorch_lightning
!pip install tfrecord
import os

import io

import warnings

import random

import glob

import psutil

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



# from torchsummary import summary

from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.callbacks import ModelCheckpoint



from tfrecord import reader

from tfrecord.tools.tfrecord2idx import create_index



%matplotlib inline



sns.set(style='white', font_scale=1.2)

warnings.filterwarnings("ignore")
np.__version__, pd.__version__, sns.__version__
torch.__version__, torchvision.__version__, pl.__version__
RNG_SEED = 9527



KGGL_NAME = 'tpu-getting-started'

KGGL_ROOT = '/kaggle/working'

DATA_ROOT = f'/kaggle/input/{KGGL_NAME}'

WORK_ROOT = f'{KGGL_ROOT}/{KGGL_NAME}'

CKPT_PATH = f'{WORK_ROOT}/checkpoints/best.ckpt'

SUBMITCSV = f'{KGGL_ROOT}/submission.csv'

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'



INPUT_SIZE = 192 # 192, 224, 331, 512

BATCH_SIZE = 64

NUM_WORKERS = 1 # psutil.cpu_count()



MAX_EPOCHS = 50



DATASET_MEAN = (0.5, 0.5, 0.5)

DATASET_STD = (0.5, 0.5, 0.5)



CLASS_NAMES = [

    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 

    'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 

    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 

    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 

    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 

    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 

    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 

    'carnation', 'garden phlox', 'love in the mist', 'cosmos',  'alpine sea holly', 

    'ruby-lipped cattleya', 'cape flower', 'great masterwort',  'siam tulip', 

    'lenten rose', 'barberton daisy', 'daffodil',  'sword lily', 'poinsettia', 

    'bolero deep blue',  'wallflower', 'marigold', 'buttercup', 'daisy', 

    'common dandelion', 'petunia', 'wild pansy', 'primula',  'sunflower', 

    'lilac hibiscus', 'bishop of llandaff', 'gaura',  'geranium', 'orange dahlia', 

    'pink-yellow dahlia', 'cautleya spicata',  'japanese anemone', 'black-eyed susan', 

    'silverbush', 'californian poppy',  'osteospermum', 'spring crocus', 'iris', 

    'windflower',  'tree poppy', 'gazania', 'azalea', 'water lily',  'rose', 

    'thorn apple', 'morning glory', 'passion flower',  'lotus', 'toad lily', 

    'anthurium', 'frangipani',  'clematis', 'hibiscus', 'columbine', 'desert-rose', 

    'tree mallow', 'magnolia', 'cyclamen ', 'watercress',  'canna lily', 

    'hippeastrum ', 'bee balm', 'pink quill',  'foxglove', 'bougainvillea', 

    'camellia', 'mallow',  'mexican petunia',  'bromelia', 'blanket flower', 

    'trumpet creeper',  'blackberry lily', 'common tulip', 'wild rose']



NUM_CLASSES = len(CLASS_NAMES)
!ls -l $DATA_ROOT

!mkdir -p $WORK_ROOT
torch.manual_seed(RNG_SEED)

np.random.seed(RNG_SEED)

random.seed(RNG_SEED)



# torch.backends.cudnn.deterministic = True

# torch.backends.cudnn.benchmark = False
IMG_SCALE = 'tfrecords-jpeg-{}x{}'.format(INPUT_SIZE, INPUT_SIZE)

train_files = glob.glob(f'{DATA_ROOT}/{IMG_SCALE}/train/*.tfrec')

valid_files = glob.glob(f'{DATA_ROOT}/{IMG_SCALE}/val/*.tfrec')

test_files = glob.glob(f'{DATA_ROOT}/{IMG_SCALE}/test/*.tfrec')

print('Files:', \

      '\n\tTrain tfrec Count:', len(train_files), \

      '\n\tValid tfrec Count:', len(valid_files), \

      '\n\tTest  tfrec Count:', len(test_files))
def create_indexes(phase):

    tfrec_files = glob.glob(f'{DATA_ROOT}/{IMG_SCALE}/{phase}/*.tfrec')

    dirpath = os.path.dirname(tfrec_files[0].replace(DATA_ROOT, WORK_ROOT))

    if not os.path.exists(dirpath):

        os.makedirs(dirpath)

    patterns = []

    for tfrec_path in tfrec_files:  

        index_path = tfrec_path.replace(DATA_ROOT, WORK_ROOT).replace('.tfrec', '.index')

        create_index(tfrec_path, index_path)

        patterns.append(tfrec_path[len(DATA_ROOT)+1:-6])

    return patterns

        

train_patterns = create_indexes('train')

valid_patterns = create_indexes('val')

test_patterns  = create_indexes('test')
train_patterns[:5]
class TFRecordFlowersDataset(torch.utils.data.IterableDataset):

    def __init__(self, patterns, labeled=True, augtrans=None, imgtrans=None):

        super().__init__()

        self.labeled = labeled

        self.augtrans = augtrans

        self.imgtrans = imgtrans 

        self.imagecnt = 0

        self.imagepat = []

        for pattern in patterns:

            tfrec_path = DATA_ROOT + f'/{pattern}.tfrec'

            index_path = WORK_ROOT + f'/{pattern}.index'

            self.imagecnt += len(np.loadtxt(index_path, dtype=np.int64)[:, 0])

            self.imagepat.append((tfrec_path, index_path))

        print('Count:', self.imagecnt)



    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            shard = worker_info.id, worker_info.num_workers

            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)

        else:

            shard = None

            

        if self.labeled:

            description = {'image': 'byte', 'class': 'int'}

        else:

            description = {'image': 'byte', 'id': 'byte'}

            

        for tfrec_path, index_path in self.imagepat:

            it = reader.tfrecord_loader(

                tfrec_path,

                index_path,

                description,

                shard

            )

            for elem in it:

                img = Image.open(io.BytesIO(elem['image']), mode='r').convert('RGB')

                if self.augtrans:

                    img = self.augtrans(img)

                if self.imgtrans:

                    img = self.imgtrans(img)

                tag = elem['class'].item() if self.labeled else str(elem['id'], encoding='utf-8')

                yield img, tag

                

    def __len__(self):

        return self.imagecnt
# train_dataset = TFRecordFlowersDataset(train_patterns)

# valid_dataset = TFRecordFlowersDataset(valid_patterns)
# test_dataset = TFRecordFlowersDataset(test_patterns, labeled=False)
sample_dataset = TFRecordFlowersDataset(train_patterns)

sample_iter = iter(sample_dataset)

sample_data = [next(sample_iter) for _ in range(10)]

sample_dataloader = DataLoader(sample_dataset, batch_size=BATCH_SIZE, shuffle=False)

sample_data[0], len(sample_dataloader)
fig, axes = plt.subplots(nrows=2, ncols=5, sharey=True, figsize=(12,4))

for r in range(2):

    for c in range(5):

        axes[r][c].set_xticks([])

        axes[r][c].set_yticks([])

        axes[r][c].imshow(np.array(sample_data[r*2 + c][0]).astype('uint8')) # 'gray_r'
def draw_image(imgdata, labelname, augtrans=None):

    img = imgdata.copy()

    if augtrans is not None:

        img = augtrans(img)

        

    font_obj = ImageFont.truetype(FONT_PATH, 16)

    draw_img = ImageDraw.Draw(img)

    draw_img.text((0, 0), labelname, font=font_obj, fill=(255, 255, 255))

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

        imgdata=img,

        labelname=CLASS_NAMES[labelid],

    ) for img, labelid in sample_data

]



plt.xticks([])

plt.yticks([])

plt.imshow(grid_image(images_2x5, cols=5));
aug_trans = RandomOrder([

    RandomRotation(degrees=30),

    RandomVerticalFlip(p=0.3),

    RandomHorizontalFlip(p=0.3),

    ColorJitter(brightness=0.55, contrast=0.3, saturation=0.25, hue=0),

])



img_trans = Compose([

    RandomResizedCrop((INPUT_SIZE, INPUT_SIZE)),

    ToTensor(),

    Normalize(mean=DATASET_MEAN, std=DATASET_STD),

])
plt.figure(figsize=(24, 12))



augment_images_2x5 = [

    draw_image(

        imgdata=img,

        labelname=CLASS_NAMES[labelid],

        augtrans = aug_trans

    ) for img, labelid in sample_data

]



plt.xticks([])

plt.yticks([])

plt.imshow(grid_image(augment_images_2x5, cols=5));
del sample_dataloader

del sample_dataset

del sample_data
backbone = torchvision.models.vgg16(pretrained=True)
# backbone
# summary(backbone, (3, INPUT_SIZE, INPUT_SIZE), device='cpu')
# layer_index = 0

for param in backbone.features.parameters():

    # if layer_index > 18:

    #     break

    # layer_index += 1

    param.requires_grad = False
METRICS = {

    'epoch':[0],

    'train_loss':[0],

    'train_acc':[0],

    'val_acc':[0],

    'val_loss':[0],

    'lr': [0],

}



def log_last_metric():

    print('{}: train_loss[{}], train_acc[{}], val_loss[{}], val_acc[{}], lr{}'.format(

        METRICS['epoch'][-1] + 1,

        round(METRICS['train_loss'][-1], 3),

        round(METRICS['train_acc'][-1], 3),

        round(METRICS['val_loss'][-1], 3),

        round(METRICS['val_acc'][-1], 3), METRICS['lr']

    ))
class ClassifierNet(pl.LightningModule):

    def __init__(self, extractor=None, num_classes=NUM_CLASSES):

        super().__init__()

        if extractor is not None:

            self.features = extractor

            self.classifier = nn.Sequential(

                nn.AdaptiveAvgPool2d(output_size=(7, 7)),

                nn.Flatten(start_dim=1, end_dim=-1),

                nn.Linear(in_features=25088, out_features=2048, bias=True),

                nn.ReLU(inplace=True),

                nn.Dropout(p=0.5, inplace=False),

                nn.Linear(in_features=2048, out_features=1024, bias=True),

                nn.ReLU(inplace=True),

                nn.Dropout(p=0.5, inplace=False),

                nn.Linear(in_features=1024, out_features=256, bias=True),

                nn.ReLU(inplace=True),

                # nn.Dropout(p=0.5, inplace=False),

                nn.Linear(in_features=256, out_features=num_classes, bias=True)

            )

        else:    

            self.features = nn.Sequential(

                nn.BatchNorm2d(num_features=3, momentum=0.1),

                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1, bias=True),

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(num_features=64, momentum=0.1, affine=True, track_running_stats=True),

                nn.MaxPool2d(kernel_size=7, stride=1, padding=3, ceil_mode=False),

                nn.Dropout(inplace=True, p=0.5),

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True),

                nn.ReLU(inplace=True),

                nn.BatchNorm2d(num_features=64, momentum=0.1, affine=True, track_running_stats=True),

                nn.MaxPool2d(kernel_size=5, stride=1, padding=2, ceil_mode=False),

                nn.Dropout(inplace=True, p=0.5)

            )

            self.classifier = nn.Sequential(

               nn.AdaptiveAvgPool2d(output_size=(5, 5)),

               nn.Flatten(start_dim=1, end_dim=-1),

               nn.Linear(in_features=1600, out_features=128, bias=True),

               nn.Dropout(inplace=True, p=0.5),

               nn.Linear(in_features=128, out_features=num_classes, bias=True),

            )

  

    def forward(self, x, *args, **kwargs):

        x = self.features(x)

        x = self.classifier(x)

        return x

            

    @property

    def metrics(self):

        return self.metrics

        

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(

            model.parameters(),

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

        self.result_df = None

        self.train_dataset = TFRecordFlowersDataset(train_patterns, True, aug_trans, img_trans) 

        self.valid_dataset = TFRecordFlowersDataset(valid_patterns, True, None, img_trans) 

        self.test_dataset  = TFRecordFlowersDataset(test_patterns, False, None, img_trans)



    def train_dataloader(self):

        return DataLoader(

                self.train_dataset,

                batch_size=BATCH_SIZE,

                num_workers=NUM_WORKERS,

                drop_last=True,

                shuffle=False)

    

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

        METRICS['train_loss'].append(loss.cpu().item())

        METRICS['train_acc'].append(acc.cpu().item())

        return {'progress_bar': {'train_loss': loss, 'train_acc': acc}}



    def val_dataloader(self):

        return DataLoader(

            self.valid_dataset,

            batch_size=BATCH_SIZE,

            num_workers=NUM_WORKERS,

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

        METRICS['val_loss'].append(loss.cpu().item())

        METRICS['val_acc'].append(acc.cpu().item())

        log_last_metric() # kaggle debug

        return {'progress_bar': {'val_loss': loss, 'val_acc': acc}}

    

    def test_dataloader(self):

        return DataLoader(

            self.test_dataset,

            batch_size=BATCH_SIZE,

            num_workers=NUM_WORKERS,

            drop_last=False,

            shuffle=False)

    

    def test_step(self, batch, batch_idx):

        x, imgid = batch

        y_pred = torch.argmax(self(x), dim=1).cpu().numpy()

        log = {'imgid': imgid, 'label': y_pred}

        return log



    def test_epoch_end(self, outputs):

        imgid = np.concatenate([x['imgid'] for x in outputs])

        label = np.concatenate([x['label'] for x in outputs])

        result = {'id': imgid, 'label': label}

        self.result_df = pd.DataFrame(data=result)  # TODO submission

        return result

    

class ClassifierTrainer(pl.Trainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)



    def on_validation_start(self):

        lrs = []

        for scheduler in self.lr_schedulers:

            ss = scheduler['scheduler']

            if isinstance(ss, torch.optim.lr_scheduler.ReduceLROnPlateau):

                for i, param_group in enumerate(ss.optimizer.param_groups):

                    lrs.append(np.float32(param_group['lr']))

            else:

                lrs.extend([np.float32(x) for x in ss.get_last_lr()])

        self.add_progress_bar_metrics({'lr': lrs})

        METRICS['lr'] = lrs

        return super().on_validation_start()    

    

    # Workaround TPU on pytorch_lightning Fail (distrib_data_parallel.py)

    def transfer_distrib_spawn_state_on_fit_end(self, model, mp_queue, results):

        if self.global_rank == 0 and mp_queue is not None: 

            mp_queue.put(CKPT_PATH) # best_path

            mp_queue.put({})        # results   # Test Phase: cannot pass real results (will block)

            mp_queue.put(None)      # last_path



    def save_spawn_weights(self, model):

        if self.is_global_zero:

            super().save_checkpoint(CKPT_PATH, weights_only=True)

        

    def load_spawn_weights(self, original_model):

        return original_model
if torch.cuda.is_available():

    args = {'gpus':[0]}

else:

    args = {'tpu_cores':[1], 'precision':16}



trainer = ClassifierTrainer(

    max_epochs=MAX_EPOCHS,

    logger=False,

    log_gpu_memory='min_max',

    weights_summary='top',

    num_sanity_val_steps=0,

    progress_bar_refresh_rate=1,

    check_val_every_n_epoch=1,

    default_root_dir=WORK_ROOT,

    resume_from_checkpoint=None,

    early_stop_callback=EarlyStopping(monitor='val_loss', patience=7, mode='min'),

    checkpoint_callback=ModelCheckpoint(monitor='val_loss', period=5, mode='min'),

    **args

)



model = ClassifierNet(backbone.features, num_classes=NUM_CLASSES)
trainer.fit(model);
trainer.test(model);



result_df = model.result_df

plt.figure(figsize=(22, 10))

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
!rm -rf $KGGL_ROOT/*



result_df.to_csv(SUBMITCSV, index=False)
!ls -l 
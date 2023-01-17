# libraries

import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

import time 

from PIL import Image

train_on_gpu = True

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import accuracy_score

import cv2
!pip install albumentations > /dev/null 2>&1

!pip install pretrainedmodels > /dev/null 2>&1

!pip install kekas > /dev/null 2>&1

!pip install adabound > /dev/null 2>&1
# more imports

import albumentations

from albumentations import torch as AT

import pretrainedmodels

import adabound



from kekas import Keker, DataOwner, DataKek

from kekas.transformations import Transformer, to_torch, normalize

from kekas.metrics import accuracy

from kekas.modules import Flatten, AdaptiveConcatPool2d

from kekas.callbacks import Callback, Callbacks, DebuggerCallback

from kekas.utils import DotDict
labels = pd.read_csv('../input/train_labels.csv')

fig = plt.figure(figsize=(25, 4))

# display 20 images

train_imgs = os.listdir("../input/train")

for idx, img in enumerate(np.random.choice(train_imgs, 20)):

    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])

    im = Image.open("../input/train/" + img)

    plt.imshow(im)

    lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]

    ax.set_title(f'Label: {lab}')
test_img = os.listdir('../input/test/')

test_df = pd.DataFrame(test_img, columns=['id'])

test_df['label'] = -1

test_df['data_type'] = 'test'

test_df['id'] = test_df['id'].apply(lambda x: x.split('.')[0])



labels['label'] = labels['label'].astype(int)

labels['data_type'] = 'train'



labels.head()
labels.shape, test_df.shape
train_short, _ = train_test_split(labels, stratify=labels.label, test_size=0.95)
train_short.shape
# splitting data into train and validation

train, valid = train_test_split(train_short, stratify=train_short.label, test_size=0.1)
def reader_fn(i, row):

    image = cv2.imread(f"../input/{row['data_type']}/{row['id']}.tif")[:,:,::-1] # BGR -> RGB

    label = torch.Tensor([row["label"]])

    return {"image": image, "label": label}
def augs(p=0.5):

    return albumentations.Compose([

        albumentations.HorizontalFlip(),

        albumentations.RandomRotate90(),

        albumentations.Transpose(),

        albumentations.Flip(),

    ], p=p)
def get_transforms(dataset_key, size, p):



    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))



    AUGS = Transformer(dataset_key, lambda x: augs()(image=x)["image"])



    NRM_TFMS = transforms.Compose([

        Transformer(dataset_key, to_torch()),

        Transformer(dataset_key, normalize())

    ])

    

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])

    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])

    

    return train_tfms, val_tfms
# getting transformations

train_tfms, val_tfms = get_transforms("image", 224, 0.5)
train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)

val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)



batch_size = 32

workers = 0



train_dl = DataLoader(train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)

val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)
test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)

test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)
# create a simple neural network using pretrainedmodels library

# https://github.com/Cadene/pretrained-models.pytorch



class Net(nn.Module):

    def __init__(

            self,

            num_classes: int,

            p: float = 0.2,

            pooling_size: int = 2,

            last_conv_size: int = 81536,

            arch: str = "densenet169",

            pretrained: str = "imagenet") -> None:

        """A simple model to finetune.

        

        Args:

            num_classes: the number of target classes, the size of the last layer's output

            p: dropout probability

            pooling_size: the size of the result feature map after adaptive pooling layer

            last_conv_size: size of the flatten last backbone conv layer

            arch: the name of the architecture form pretrainedmodels

            pretrained: the mode for pretrained model from pretrainedmodels

        """

        super().__init__()

        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)

        modules = list(net.children())[:-1]  # delete last layer

        # add custom head

        modules += [nn.Sequential(

            # AdaptiveConcatPool2d is a concat of AdaptiveMaxPooling and AdaptiveAveragePooling 

            AdaptiveConcatPool2d(size=pooling_size),

            Flatten(),

            nn.BatchNorm1d(13312),

            nn.Dropout(p),

            nn.Linear(13312, num_classes)

        )]

        self.net = nn.Sequential(*modules)



    def forward(self, x):

        logits = self.net(x)

        return logits
dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)



criterion = nn.BCEWithLogitsLoss()
def step_fn(model: torch.nn.Module,

            batch: torch.Tensor) -> torch.Tensor:

    """Determine what your model will do with your data.



    Args:

        model: the pytorch module to pass input in

        batch: the batch of data from the DataLoader



    Returns:

        The models forward pass results

    """

    

    inp = batch["image"]

    return model(inp)
def bce_accuracy(target: torch.Tensor,

                 preds: torch.Tensor) -> float:

    thresh = 0.5

    target = target.cpu().detach().numpy()

    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)

    return accuracy_score(target, preds)
keker = Keker(model=model,

              dataowner=dataowner,

              criterion=criterion,

              step_fn=step_fn,

              target_key="label",

              metrics={"acc": bce_accuracy},

              opt=torch.optim.SGD,

              opt_params={"momentum": 0.99})
keker.unfreeze(model_attr="net")



# If you want to freeze till some layer:

layer_num = -1

keker.freeze_to(layer_num, model_attr="net")
keker.kek_lr(final_lr=0.001, logdir="logs")
# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip

# !unzip ngrok-stable-linux-amd64.zip

# LOG_DIR = 'logs' # Here you have to put your log directory

# get_ipython().system_raw(

#     'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'

#     .format(LOG_DIR)

# )

# get_ipython().system_raw('./ngrok http 6006 &')

# ! curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

#!tensorboard --logdir logs --host 0.0.0.0 --port 6007
keker.plot_kek_lr('logs')
# simply training the model with pre-defined parameters.

keker.kek(lr=2e-5, epochs=1, logdir='train_logs0')
# we plot all three graphs by default, but we can also choose which of them will be shown.

keker.plot_kek('train_logs0', metrics=['loss', 'acc'])
keker.kek(lr=2e-5, epochs=3,

         sched=torch.optim.lr_scheduler.StepLR,

              sched_params={"step_size":1, "gamma": 0.9},# optimizer class. if note specifiyng, 

                                                  # an SGD is using by default

              opt_params={"weight_decay": 1e-5},

              early_stop_params={

              "patience": 2,   

              "metric": "acc", 

              "mode": "min",   

              "min_delta": 0

          }, logdir='train_logs')
keker.plot_kek('train_logs')
keker.kek_one_cycle(max_lr=1e-4,                  # the maximum learning rate

                    cycle_len=3,                  # number of epochs, actually, but not exactly

                    momentum_range=(0.95, 0.85),  # range of momentum changes

                    div_factor=25,                # max_lr / min_lr

                    increase_fraction=0.3,        # the part of cycle when learning rate increases

                    logdir='train_logs1')
keker.plot_kek('train_logs1')
train, valid = train_test_split(labels, stratify=labels.label, test_size=0.1)



train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)

val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)



train_dl = DataLoader(train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)

val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)



test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)

test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)



dataowner = DataOwner(train_dl, val_dl, None)



keker = Keker(model=model,

              dataowner=dataowner,

              criterion=criterion,

              step_fn=step_fn,

              target_key="label",

              metrics={"acc": bce_accuracy},

              opt=adabound.AdaBound,

              opt_params={'final_lr': 0.001})



keker.unfreeze(model_attr="net")



layer_num = -3

keker.freeze_to(layer_num, model_attr="net")
keker.kek_one_cycle(max_lr=1e-4,

                    cycle_len=6,

                    momentum_range=(0.95, 0.85),

                    div_factor=25,

                    increase_fraction=0.3)
# keker.kek(lr=2e-5, epochs=3,

#          sched=ReduceLROnPlateau,

#               sched_params={"patience":2, "factor": 0.1},# optimizer class. if note specifiyng, 

#                                                   # an SGD is using by default

#               early_stop_params={

#               "patience": 2,   

#               "metric": "acc", 

#               "mode": "min",   

#               "min_delta": 0

#           }, logdir='train_logs_fin')
keker.plot_kek_lr('train_logs_fin')
# simple prediction

preds = keker.predict_loader(loader=test_dl)
# TTA

flip_ = albumentations.Flip(always_apply=True)

transpose_ = albumentations.Transpose(always_apply=True)



def insert_aug(aug, dataset_key="image", size=224):    

    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))

    

    AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])

    

    NRM_TFMS = transforms.Compose([

        Transformer(dataset_key, to_torch()),

        Transformer(dataset_key, normalize())

    ])

    

    tfm = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])

    return tfm



flip = insert_aug(flip_)

transpose = insert_aug(transpose_)



tta_tfms = {"flip": flip, "transpose": transpose}



# third, run TTA

keker.TTA(loader=test_dl,                # loader to predict on 

          tfms=tta_tfms,                # list or dict of always applying transforms

          savedir="tta_preds1",  # savedir

          prefix="preds")               # (optional) name prefix. default is 'preds'



# it will saves predicts for each augmentation to savedir with name

#  - {prefix}_{name_from_dict}.npy if tfms is a dict

#  - {prefix}_{index}.npy          if tfms is a list
prediction = np.zeros((test_df.shape[0], 1))

for i in os.listdir('tta_preds1'):

    pr = np.load('tta_preds1/' + i)

    prediction += pr

prediction = prediction / len(os.listdir('tta_preds1'))
test_preds = pd.DataFrame({'imgs': test_df.id.values, 'preds': prediction.reshape(-1,)})

test_preds.columns = ['id', 'label']

test_preds.to_csv('sub.csv', index=False)
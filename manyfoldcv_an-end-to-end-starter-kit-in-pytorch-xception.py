# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
! pip install pretrainedmodels
import os
import time
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pretrainedmodels import xception

import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import warnings

warnings.filterwarnings("ignore")
IMG_SIZE = 128
NUM_CHANNEL = 3

PATH_TO_IMAGES = '../input/train/'
PATH_TO_TEST_IMAGES = '../input/test/'
PATH_TO_META = '../input/train.csv'
SAMPLE_SUBMIT = '../input/sample_submission.csv'

SEED = 666
DEVICE = 'cuda'

LABEL_MAP = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"}


# Originally from https://www.kaggle.com/crimzoid/4-channel-resnet-from-scratch-pytorch-lb-0-16
# Modified by Ji Yang

image_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomChoice([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine((0, 0)),
        transforms.RandomAffine((90, 90)),
        transforms.RandomAffine((180, 180)),
        transforms.RandomAffine((270, 270)),
    ]),
    transforms.ToTensor(),
])

sub_image_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),

])


class MultiBandMultiLabelDataset(Dataset):
    # BANDS_NAMES = ['_red.png', '_green.png', '_blue.png']
    BANDS_NAMES = ['_red.png', '_green.png', '_blue.png', '_yellow.png']

    def __len__(self):
        return len(self.images_df)

    def __init__(self, images_df, base_path, image_transform, train_mode=True):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)

        self.images_df = images_df.copy()
        self.image_transform = image_transform
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))
        self.train_mode = train_mode

    def __getitem__(self, index):
        y = None
        X = self._load_multiband_image(index)
        if self.train_mode:
            y = self._load_multilabel_target(index)

        X = self.image_transform(X)

        return X, y

    def _load_multiband_image(self, index):
        row = self.images_df.iloc[index]
        image_bands = []
        for band_name in self.BANDS_NAMES:
            p = str(row.Id.absolute()) + band_name
            img = PIL.Image.open(p)
            image_bands.append(img)

        # lets pretend its a RBGA image to support 4 channels
        band4image = PIL.Image.merge('RGBA', bands=image_bands)

        return band4image

    def _load_multilabel_target(self, index):
        return list(map(int, self.images_df.iloc[index].Target.split(' ')))

    def collate_func(self, batch):
        labels = None
        images = [x[0] for x in batch]

        if self.train_mode:
            labels = [x[1] for x in batch]
            labels_one_hot = self.mlb.fit_transform(labels)
            labels = torch.FloatTensor(labels_one_hot)

        return torch.stack(images)[:, :4, :, :], labels


df = pd.read_csv(PATH_TO_META)
df_train, df_test = train_test_split(df, test_size=.2, random_state=SEED)
df_submission = pd.read_csv(SAMPLE_SUBMIT)
debugging = True
if debugging:
    EPOCHS = 3
    BATCH_SIZE = 24
    BASE_LR = 0.001

if debugging:
    df_train = df_train[:300]
    df_test = df_test[:50]
    df_submission = df_submission

gtrain = MultiBandMultiLabelDataset(df_train, base_path=PATH_TO_IMAGES, image_transform=image_transform)
train_load = DataLoader(gtrain, collate_fn=gtrain.collate_func, batch_size=BATCH_SIZE, num_workers=2)
# You may want to use this to find the best value of number of workers...
for i in range(1, 9):
    train_load = DataLoader(gtrain, collate_fn=gtrain.collate_func, batch_size=BATCH_SIZE, num_workers=i)
    start = time.time()
    for batch_idx, batch in enumerate(train_load):
        if batch_idx == 50:
            break
    print('# of workers: {} | {:4f}'.format(i, (time.time() - start) / 20))

for batch_idx, batch in enumerate(train_load):
    break
images, labels = batch

fig, ax = plt.subplots(3, 4, figsize=(15, 9))
for r in range(3):
    for c in range(4):
        ax[r][c].imshow(images[r * 4 + c].permute(1, 2, 0))
print('min: {0}, max: {1}'.format(images.min(), images.max()))
class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


class Xception(nn.Module):
    def __init__(self, num_classes=28, pretrained='imagenet', dropout_rate=0.):
        super(Xception, self).__init__()
        self.pretrained_model = xception(pretrained=pretrained)
        self.process_input = nn.Sequential(
            ConvBnRelu2d(4, 16),
            ConvBnRelu2d(16, 3),
        )
        self.fc_input = 2048
        self.activation = nn.ReLU()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.last_linear = nn.Linear(self.fc_input, num_classes)

    def forward(self, x):
        x = self.process_input(x)
        x = self.pretrained_model.features(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x
def f1_loss(target, output, epsilon=1e-7):
    y_pred = nn.Sigmoid()(output)
    y_true = target.double()

    TP = (y_pred * y_true).sum(1)
    prec = TP / (y_pred.sum(1) + epsilon)
    rec = TP / (y_true.sum(1) + epsilon)
    res = 2 * prec * rec / (prec + rec + epsilon)

    f1 = res
    f1 = f1.clamp(min=0)
    return 1 - f1.mean()


from sklearn.metrics import f1_score

sk_f1_score = f1_score


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def init(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return str(self.avg)

EPOCHS, BATCH_SIZE, BASE_LR, DEVICE
model = Xception(num_classes=28, pretrained=None).to(DEVICE)


gtrain = MultiBandMultiLabelDataset(df_train, base_path=PATH_TO_IMAGES, image_transform=image_transform)
gtest = MultiBandMultiLabelDataset(df_test, base_path=PATH_TO_IMAGES, image_transform=sub_image_transform)

train_load = DataLoader(gtrain, collate_fn=gtrain.collate_func, batch_size=BATCH_SIZE, num_workers=6)
test_load = DataLoader(gtest, collate_fn=gtest.collate_func, batch_size=BATCH_SIZE, num_workers=6)


lr = BASE_LR
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

patience = 0
best_validation = 0
record = []
val_record = []
lrs = []

for epoch in range(1, EPOCHS):
    if epoch % 2 == 0:
        lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('Epoch {} with starting LR {}\n'.format(epoch, lr))
    lrs.append(lr)
    epoch_loss = AverageMeter()
    epoch_scr = AverageMeter()
    epoch_skl = AverageMeter()
    start = time.time()
    for batch_idx, batch in enumerate(train_load):
        data_, target = batch
        data = data_.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        loss = f1_loss(target.double(), output.double())
        # f1 = f1_score(target, output)
        f1 = sk_f1_score(target.cpu().data.numpy(), (output > 0.5).cpu().data.numpy(), average='macro')
        # sk_f1 = sk_f1_score(target.cpu().data.numpy(), (output > 0.5).cpu().data.numpy(), average='macro')
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.update(loss.item())
        epoch_scr.update(f1)
    print('Epoch {:3} Trn --> Loss: {:.5f}, F1: {:.5f}'.format(epoch, epoch_loss.avg, epoch_scr.avg), end=' | ')
    record.append((epoch_loss, epoch_scr))

    with torch.no_grad():
        val_epoch_loss = AverageMeter()
        val_epoch_scr = AverageMeter()
        for batch_idx, batch in enumerate(test_load):
            data_, target = batch
            data = data_.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            loss = f1_loss(target.double(), output.double())
            f1 = sk_f1_score(target.cpu().data.numpy(), (output > 0.5).cpu().data.numpy(), average='macro')
            val_epoch_loss.update(loss.item())
            val_epoch_scr.update(f1)
        print('Epoch {:3} Val --> Loss: {:.5f}, F-1: {:.5f}'.format(epoch, val_epoch_loss.avg, val_epoch_scr.avg),
              end=' | \n')
        val_record.append((val_epoch_loss, val_epoch_scr))
with torch.no_grad():
    for thres in range(1, 10):
        thres /= 10
        val_epoch_loss = AverageMeter()
        val_epoch_scr = AverageMeter()
        for batch_idx, batch in enumerate(test_load):
            data_, target = batch
            data = data_.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            loss = f1_loss(target.double(), output.double())
            f1 = sk_f1_score(target.cpu().data.numpy(), (output > 0.5).cpu().data.numpy(), average='macro')
            val_epoch_loss.update(loss.item())
            val_epoch_scr.update(f1)
        print('Threshold: {} | Val --> Loss: {:.5f}, F-1: {:.5f}'.format(thres, val_epoch_loss.avg,
                                                                         val_epoch_scr.avg), end=' | \n')
gsub = MultiBandMultiLabelDataset(df_submission, base_path=PATH_TO_TEST_IMAGES, train_mode=False,
                                  image_transform=sub_image_transform)
submission_load = DataLoader(gsub, collate_fn=gsub.collate_func, batch_size=BATCH_SIZE, num_workers=2)

result = []
with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(submission_load)):
        data_, _ = batch
        data = data_.to(DEVICE)
        output = model(data)
        result.append(output.sigmoid().cpu().data.numpy())

result = np.concatenate(result)
result.shape

def make_submission_file(sample_submission_df, predictions):
    submissions = []
    for row in predictions:
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)

    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('submission.csv', index=None)

    return sample_submission_df

# prepare the submission file and
THRESHOLD = 0.4
p = result > THRESHOLD

submission_file = make_submission_file(sample_submission_df=df_submission,
                                       predictions=p)

submission_file = submission_file.sample(frac=1).reset_index(drop=True)
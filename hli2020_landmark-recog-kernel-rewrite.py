# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.getcwd())

print('number {:d}'.format(4))



# os.walk('/kaggle/input/Fashion MNIST') 有空格 错误的文件路径写法

for dirname, _, filenames in os.walk('/kaggle/input/fashionmnist'):

# for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#!/usr/bin/env python

import multiprocessing

import os

import time



import numpy as np

import pandas as pd



from typing import Any, Optional, Tuple



import torch

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn



from torch.utils.data import TensorDataset, DataLoader, Dataset

from sklearn.preprocessing import LabelEncoder

from PIL import Image

from tqdm import tqdm
IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

# 每一个类别的样本数目至少是50

MIN_SAMPLES_PER_CLASS = 50

BATCH_SIZE = 512

LEARNING_RATE = 1e-3

LR_STEP = 1          # epoch

LR_FACTOR = 0.5

NUM_WORKERS = multiprocessing.cpu_count()

# 需要大家改

# MAX_STEPS_PER_EPOCH = 15000

MAX_STEPS_PER_EPOCH = 15 

NUM_EPOCHS = 2

# 每几个iteration显示loss

LOG_FREQ = 5

# 这是什么意思？？？？

NUM_TOP_PREDICTS = 20
class LandmarkDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, mode):

        print(f'creating data loader - {mode}')

        assert mode in ['train', 'val', 'test']

        

        self.df = dataframe   # 从 ‘train.csv’文件 读好的数据

        self.mode = mode

        transforms_list = []



        if self.mode == 'train':

            transforms_list = [

                transforms.RandomHorizontalFlip(),

                transforms.RandomChoice([

                    transforms.RandomResizedCrop(64),

                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),

                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),

                                            scale=(0.8, 1.2), shear=15,

                                            resample=Image.BILINEAR)

                ])

            ]



        transforms_list.extend([

            transforms.ToTensor(),

            # 这些数值 可能是landmark_Recognition 上根据image 算出来的 / 也可能是iamgenet数据

            transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                  std=[0.229, 0.224, 0.225]),

        ])

        self.transforms = transforms.Compose(transforms_list)

    

    def __getitem__(self, index):

        ''' Returns: tuple (sample, target) '''

        # df 是dataframe, csv文件， train.csv, 是由数据库本身提供的。

        filename = self.df.id.values[index]   # 45465656.jpg



        part = 1 if self.mode == 'test' or filename[0] in '01234567' else 2

        directory = 'test' if self.mode == 'test' else 'train_' + filename[0]

        sample = Image.open(f'../input/google-landmarks-2019-64x64-part{part}/{directory}/{self.mode}_64/{filename}.jpg')

        assert sample.mode == 'RGB'



        image = self.transforms(sample)



        if self.mode == 'test':

            return image

        elif self.mode == 'train':

            # image and target

            return image, self.df.landmark_id.values[index]



    def __len__(self):

        return self.df.shape[0]
def load_data():

    # 这个我也不知道 就抄过来吧。

    torch.multiprocessing.set_sharing_strategy('file_system')

    cudnn.benchmark = True



    # only use classes which have at least MIN_SAMPLES_PER_CLASS samples

    print('loading data...')

    df = pd.read_csv('../input/google-landmarks-2019-64x64-part1/train.csv')

    df.drop(columns='url', inplace=True)



    counts = df.landmark_id.value_counts()

    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index

    num_classes = selected_classes.shape[0]

    print('# of classes with at least {:d} samples: {:d}'.format(MIN_SAMPLES_PER_CLASS, num_classes))



    train_df = df.loc[df.landmark_id.isin(selected_classes)].copy()

    print('train_df', train_df.shape)



    test_df = pd.read_csv('../input/google-landmarks-2019-64x64-part1/test.csv', dtype=str)

    test_df.drop(columns='url', inplace=True)

    print('test_df', test_df.shape)



    # filter non-existing test images

    exists = lambda img: os.path.exists(f'../input/google-landmarks-2019-64x64-part1/test/test_64/{img}.jpg')

    test_df = test_df.loc[test_df.id.apply(exists)].copy()

    print('test_df after filtering', test_df.shape)

    assert test_df.shape[0] > 112000



    label_encoder = LabelEncoder()

    label_encoder.fit(train_df.landmark_id.values)

    print('found classes', len(label_encoder.classes_))

    assert len(label_encoder.classes_) == num_classes



    train_df.landmark_id = label_encoder.transform(train_df.landmark_id)



    # 下面这些都是标准pytorch语法

    train_dataset = LandmarkDataset(train_df, mode='train')

    test_dataset = LandmarkDataset(test_df, mode='test')



    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,

                              shuffle=True, num_workers=NUM_WORKERS, drop_last=True)



    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,

                             shuffle=False, num_workers=NUM_WORKERS)



    return train_loader, test_loader, label_encoder, num_classes
class AverageMeter:

    ''' Computes and stores the average and current value '''

    def __init__(self) -> None:

        self.reset()



    def reset(self) -> None:

        self.val = 0.0

        self.avg = 0.0

        self.sum = 0.0

        self.count = 0



    def update(self, val: float, n: int = 1) -> None:

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:

    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''

    assert len(predicts.shape) == 1

    assert len(confs.shape) == 1

    assert len(targets.shape) == 1

    assert predicts.shape == confs.shape and confs.shape == targets.shape



    _, indices = torch.sort(confs, descending=True)



    confs = confs.cpu().numpy()

    predicts = predicts[indices].cpu().numpy()

    targets = targets[indices].cpu().numpy()



    res, true_pos = 0.0, 0



    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):

        rel = int(p == t)

        true_pos += rel



        res += true_pos / (i + 1) * rel



    res /= targets.shape[0] 

    return res

def train(train_loader, model, criterion, optimizer,

          epoch, lr_scheduler):

    print(f'epoch {epoch}')

    batch_time = AverageMeter()

    losses = AverageMeter()

    avg_score = AverageMeter()



    model.train()

    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)  # 是 20

    print(f'total batches: {num_steps}')



    end = time.time()

    lr_str = ''



    for i, (input_, target) in enumerate(train_loader):

        if i >= num_steps:

            break



        output = model(input_.cuda())

        loss = criterion(output, target.cuda())



        confs, predicts = torch.max(output.detach(), dim=1)

        avg_score.update(GAP(predicts, confs, target))



        losses.update(loss.data.item(), input_.size(0))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        batch_time.update(time.time() - end)

        end = time.time()



        if i % LOG_FREQ == 0:

            print(f'{epoch} [{i}/{num_steps}]\t'

                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'

                        f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'

                        + lr_str)

            

    print(f' * average GAP(accuray/error) on train {avg_score.avg:.4f}')
def inference(data_loader: Any, model: Any) -> Tuple[torch.Tensor, torch.Tensor,

                                                     Optional[torch.Tensor]]:

    ''' Returns predictions and targets, if any. '''

    model.eval()

    activation = nn.Softmax(dim=1)

    all_predicts, all_confs, all_targets = [], [], []



    with torch.no_grad():

        for i, data in enumerate(tqdm(data_loader, disable=IN_KERNEL)):

            if data_loader.dataset.mode != 'test':

                input_, target = data

            else:

                input_, target = data, None



            output = model(input_.cuda())

            output = activation(output)



            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)

            all_confs.append(confs)

            all_predicts.append(predicts)



            if target is not None:

                all_targets.append(target)



    predicts = torch.cat(all_predicts)

    confs = torch.cat(all_confs)

    targets = torch.cat(all_targets) if len(all_targets) else None



    return predicts, confs, targets
def generate_submission(test_loader: Any, model: Any, label_encoder: Any) -> np.ndarray:

    

    # 坑："-" 中间横线，不是下划线！！！

    sample_sub = pd.read_csv('../input/julyonline-landmark/haha_sample_submission.csv')



    predicts_gpu, confs_gpu, _ = inference(test_loader, model)

    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()



    labels = [label_encoder.inverse_transform(pred) for pred in predicts]

    print('labels')

    print(np.array(labels))

    print('confs')

    print(np.array(confs))



    sub = test_loader.dataset.df

    def concat(label: np.ndarray, conf: np.ndarray) -> str:

        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])

    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]



    sample_sub = sample_sub.set_index('id')

    sub = sub.set_index('id')

    sample_sub.update(sub)



    sample_sub.to_csv('july_today_Dec_3_NEW.csv')
# 数据

train_loader, test_loader, label_encoder, num_classes = load_data()



model = torchvision.models.resnet50(pretrained=False)

# 要事先浏览resnet-50 每一层的名字，才知道 有 "avg_pool"

model.avg_pool = nn.AdaptiveAvgPool2d(1)

model.fc = nn.Linear(model.fc.in_features, num_classes)  # 1.8w 类的个数

model.cuda()   # 送到GPU中



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP,

                                               gamma=LR_FACTOR)



for epoch in range(1, NUM_EPOCHS + 1):

    print('-' * 50)

    train(train_loader, model, criterion, optimizer, epoch, lr_scheduler)

    lr_scheduler.step()



print('inference mode')

generate_submission(test_loader, model, label_encoder)

print('done')
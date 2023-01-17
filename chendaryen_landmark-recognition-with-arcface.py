import os



os.system('pip install ../input/landmark-recognition-2020-packages/efficientnet_pytorch-0.7.0-py3-none-any.whl')



import numpy as np

import torch

import torch.nn as nn

import torch.utils.data as Data

import torch.nn.functional as F

import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet

import pandas as pd

from PIL import Image

from sklearn.preprocessing import LabelEncoder

import multiprocessing

import time

import math

import matplotlib.pyplot as plt

import itertools

import matplotlib.pyplot as plt



import warnings



warnings.filterwarnings("ignore")
# dir

SELF_DIR = '../input/landmark-recognition-2020-self/'

ORIGIN_TRAIN_CSV = '../input/landmark-recognition-2020/train.csv'

TRAIN_CSV = '../input/landmark-recognition-2020-self/train.csv'

VALID_CSV = '../input/landmark-recognition-2020-self/valid.csv'

TEST_CSV = '../input/landmark-recognition-2020/sample_submission.csv'

TRAIN_DIR = '../input/landmark-recognition-2020/train/'

TEST_DIR = '../input/landmark-recognition-2020/test/'



# general

USE_CUDA = torch.cuda.is_available()

CPU_NUM = multiprocessing.cpu_count()

LOG_STEPS = 500

CLASSES_NUM = 10000

EPS = 1e-6

EPS_LOG = 1e-40



# training

EPOCHS = 18

BATCH_SIZE = 64

INPUT_SIZE = 288



# model

COMPOUND_COEF = 1

FEATURES_NUM = 2048

EMBEDDING_SIZE = 512

DROPOUT_RATE = 0.2



# criterion

S = 30

M = 0.3

GAMMA_FOCAL = 6



# optimizer

WEIGHT_DECAY = 5e-4

MOMENTUM = 0.9

LR_ENCODER = 1e-4

LR_ARC = 5e-4



# lr_scheduler

T_0 = 20

T_MULT = 2
class Dataset(Data.Dataset):

    def __init__(self, is_train, dataframe, data_dir):

        self.is_train = is_train

        self.dataframe = dataframe

        self.data_dir = data_dir



        self.backup = None



        if self.is_train:

            transforms_list = [

                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),

                transforms.RandomApply([transforms.RandomResizedCrop(size=INPUT_SIZE)], p=0.33),

                transforms.RandomChoice([

                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),

                    transforms.RandomAffine(

                        degrees=10, translate=(0.2, 0.2),

                        scale=(0.8, 1.2),

                        resample=Image.BILINEAR)

                ]),

                transforms.ToTensor(),

                transforms.RandomApply([transforms.RandomErasing(p=1, scale=(0.2, 0.33), ratio=(0.5, 2))], p=0.8),

                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))

            ]

        else:

            transforms_list = [

                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),

                transforms.ToTensor(),

                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))

            ]

        self.transforms = transforms.Compose(transforms_list)



    def __getitem__(self, index):

        image_id = self.dataframe.iloc[index].id

        image_path = os.path.join(self.data_dir, image_id[0], image_id[1], image_id[2], '{}.jpg'.format(image_id))

        image = Image.open(image_path)

        image = self.transforms(image)



        if self.is_train:

            return [image, self.dataframe.iloc[index].landmark_id]

        else:

            return [image_id, image]



    def __len__(self):

        return self.dataframe.shape[0]



    

def split_and_save_data(data_csv, valid_ratio):

    data = pd.read_csv(data_csv)



    counts = data.landmark_id.value_counts()

    selected_classes = counts[:CLASSES_NUM].index



    data = data.loc[data.landmark_id.isin(selected_classes)]



    valid = data.sample(frac=valid_ratio, replace=False, random_state=1)

    train = data.loc[~data.id.isin(valid.id)]



    valid.to_csv('valid.csv')

    train.to_csv('train.csv')





def load_data(train_csv, valid_csv, dir):

    train = pd.read_csv(train_csv)

    valid = pd.read_csv(valid_csv)



    label_encoder = LabelEncoder()

    label_encoder.fit(np.hstack((train.landmark_id.values, valid.landmark_id.values)))



    train.landmark_id = label_encoder.transform(train.landmark_id)

    valid.landmark_id = label_encoder.transform(valid.landmark_id)



    train_loader = Data.DataLoader(

        dataset=Dataset(True, train, dir),

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=4,

    )

    valid_loader = Data.DataLoader(

        dataset=Dataset(True, valid, dir),

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=4,

    )



    return train_loader, valid_loader
class SwishFunc(torch.autograd.Function):

    @staticmethod

    def forward(ctx, x, beta):

        y = x * torch.sigmoid(beta * x)

        ctx.save_for_backward(x, y, beta)



        return y



    @staticmethod

    def backward(ctx, grad_output):

        x, y, beta = ctx.saved_tensors



        grad_x = grad_output * (beta * y + torch.sigmoid(beta * x) * (1 - beta * y))

        grad_beta = grad_output * (x * y - y ** 2)



        return grad_x, grad_beta



    

class Swish(nn.Module):

    def __init__(self):

        super(Swish, self).__init__()

        self.beta = nn.Parameter(torch.FloatTensor([1]))  # beta is initialized to be 1



    def forward(self, x):

        return SwishFunc.apply(x, self.beta)
class SeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(SeparableConv, self).__init__()

        self.pointwise = nn.Conv2d(

            in_channels=in_channels,

            out_channels=out_channels,

            kernel_size=1,

        )

        self.swish1 = Swish()

        self.depthwise = nn.Conv2d(

            in_channels=out_channels,

            out_channels=out_channels,

            kernel_size=3,

            padding=1,

            groups=out_channels,

            bias=False,

        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.99)

        self.swish2 = Swish()



    def forward(self, inputs):

        x = self.pointwise(inputs)

        x = self.swish1(x)

        x = self.depthwise(x)

        x = self.swish2(self.bn(x))



        return x





class SAM(nn.Module):

    def __init__(self):

        super(SAM, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        self.sigmoid = nn.Sigmoid()



    def forward(self, features):

        avg_descriptor = torch.mean(features, dim=1, keepdim=True)

        max_descriptor, _ = torch.max(features, dim=1, keepdim=True)



        descriptor = torch.cat([avg_descriptor, max_descriptor], dim=1)



        attention_map = self.sigmoid(self.conv(descriptor))



        del avg_descriptor, max_descriptor, descriptor



        return features * attention_map
class ArcFace(nn.Module):

    def __init__(self, embedding_size, class_num, s=64.0, m=0.50):

        super().__init__()

        self.in_features = embedding_size

        self.out_features = class_num

        self.s = s

        self.m = m  # the angular penalty

        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))

        nn.init.xavier_uniform_(self.weight)



        self.cos_m = math.cos(m)

        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)

        self.mm = math.sin(m) * m



    def forward(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # cos(theta)

        bs = len(label)

        label_cosine = cosine[range(bs), label]

        label_sine = ((1.0 - label_cosine.pow(2)).clamp(0, 1)).sqrt()

        phi = label_cosine * self.cos_m - label_sine * self.sin_m  # cos(theta+m)



        # if theta+m > pi use penalty of CosFace cos(theta) - self.mm

        phi = torch.where(label_cosine > self.th, phi, label_cosine - self.mm)



        output = cosine * 1.0  # make backward work

        output[range(bs), label] = phi

        

        return output * self.s

    

    def valid(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # cos(theta)

        bs = len(label)

        label_cosine = cosine[range(bs), label]

        label_sine = ((1.0 - label_cosine.pow(2)).clamp(0, 1)).sqrt()

        phi = label_cosine * self.cos_m - label_sine * self.sin_m  # cos(theta+m)



        # if theta+m > pi use penalty of CosFace cos(theta) - self.mm

        phi = torch.where(label_cosine > self.th, phi, label_cosine - self.mm)



        output = cosine * 1.0  # make backward work

        output[range(bs), label] = phi

        

        return cosine, output * self.s



    def inference(self, input):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))



        return cosine
class Extractor(nn.Module):

    def __init__(self, in_channels, out_features):

        super(Extractor, self).__init__()



        mid_channels = int(in_channels * math.pow(out_features / in_channels, 0.5))



        self.sep1 = SeparableConv(in_channels, mid_channels)

        self.sep2 = SeparableConv(mid_channels, out_features)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)



    def forward(self, input):

        x = self.sep1(input)

        x = self.sep2(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, start_dim=1)



        return x



    

class Encoder(nn.Module):

    def __init__(self, compound_coef, out_features):

        super(Encoder, self).__init__()

        self.compound_coef = compound_coef



        self.base = EfficientNet.from_name('efficientnet-b{}'.format(self.compound_coef))

        self.sam1 = SAM()

        self.sam2 = SAM()

        self.sam3 = SAM()



        features_num = self.base._conv_head.in_channels

        self.extractor = Extractor(features_num, out_features)



    def forward(self, input):

        # Stem

        x = self.base._swish(self.base._bn0(self.base._conv_stem(input)))



        # Blocks

        x = self.sam1(x)

        for idx, block in enumerate(self.base._blocks):

            drop_connect_rate = self.base._global_params.drop_connect_rate

            if drop_connect_rate:

                drop_connect_rate *= float(idx) / len(self.base._blocks)  # scale drop connect_rate

            x = block(x, drop_connect_rate=drop_connect_rate)



        # Head

        x = self.extractor(self.sam2(x))



        return x



    

class EncoderHead(nn.Module):

    def __init__(self, in_features, out_features):

        super(EncoderHead, self).__init__()

        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.fc = nn.Linear(in_features, out_features)

        self.bn = nn.BatchNorm1d(out_features)

        

    

    def forward(self, input):

        return self.bn(self.fc(self.dropout(input)))
class FocalLoss(nn.Module):

    def __init__(self, gamma):

        super(FocalLoss, self).__init__()

        self.gamma = gamma



        self.softmax = nn.Softmax(dim=1)



    def forward(self, output, label):

        pred = self.softmax(output)

        y_ = pred[range(len(label)), label]

        loss = -torch.mean(torch.pow(1 - y_, self.gamma) * torch.log(y_ + EPS_LOG))

        

        return loss
class AccCounter:

    def __init__(self):

        super(AccCounter, self).__init__()

        self.total = .0

        self.correction = .0



    def update(self, logits, label):

        _, pred = torch.max(logits, dim=1)



        self.total += len(label)

        self.correction += torch.sum(pred == label).data.item()



    def show(self):

        return self.correction / self.total





def train(start_epoch, max_epoch,

          encoder, encoder_head, arc, criterion,

          optim_encoder, optim_arc, scheduler_encoder, scheduler_arc,

          train_loader, loss_list, valid_loader=None, valid_n_epochs=1):

    encoder.train()

    encoder_head.train()

    arc.train()



    for epoch_index in range(start_epoch, max_epoch):

        print('-----{} epoch-----'.format(epoch_index))



        losses = 0

        start_time = time.time()



        for batch_index, (input, label) in enumerate(train_loader):

            if USE_CUDA:

                input = input.cuda()

                label = label.cuda()



            features = encoder_head(encoder(input))

            logits = arc(features, label)



            loss = criterion(logits, label)

            loss_list.append(loss.data.item())

            losses += loss.data.item()



            optim_encoder.zero_grad()

            optim_arc.zero_grad()

            loss.backward()

            optim_encoder.step()

            optim_arc.step()

            scheduler_encoder.step()

            scheduler_arc.step()



            if batch_index % LOG_STEPS == LOG_STEPS - 1:

                print('{} / {} | '.format(batch_index + 1, len(train_loader)),

                      'time {:.3f} | '.format((time.time() - start_time) / (batch_index + 1)),

                      'avg loss {:.5f} | '.format(losses / LOG_STEPS),

                      'lr {:.7f}'.format(optim_arc.param_groups[0]['lr'])

                     )



                losses = 0



        state = {

            'encoder': encoder.state_dict(),

            'encoder_head': encoder_head.state_dict(),

            'arc': arc.state_dict(),

            'optim_encoder': optim_encoder.state_dict(),

            'optim_arc': optim_arc.state_dict(),

            'scheduler_encoder': scheduler_encoder.state_dict(),

            'scheduler_arc': scheduler_arc.state_dict(),

            'epoch': epoch_index,

        }

        torch.save(state, 'epoch_arc{}.csv'.format(epoch_index))



        loss_list_np = np.array(loss_list)

        np.save('loss_list_arc.npy', loss_list_np)



        if epoch_index % valid_n_epochs == valid_n_epochs - 1 and valid_loader is not None:

            valid(encoder, encoder_head, arc, criterion, valid_loader)



    draw_loss_trend(loss_list)





def valid(encoder, encoder_head, arc, criterion, valid_loader):

    print('validation start')

    start_time = time.time()



    encoder.eval()

    encoder_head.eval()

    arc.eval()



    losses = 0

    acc_counter = AccCounter()



    with torch.no_grad():

        for _, batch in enumerate(valid_loader):

            input, label = batch

            if USE_CUDA:

                input = input.cuda()

                label = label.cuda()



            features = encoder_head(encoder(input))

            logits_inf, logits = arc.valid(features, label)

            

            loss = criterion(logits, label)

            losses += loss.data.item()

            acc_counter.update(logits_inf, label)



    avg_loss = losses / len(valid_loader)



    encoder.train()

    encoder_head.train()

    arc.train()



    print('validation complete, avg loss {:.5f}, accuracy {:.5f}, cost {:.3f} seconds'

          .format(avg_loss, acc_counter.show(), time.time() - start_time))

    

    

def draw_loss_trend(loss_list):

    interval = 1000

    

    loss_list_ = [np.mean(loss_list[i * interval : (i + 1) * interval]) for i in range(int(len(loss_list) / interval))]

    

    x = np.arange(1, len(loss_list_) + 1) * interval



    plt.title('ArcFace loss trend')

    plt.xlabel('iterations')

    plt.ylabel('loss')

    plt.plot(x, loss_list_)

    plt.savefig('loss_iter{}.jpg'.format(len(loss_list_) * interval))

    plt.clf()
def seed_everything(seed=2020):

    np.random.seed(seed)

    torch.manual_seed(seed)
seed_everything(int(time.time()))



checkpoint_file = 'epoch_arc15.csv'

if USE_CUDA:

    checkpoint = torch.load(os.path.join(SELF_DIR, checkpoint_file))

else:

    checkpoint = torch.load(os.path.join(SELF_DIR, checkpoint_file), map_location='cpu')



train_loader, valid_loader = load_data(TRAIN_CSV, VALID_CSV, TRAIN_DIR)



encoder = Encoder(COMPOUND_COEF, FEATURES_NUM)

encoder_head = EncoderHead(FEATURES_NUM, EMBEDDING_SIZE)

arc = ArcFace(EMBEDDING_SIZE, CLASSES_NUM, S, M)



criterion = FocalLoss(GAMMA_FOCAL)



if USE_CUDA:

    encoder = encoder.cuda()

    encoder_head = encoder_head.cuda()

    arc = arc.cuda()



optim_encoder = torch.optim.Adam(

    params=encoder.parameters(),

    lr=LR_ENCODER,

    weight_decay=WEIGHT_DECAY,

)

optim_arc = torch.optim.Adam(

    params=itertools.chain(encoder_head.parameters(), arc.parameters()),

    lr=LR_ARC,

    weight_decay=WEIGHT_DECAY,

)

scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(

    optimizer=optim_encoder,

    T_0=len(train_loader) * T_0,

    T_mult=T_MULT,

    eta_min=1e-6,

)

scheduler_arc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(

    optimizer=optim_arc,

    T_0=len(train_loader) * T_0,

    T_mult=T_MULT,

    eta_min=1e-6,

)



encoder.load_state_dict(checkpoint['encoder'])

encoder_head.load_state_dict(checkpoint['encoder_head'])

arc.load_state_dict(checkpoint['arc'])

optim_encoder.load_state_dict(checkpoint['optim_encoder'])

optim_arc.load_state_dict(checkpoint['optim_arc'])

scheduler_encoder.load_state_dict(checkpoint['scheduler_encoder'])

scheduler_arc.load_state_dict(checkpoint['scheduler_arc'])



loss_list = np.load(os.path.join(SELF_DIR, 'loss_list_arc.npy')).tolist()



train(

    start_epoch=checkpoint['epoch']+1,

    max_epoch=EPOCHS,

    encoder=encoder,

    encoder_head=encoder_head,

    arc=arc,

    criterion=criterion,

    optim_encoder=optim_encoder,

    optim_arc=optim_arc,

    scheduler_encoder=scheduler_encoder,

    scheduler_arc=scheduler_arc,

    train_loader=train_loader,

    loss_list=loss_list,

    valid_loader=valid_loader,

    valid_n_epochs=2,

)
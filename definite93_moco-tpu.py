VERSION = "20200516"  #@param ["1.5" , "20200516", "nightly"]

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version $VERSION
import gc

import logging

import numpy as np

import os

import pandas as pd

import time

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch_xla

import torch_xla.core.xla_model as xm

import torch_xla.debug.metrics as met

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.utils.utils as xu

from torch.autograd import Variable as var

from torchvision import datasets, transforms

import torchvision.models as models

import matplotlib.pyplot as plot

from torch.utils.data import Dataset

from PIL import Image

from glob import glob





SERIAL_EXEC = xmp.MpSerialExecutor()

losses_df = []

logging.basicConfig(filename='./moco_tpu.log', filemode='w', format='%(levelname)s - %(message)s')
def saveModel(epoch, model, optimizer, loss, path):

    torch.save({

              'epoch': epoch,

              'model_state_dict': model.state_dict(),

              'optimizer_state_dict': optimizer.state_dict(),

              'loss': loss

              }, path)
def loadModel(model, path):

    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']

    loss = checkpoint['loss']



    print('Model Loaded! Epoch: ',epoch,'Loss: ',loss)

    return model, epoch, loss;
def get_random_augmentation():

    return transforms.Compose([

        transforms.RandomResizedCrop(size=32,scale=(0.2, 1.)),

        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),

        transforms.RandomHorizontalFlip(),

        transforms.RandomGrayscale(p=0.2),

        transforms.ToTensor(),        

        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    ])
class TwoRandomCrops:

    def __init__(self, transform):

        self.transform = transform

    

    def __call__(self,x):

        query = self.transform(x)

        key = self.transform(x)

        return [query, key]
PARAMETERS = {}

PARAMETERS['data_dir'] = '../input/imagenetmini-1000/imagenet-mini/'

PARAMETERS['train_dir'] = '../input/imagenetmini-1000/imagenet-mini/train'

PARAMETERS['val_dir'] = '../input/imagenetmini-1000/imagenet-mini/val'

PARAMETERS['model_name'] = 'resnet50'

PARAMETERS['model_saved'] = 'moco_saved_imgnet_mini_4096K_M0999.pth'

PARAMETERS['model_load'] = '../input/models/moco_saved_imgnet_mini_4096K_M0999.pth'

PARAMETERS['learning_rate'] = 0.03

PARAMETERS['momentum'] = 0.999

PARAMETERS['epochs'] = 120

PARAMETERS['weight_decay'] = 0.0001

PARAMETERS['batch_size'] = 128

PARAMETERS['temperature'] = 0.07

PARAMETERS['num_channels'] = 3

PARAMETERS['dictionary_size'] = 4096

PARAMETERS['num_workers'] = 4

PARAMETERS['num_cores'] = 8

PARAMETERS['log_steps'] = 20

PARAMETERS['load_from_saved'] = True

PARAMETERS['start_epoch'] = 1

PARAMETERS['train_mode'] = False

world_size = xm.xrt_world_size()

rank = xm.get_ordinal()
os.environ['MASTER_ADDR'] = 'localhost'

os.environ['MASTER_PORT'] = '12355'

torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
device = xm.xla_device()
class ImageNet(Dataset):

    def __init__(self, root_dir, train=False, transform=None):



            self.root_dir = root_dir

            

            self.transform = transform



            self.sub_directory = 'train' if train else 'val'

            

            path = os.path.join(

            root_dir, self.sub_directory, "*","*")

            

            self.imgs = glob(path)

            

    def __len__(self):

        return len(self.imgs)

    

    def __getitem__(self,idx):

        img = Image.open(self.imgs[idx],).convert('RGB')

        if self.transform is not None:

            img = self.transform(img);



        return img;




# train_data = datasets.EMNIST(root='./data/', split='byclass', train=True,

#                                         download=True, transform=transform)

# test_data = datasets.EMNIST(root='./data/', split='byclass', train=False,

#                                        download=True, transform=transform)



# train_data = datasets.CIFAR10(root='./data/', train=True,

#                                         download=True, transform=transform)

# test_data = datasets.CIFAR10(root='./data/', train=False,

#                                        download=True, transform=transform)



# train_data = ImageNet(root_dir=PARAMETERS['data_dir'], train=True,

#                                          transform=TwoRandomCrops(get_random_augmentation()))

# test_data = ImageNet(root_dir=PARAMETERS['data_dir'], train=False,

#                                        transform=TwoRandomCrops(get_random_augmentation()))



train_data = datasets.ImageFolder(PARAMETERS['train_dir'], TwoRandomCrops(get_random_augmentation()))

test_data = datasets.ImageFolder(PARAMETERS['val_dir'], TwoRandomCrops(get_random_augmentation()))



train_sampler = torch.utils.data.distributed.DistributedSampler(

    train_data,

    num_replicas = xm.xrt_world_size(),

    rank = xm.get_ordinal())



train_set = torch.utils.data.DataLoader(

    train_data, 

    batch_size = PARAMETERS['batch_size'],

    sampler = train_sampler,

    num_workers = PARAMETERS['num_workers'], 

    pin_memory = True,drop_last=True)



test_set = torch.utils.data.DataLoader(

    test_data, 

    batch_size = PARAMETERS['batch_size'],

    shuffle = False,

    num_workers = PARAMETERS['num_workers'],

    pin_memory = True,drop_last=True)
N = PARAMETERS['batch_size']

LR = PARAMETERS['learning_rate'] * xm.xrt_world_size()

T = PARAMETERS['temperature']

C = PARAMETERS['num_channels']

K = PARAMETERS['dictionary_size']

m = PARAMETERS['momentum']
@torch.no_grad()

def gather_tensors_from_tpu(tensor):

#     tensors_gather = [torch.ones_like(tensor)

#         for _ in range(torch.distributed.get_world_size())]

    tensors_gather = xm.all_gather(tensor,dim=0)



#     return torch.cat(tensors_gather,dim=0)

    return tensors_gather
class EncoderModel(nn.Module):

    def __init__(self, base_model_name, channels_out):

        super(EncoderModel, self).__init__()



        if base_model_name == 'resnet50':

            model = models.resnet50(pretrained=False)

        elif base_model_name == 'resnet18':

            model = models.resnet18(pretrained=False)

        

        penultimate = model.fc.weight.shape[1]

        modules = list(model.children())[:-1]

        self.encoder = nn.Sequential(*modules)



        self.relu = nn.ReLU()

        self.fc = nn.Linear(penultimate, channels_out);

    

    def forward(self,x):

        x = self.encoder(x)

        x = x.view(x.size(0),-1)

        x = self.relu(x)

        x = self.fc(x)

        

        return x
class MoCoModel(nn.Module):

    def __init__(self):

        super(MoCoModel, self).__init__()



        self.query_enc = EncoderModel(PARAMETERS['model_name'],N)

        self.key_enc = EncoderModel(PARAMETERS['model_name'],N)



        for param_q, param_k in zip(self.query_enc.parameters(), self.key_enc.parameters()):

            param_k.data.copy_(param_q.data)

            param_k.requires_grad = False  # not update by gradient



        self.register_buffer("queue", torch.randn(N, K))

        self.queue = nn.functional.normalize(self.queue, dim=0)



        self.register_buffer("queue_index", torch.zeros(1, dtype=torch.long))



    @torch.no_grad()

    def update_key_params(self):

        for p_k,p_q in zip(self.key_enc.parameters(),self.query_enc.parameters()):

            val = (1-m)*p_q.data + m*p_k.data

            p_k.data = p_k.data.copy_(val)



    @torch.no_grad()

    def update_queue(self, keys):

        keys = gather_tensors_from_tpu(keys)



        index = int(self.queue_index)



        self.queue[:, index:index + N] = keys.T

        index = (index + N) % K



        self.queue_index[0] = index



    @torch.no_grad()

    def shuffle(self, x):



        current_batch_size = x.shape[0]

        x_gather = gather_tensors_from_tpu(x)

        gathered_batch_size = x_gather.shape[0]



        num_tpus = gathered_batch_size // current_batch_size



        shuffle_index = torch.randperm(gathered_batch_size).cpu()



        torch.distributed.broadcast(shuffle_index, src=0)



        unshuffle_index = torch.argsort(shuffle_index)



        current_tpu = xm.get_ordinal()

        current = shuffle_index.view(num_tpus, -1)[current_tpu]



        return x_gather[current], unshuffle_index



    @torch.no_grad()

    def unshuffle(self, x, unshuffle_index):



        current_batch_size = x.shape[0]

        x_gather = gather_tensors_from_tpu(x)

        gathered_batch_size = x_gather.shape[0]



        num_tpus = gathered_batch_size // current_batch_size



        current_tpu = xm.get_ordinal()

        current = unshuffle_index.view(num_tpus, -1)[current_tpu]



        return x_gather[current]



    def forward(self, images_q, images_k):

        q = self.query_enc(images_q)

        q = nn.functional.normalize(q,dim=1)



        with torch.no_grad():

            self.update_key_params()

            images_k, unshuffle_index = self.shuffle(images_k)

            

            k = self.key_enc.forward(images_k)

            k = nn.functional.normalize(k, dim=1)

            

            k = self.unshuffle(k, unshuffle_index)



        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)



        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])





        logits = torch.cat([l_pos, l_neg], dim=1)



        labels = torch.zeros(N).type(torch.LongTensor).to(device)



        logits = logits/T;



        self.update_queue(k)



        return logits,labels
def accuracy(output, target, topk=(1,)):

    with torch.no_grad():

        maxk = max(topk)

        batch_size = target.size(0)



        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))



        res = []

        for k in topk:

            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))

        return res
class EpochAccuracy(object):



    def __init__(self, acc_type):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0

        self.acc_type = acc_type



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count



    def __get__(self):

        return self.avg
def train_model():



    

    

    model = MoCoModel()

    model = torch.nn.parallel.DistributedDataParallel(model)



    # Wrapping to Xla XMP Wrapper

    WRAPPED_MODEL = xmp.MpModelWrapper(model)

    print(PARAMETERS['load_from_saved'])



    if (PARAMETERS['load_from_saved']):

        moco_model, PARAMETERS['start_epoch'], loss = loadModel(model, PARAMETERS['model_load'])

        PARAMETERS['start_epoch'] += 1

        PARAMETERS['load_from_saved'] = False

        print("Loaded model loss", loss)

        WRAPPED_MODEL = xmp.MpModelWrapper(moco_model)



    # Only instantiate model weights once in memory.

    moco_model = WRAPPED_MODEL.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(moco_model.parameters(), lr=LR, momentum=0.9, weight_decay=PARAMETERS['weight_decay'])



    def training_loop(data):

        epoch_loss = 0.0

        running_loss = 0.0

        tracker = xm.RateTracker()

        moco_model.train()

        top1_acc = EpochAccuracy('top1')

        top5_acc = EpochAccuracy('top5')



        for i, (images,_) in enumerate(data):

            optimizer.zero_grad()



            logits, labels = moco_model.forward(images[0],images[1])



            loss = loss_function(logits, labels)



            loss.backward()



            xm.optimizer_step(optimizer)



            epoch_loss += loss.item()

            running_loss += loss.item()

            

            batch_acc1, batch_acc5 = accuracy(logits,labels,topk=(1,5))



            top1_acc.update(batch_acc1[0].item(),images[0].size(0))

            top5_acc.update(batch_acc5[0].item(),images[0].size(0))



            tracker.add(PARAMETERS['batch_size'])

            # if((i+1) % 5 == 0):

            #   print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(

            #         xm.get_ordinal(), i, running_loss/5, tracker.rate(),

            #         tracker.global_rate(), time.asctime()), flush=True)

            #   running_loss = 0.0



        return epoch_loss, running_loss, top5_acc.__get__()



    def testing_loop(data):

        total = 0

        correct = 0

        validation_loss = 0

        top1_acc = EpochAccuracy('top1')

        top5_acc = EpochAccuracy('top5')

        moco_model.eval()

        images, labels, pred = None, None, None

        with torch.no_grad():

            for i, (images,_) in enumerate(data):

                logits, labels = moco_model.forward(images[0],images[1])



                loss_v = loss_function(logits,labels)

                validation_loss += loss_v.item()



                batch_acc1, batch_acc5 = accuracy(logits,labels,topk=(1,5))



                top1_acc.update(batch_acc1[0].item(),images[0].size(0))

                top5_acc.update(batch_acc5[0].item(),images[0].size(0))



        epoch_acc = (top1_acc.__get__(),top5_acc.__get__())

        return epoch_acc, validation_loss/len(data)





    acc = 0.0

    data, pred, target = None, None, None

    

    if(PARAMETERS['train_mode']):

        for epoch in range(PARAMETERS['start_epoch'], PARAMETERS['epochs'] + 1):

            para_loader = pl.ParallelLoader(train_set, [device],fixed_batch_size=True)



            #Train for single epoch

            epoch_loss, running_loss, train_acc_epoch = training_loop(para_loader.per_device_loader(device))



            xm.save({

                      'epoch': epoch,

                      'model_state_dict': moco_model.state_dict(),

                      'optimizer_state_dict': optimizer.state_dict(),

                      'loss': (epoch_loss/len(train_set))

                      }, PARAMETERS['model_saved'])



            train_loss = epoch_loss/len(train_set)

            #para_loader = pl.ParallelLoader(test_set, [device],fixed_batch_size=True)

            #acc, validation_loss = testing_loop(para_loader.per_device_loader(device))



    #         xm.master_print("["+str(epoch)+" , "+str(train_loss)+" , "+str(train_acc_epoch)+", "+str(validation_loss)+","+str(acc[1])+"]")

            xm.master_print("["+str(epoch)+" , "+str(train_loss)+" , "+str(train_acc_epoch)+"]")



            #logging.INFO('Epoch: ', epoch + 1, 'Loss: ', (epoch_loss/len(train_set)),'Top1Accuracy: ',acc[0],'%', ' Top5Accuracy: ',acc[1],'%', 'Validation Loss ', validation_loss)

        

    

    para_loader = pl.ParallelLoader(test_set, [device],fixed_batch_size=True)

    acc, validation_loss = testing_loop(para_loader.per_device_loader(device))

    return acc, data, pred, target, moco_model
def start_training(rank, parameters):

    global PARAMETERS

    PARAMETERS = parameters

    torch.set_default_tensor_type('torch.FloatTensor')

    acc, data, pred, target, model = train_model()

    print('Top1-Accuracy: ',str(acc[0]),'%', ' Top5-Accuracy: ',str(acc[1]),'%')





PARAMETERS['load_from_saved'] = True

PARAMETERS['train_mode'] = True

# xmp.spawn(start_training, args=(PARAMETERS, ), nprocs = PARAMETERS['num_cores'],

#           start_method='fork')

xmp.spawn(start_training, args=(PARAMETERS, ), nprocs = world_size,

          start_method='fork')
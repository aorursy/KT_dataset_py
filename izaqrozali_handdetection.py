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
%matplotlib inline

import os

import matplotlib.pyplot as plt

import numpy as np



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils import data



import torchvision

import torchvision.datasets as dset

import torchvision.transforms as transforms



import time

import shutil
base_path = "../input/hand/hand"

train_path = base_path + "/train"

valid_path = base_path + "/valid"
mean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tmft =  transforms.Compose([

    transforms.RandomRotation(30),

    transforms.CenterCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(mean_val,std_val)

])



valid_tmft =  transforms.Compose([

    transforms.Resize(224),

#     transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean_val,std_val)

])



# TODO: Load the datasets with ImageFolder

trainset = torchvision.datasets.ImageFolder(train_path, transform=train_tmft)

validset = torchvision.datasets.ImageFolder(valid_path, transform=valid_tmft)



BSIZE = 20

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BSIZE, shuffle=True, num_workers=0)

valid_loader = torch.utils.data.DataLoader(validset, batch_size=BSIZE, shuffle=True, num_workers=0)



CLAZZ = len(trainset.classes)

CLAZZ


dataiter = iter(valid_loader)

images, labels = dataiter.next()
img = images[2].permute(1,2,0).numpy()

# img = np.clip(img, 0, 1)

plt.imshow(img, cmap='gray')

class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):

        self.reset()



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

        

        

def accuracy(output, target, topk=(1,)):

    """Computes the accuracy over the k top predictions for the specified values of k"""

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
class AdaptiveConcatPool2d(nn.Module):

    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz:int=None):

        "Output will be 2*sz or 2 if sz is None"

        super().__init__()

        sz = sz or 1

        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)



class Lambda(nn.Module):

    "An easy way to create a pytorch layer for a simple `func`."

    def __init__(self, func):

        "create a layer that simply calls `func` with `x`"

        super().__init__()

        self.func=func



    def forward(self, x): return self.func(x)

    

def Flatten()->torch.Tensor:

    "Flattens `x` to a single dimension, often used at the end of a model."

    return Lambda(lambda x: x.view((x.size(0), -1)))





class HandNetwork(nn.Module):

    def __init__(self, resnet, clazz):

        super(HandNetwork, self).__init__()

        self.pool_size = 1

        self.resnet = resnet

        

        # out_channels multiple by pool size and multiply by 2

        # multiply by 2 is get from torch cat of AdaptiveAvgPool2d and AdaptiveMaxPool2d

        in_features = self.get_last_layer_out_channels() * self.pool_size*self.pool_size*2

        

        self.resnet.avgpool = nn.Sequential(

            AdaptiveConcatPool2d(self.pool_size),

            Flatten()

        )

        

        self.resnet.fc = nn.Sequential(

            nn.BatchNorm1d(in_features),

            nn.Dropout(0.5),

            nn.Linear(in_features, in_features//2),

            nn.ReLU(inplace=True),

            

            nn.BatchNorm1d(in_features//2),

            nn.Dropout(0.3),

            nn.Linear(in_features//2, in_features//4),

            nn.ReLU(inplace=True),

            

            nn.BatchNorm1d(in_features//4),

            nn.Dropout(0.2),

            nn.Linear(in_features//4, clazz),

        )

        

    def forward(self, x):

        x = self.resnet(x)

        return x

        

    def get_last_layer_out_channels(self):

        if len(self.resnet.layer4) >=3:

            if type(self.resnet.layer4[2]) == torchvision.models.resnet.BasicBlock:

                return self.resnet.layer4[2].conv2.out_channels

            elif type(self.resnet.layer4[2]) == torchvision.models.resnet.Bottleneck:

                return self.resnet.layer4[2].conv3.out_channels

            else:

                return 0

        else:

            if type(self.resnet.layer4[1]) == torchvision.models.resnet.BasicBlock:

                return self.resnet.layer4[1].conv2.out_channels

            elif type(self.resnet.layer4[1]) == torchvision.models.resnet.Bottleneck:

                return self.resnet.layer4[1].conv3.out_channels

            else:

                return 0

            

    def freeze(self):

        for param in self.resnet.parameters():

            param.require_grad = False

        for param in self.resnet.fc.parameters():

            param.require_grad= True

            

    def unfreeze(self):

        for param in self.resnet.parameters():

            param.require_grad = True

                                         

  
resnet = torchvision.models.resnet18(pretrained=True)

model = HandNetwork(resnet, CLAZZ)

if torch.cuda.is_available():

    model.cuda()
batch_history = {

    'train':{'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]},

    'valid':{'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]}

}



def train(train_loader, model, criterion, optimizer, epoch, print_freq, save_history=True, ngpu=1):

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    batch_time = AverageMeter()

    data_time = AverageMeter()

    losses = AverageMeter()

    top1 = AverageMeter()

    top5 = AverageMeter()

    history = {'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]}

    

    # switch to train mode

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        if torch.cuda.is_available():

            input = input.cuda()

            target = target.cuda()

            

        optimizer.zero_grad()

        output = model(input)

        loss = criterion(output, target)

        

        #calculate everything

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1[0], input.size(0))

        top5.update(acc5[0], input.size(0))

        

        

        # compute gradient and do SGD step

        loss.backward()

        optimizer.step()

        

        # measure elapsed time

        batch_time.update(time.time() - end)

        end = time.time()



        if i % print_freq == 0:

            history['epoch'].append(epoch)

            history['loss'].append(losses.avg)

            history['acc_topk1'].append(top1.avg)

            history['acc_topk5'].append(top5.avg)

            print('Epoch: [{0}][{1}/{2}]\t'

                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'

                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'

                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(

                   epoch, i, len(train_loader), batch_time=batch_time,

                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return history





def validate(val_loader, model, criterion, epoch, print_freq, save_history=True, ngpu=1):

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    batch_time = AverageMeter()

    losses = AverageMeter()

    top1 = AverageMeter()

    top5 = AverageMeter()

    history = {'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]}



    # switch to evaluate mode

    model.eval()



    with torch.no_grad():

        end = time.time()

        for i, (input, target) in enumerate(val_loader):

            

            input = input.to(device)

            target = target.to(device)



            # compute output

            output = model(input)

            loss = criterion(output, target)



            # measure accuracy and record loss

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))

            top1.update(acc1[0], input.size(0))

            top5.update(acc5[0], input.size(0))



            # measure elapsed time

            batch_time.update(time.time() - end)

            end = time.time()



            if i % print_freq == 0:

                history['epoch'].append(epoch)

                history['loss'].append(losses.avg)

                history['acc_topk1'].append(top1.avg)

                history['acc_topk5'].append(top5.avg)

                print('Test: [{0}/{1}]\t'

                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'

                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(

                       i, len(val_loader), batch_time=batch_time, loss=losses,

                       top1=top1, top5=top5))



        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'

              .format(top1=top1, top5=top5))

    

    return history





def save_checkpoint(state, is_best, filename='checkpoint.pth'):

    torch.save(state, filename)

    if is_best:

        shutil.copyfile(filename, 'model_best.pth')
best_acc1 = 0

history = {'epoch':[], 'train_detail':[],'valid_detail':[],}



NUM_EPOCH = 20

NGPU = 1

PRINT_FREQ = 10

LRATE = 0.05



model.freeze()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.resnet.fc.parameters(), lr=LRATE, momentum=0.9)



for epoch in range(NUM_EPOCH):

    prev_time = time.time()

     # train for one epoch

    train_history = train(train_loader, model, criterion, optimizer, epoch, print_freq=PRINT_FREQ)

    

    # evaluate on validation set

    valid_history = validate(valid_loader, model, criterion, epoch, print_freq=PRINT_FREQ)

    acc1 = valid_history['acc_topk1'][len(valid_history['acc_topk1'])-1]

    

    history['epoch'].append(epoch)

    history['train_detail'].append(train_history)

    history['valid_detail'].append(valid_history)

    

    # remember best acc@1 and save checkpoint

    is_best = acc1 > best_acc1

    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({

        'epoch': epoch + 1,

        'batch_size': BSIZE,

        'learning_rate': LRATE,

        'total_clazz': CLAZZ,

        'class_to_idx': trainset.class_to_idx,

        'history': history,

        'arch': 'resnet18',

        'state_dict': model.state_dict(),

        'best_acc_topk1': best_acc1,

        'optimizer' : optimizer.state_dict(),

    }, is_best)

    

    curr_time = time.time()

    total_time = curr_time - prev_time

    print(f'Total Time / Epcoh : {total_time}')

   
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_network(filename):

    if os.path.isfile(filename): 

        checkpoint = torch.load(filename)

        resnet = torchvision.models.resnet18(pretrained=True)

        clazz = checkpoint['total_clazz']

        model = HandNetwork(resnet, clazz)

        model.load_state_dict(checkpoint['state_dict'])

        return model

    else:

        return None

    



def load_checkpoint(filename):

    if os.path.isfile(filename): 

        checkpoint = torch.load(filename)

        return checkpoint

    else:

        return None

    

model = load_network('checkpoint.pth')

checkpoint = load_checkpoint('checkpoint.pth')
import PIL

import PIL.Image

import torch.nn.functional as F



def idx_to_class(class_to_idx):

    idx2class = {}

    for key, val in class_to_idx.items():

        dt = {val:key}

        idx2class.update(dt)

    return idx2class



def get_hand_number(classes, class_to_idx):

    idx2class = idx_to_class(class_to_idx)

    nclass = classes.data.squeeze().numpy().tolist()

    name = []

    for key in nclass:

        name.append(idx2class[key])

    return nclass, name

    



def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,

        returns an Numpy array

    '''

    im = PIL.Image.open(image)

    mean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    do_transforms =  transforms.Compose([

        transforms.Resize(224),

#         transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(mean_val,std_val)

    ])

    im_tfmt = do_transforms(im)

    im_add_batch = im_tfmt.view(1, im_tfmt.shape[0], im_tfmt.shape[1], im_tfmt.shape[2])

    return im_add_batch





def predict(image_path, model, topk=5):

    ''' Predict the class (or classes) of an image using a trained deep learning model.

    '''

    image = process_image(image_path)

    model.eval()

    model = model.cpu()

    with torch.no_grad():

        output = model.forward(image)

        output = F.log_softmax(output, dim=1)

        ps = torch.exp(output)

        result = ps.topk(topk, dim=1, largest=True, sorted=True)

        

    return result
def imshow(image, ax=None, title=None):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    

    # PyTorch tensors assume the color channel is the first dimension

    # but matplotlib assumes is the third dimension

    image = image.numpy().transpose((1, 2, 0))

    

    # Undo preprocessing

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed

    image = np.clip(image, 0, 1)

    

    ax.imshow(image)

    

    return ax



image_file = valid_path + "/0/r_0_l_04.jpg"

out_im = process_image(image_file)

imshow(out_im.squeeze())



probs, classes = predict(image_file, model, topk=5)

class_index, class_name = get_hand_number(classes, checkpoint['class_to_idx'])



print(probs)

print(classes)

print(class_index)

print(class_name)
def view_classify(img_path, label_idx, prob, classes, class_to_idx):

    ''' Function for viewing an image and it's predicted classes.

    '''

    idx2class = idx_to_class(class_to_idx)

    img = np.asarray(PIL.Image.open(img_path))

    ps = prob.data.numpy().squeeze().tolist()

    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)

    ax1.imshow(img.squeeze())

    ax1.set_title(idx2class[label_idx])

    ax1.axis('off')

    

    ax2.barh(np.arange(5), ps)

    ax2.set_aspect(0.2)

    ax2.set_yticks(np.arange(5))

    

    

    class_idx, class_name = get_hand_number(classes, class_to_idx)

    ax2.set_yticklabels(class_name, size='large');

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
%%time

image_file = valid_path + "/2/r_2_l_21.jpg"

probs, classes = predict(image_file, model)

class_index, class_name = get_hand_number(classes, checkpoint['class_to_idx'])

print(probs)

print(classes)

print(class_index)

print(class_name)

# view_classify(image_file, 2, probs, classes, checkpoint['class_to_idx'])
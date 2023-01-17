# Imports here
%matplotlib inline
import numpy as np
import cv2
import os
import sys
import time
import json
import shutil

import PIL
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
print('Torch Version\t',torch.__version__)
print('PIL Version\t',PIL.__version__)
print('Torchvision\t',torchvision.__version__)
print('GPU Available \t',torch.cuda.is_available())
NGPU = 1
NUM_EPOCH = 100
BSIZE = 32
LRATE = 0.005
MOMENTUM=0.9

LRSTEP=7
GAMMA=0.1
PRINT_FREQ = 5
device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
print(device)
data_dir = '/kaggle/input/flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
# TODO: Define your transforms for the training and validation sets
#  scaling, rotations, mirroring, and/or cropping
mean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_transforms =  transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_val,std_val)
])

valid_transforms =  transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_val,std_val)
])

# TODO: Load the datasets with ImageFolder
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)

CLAZZ = len(train_dataset.classes)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BSIZE, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BSIZE, shuffle=True, num_workers=0)
# get category to flower name
def get_all_flower_names():
    with open('/kaggle/input/cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name

def flower_name(val, array_index=False):
    labels = get_all_flower_names()
    if array_index:
        val = val + 1
    return labels[str(val)]

FLOWER_LABELS = get_all_flower_names()
#test function
# print(FLOWER_LABELS)
# obtain one batch of training images
denorm = NormalizeInverse(mean_val, std_val)
dataiter = iter(train_loader)
images, labels = dataiter.next()
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(BSIZE-1):
    ax = fig.add_subplot(2, BSIZE/2, idx+1, xticks=[], yticks=[])
    img = denorm(images[idx]).permute(1,2,0).numpy()
    img = np.clip(img, 0, 1)
    plt.imshow(img, cmap='gray')
    ax.set_title(flower_name(labels[idx].item(), array_index=True))
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
    

# TODO: Build and train your network
class ResidualFlowerNetwork(nn.Module):
    def __init__(self, resnet, clazz):
        super(ResidualFlowerNetwork, self).__init__()
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
#             nn.ReLU(inplace=True)
#             nn.Linear(512, clazz)
        )
    def get_last_layer_out_channels(self):
        if type(self.resnet.layer4[2]) == torchvision.models.resnet.BasicBlock:
            return self.resnet.layer4[2].conv2.out_channels
        elif type(self.resnet.layer4[2]) == torchvision.models.resnet.Bottleneck:
            return self.resnet.layer4[2].conv3.out_channels
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
    
    def forward(self, x):
        x = self.resnet(x)
        return x

    
resnet = torchvision.models.resnet34(pretrained=True)
# print(resnet.fc.in_features) 
model = ResidualFlowerNetwork(resnet, CLAZZ)

if torch.cuda.is_available():
    model.cuda()
model    
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.resnet.fc.parameters(), lr=LRATE, momentum=MOMENTUM)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LRSTEP, gamma=GAMMA)
# x = torch.rand(2,3,224,224).cuda()
# model(x)
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
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
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


def adjust_learning_rate(optimizer, epoch, decay, lrate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lrate * (0.1 ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
       
def getsteplr(base_lr=0.001, max_lr=0.1, step=4):
    lr = base_lr
    hlr = max_lr
    step = hlr/(step-1)
    step_lr = np.arange(lr, hlr+step, step).tolist()
    return step_lr
getsteplr(base_lr=0.001, step=6)
    
best_acc1 = 0
history = {'epoch':[], 'train_detail':[],'valid_detail':[],}

NUM_EPOCH=34
#train only classifier
model.freeze()
model = model.cuda()

LRATE=0.05
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.resnet.fc.parameters(), lr=LRATE, momentum=MOMENTUM)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LRSTEP, gamma=GAMMA)

for epoch in range(NUM_EPOCH):    
    scheduler.step()
    # train pretrained network
    if epoch == 13:
        model.unfreeze()
        step_lr = getsteplr(base_lr=LRATE/100, max_lr=LRATE, step=6)
        print(step_lr)
        optimizer = optim.SGD(
            [
                {'params': model.resnet.conv1.parameters()},
                {'params': model.resnet.bn1.parameters()},
                {'params': model.resnet.relu.parameters()},
                {'params': model.resnet.maxpool.parameters()},
                {'params': model.resnet.layer1.parameters(), 'lr':step_lr[1]},
                {'params': model.resnet.layer2.parameters(), 'lr':step_lr[2]},
                {'params': model.resnet.layer3.parameters(), 'lr':step_lr[3]},
                {'params': model.resnet.layer4.parameters(), 'lr':step_lr[4]},
                {'params': model.resnet.avgpool.parameters(), 'lr':step_lr[4]},
                {'params': model.resnet.fc.parameters(), 'lr': step_lr[4]}
            ],
            lr=step_lr[0])
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LRSTEP, gamma=GAMMA)
    
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
        'class_to_idx': train_dataset.class_to_idx,
        'labels': FLOWER_LABELS,
        'history': history,
        'arch': 'resnet101',
        'state_dict': model.state_dict(),
        'best_acc_topk1': best_acc1,
        'optimiz1er' : optimizer.state_dict(),
    }, is_best)
# TODO: Save the checkpoint
train_dataset.class_to_idx
# checkpoint = torch.load('checkpoint.pth.tar')
# checkpoint['history']['train']
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_flower_network(filename):
    if os.path.isfile(filename): 
        checkpoint = torch.load(filename)
        resnet = torchvision.models.resnet34(pretrained=True)
        clazz = checkpoint['total_clazz']
        model = ResidualFlowerNetwork(resnet, clazz)
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
    
model = load_flower_network('checkpoint.pth')
checkpoint = load_checkpoint('checkpoint.pth')
model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = PIL.Image.open(image)
    mean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    do_transforms =  transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_val,std_val)
    ])
    im_tfmt = do_transforms(im)
    im_add_batch = im_tfmt.view(1, im_tfmt.shape[0], im_tfmt.shape[1], im_tfmt.shape[2])
    return im_add_batch
# test process image
valid_dir = data_dir + '/valid/'
image_file = valid_dir +"10/image_07094.jpg"
out_im = process_image(image_file)
out_im.shape
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

imshow(out_im.squeeze())
def getFlowerClassIndex(classes, class_to_idx):
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    class_to_flower_class_idx = [idx_to_class[lab] for lab in classes.squeeze().numpy().tolist()]
    flower_class_to_name = [flower_name(cls_idx) for cls_idx in class_to_flower_class_idx]
    return class_to_flower_class_idx, flower_class_to_name

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


valid_dir = data_dir + '/valid/'
image_file = valid_dir +"10/image_07094.jpg"
probs, classes = predict(image_file, model, topk=5)
class_index, class_name = getFlowerClassIndex(classes, checkpoint['class_to_idx'])

print(probs)
print(classes)
print(class_index)
print(class_name)
def view_classify(img_path, label_idx, prob, classes, class_to_idx):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img = np.asarray(PIL.Image.open(img_path))
    ps = prob.data.numpy().squeeze().tolist()
    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
    ax1.imshow(img.squeeze())
    ax1.set_title(flower_name(label_idx))
    ax1.axis('off')
    
    ax2.barh(np.arange(5), ps)
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(5))
    
    
    class_idx, class_name = getFlowerClassIndex(classes, class_to_idx)
    ax2.set_yticklabels(class_name, size='large');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


valid_dir = data_dir + '/valid/'
image_file = valid_dir +"2/image_05136.jpg"
probs, classes = predict(image_file, model)
print(probs)
view_classify(image_file, 2, probs, classes, checkpoint['class_to_idx'])
# # track test loss

# valid_loader2 = torch.utils.data.DataLoader(valid_dataset, batch_size=BSIZE, shuffle=True, num_workers=0)
# test_loss = 0.0
# class_correct = list(0. for i in range(102))
# class_total = list(0. for i in range(102))
# batch_size = BSIZE
# model.eval()
# model.to(device)
# # iterate over test data
# for data, target in valid_loader2:
#     # move tensors to GPU if CUDA is available
#     data, target = data.to(device), target.to(device)
#     # forward pass: compute predicted outputs by passing inputs to the model
#     output = model(data)
#     # calculate the batch loss
#     loss = criterion(output, target)
#     # update test loss 
#     test_loss += loss.item()*data.size(0)
#     # convert output probabilities to predicted class
#     _, pred = torch.max(output, 1)    
#     # compare predictions to true label
#     correct_tensor = pred.eq(target.data.view_as(pred))
#     correct = np.squeeze(correct_tensor.numpy())
#     # calculate test accuracy for each object class
#     batch_size = target.size(0)
#     for i in range(batch_size):
#         label = target.data[i]
#         class_correct[label] += correct[i].item()
#         class_total[label] += 1
# # average test loss
# test_loss = test_loss/len(valid_loader.dataset)
# print('Test Loss: {:.6f}\n'.format(test_loss))
# # valid_loader.dataset.class_to_idx['1'], class_correct[0], class_total[0]
# print(getFlowerClassIndex(pred.cpu(), valid_loader.dataset.class_to_idx))
# for i in range():
#     if class_total[i] > 0:
#         total = 100 * class_correct[i] / class_total[i]
#         total_correct = np.sum(class_correct[i])
#         total_class = np.sum(class_total[i])
#         clzz = valid_loader.dataset.class_to_idx[str(i+1)]
#         print(f'Test Accuracy of {clzz}: {total}% ({total_correct}/{total_class})')
        
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

# print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#     100. * np.sum(class_correct) / np.sum(class_total),
#     np.sum(class_correct), np.sum(class_total)))
# valid_history = validate(valid_loader, model, criterion, epoch, print_freq=PRINT_FREQ)

# # valid_history
!ls -al
# !mv checkpoint.pth resnet_101_checkpoint.pth
# !mv checkpoint.pth resnet_101_checkpoint.pth
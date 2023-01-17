import sys
import os
import os.path
import random
import collections
import shutil
import time
import glob
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

ROOT_DIR = os.getcwd()
DATA_HOME_DIR = '../input/chest_xray/chest_xray'
%matplotlib inline
print(os.listdir('../input/chest_xray/chest_xray'))
# paths
data_path = DATA_HOME_DIR + '/' 
split_train_path = data_path + 'train/'
valid_path = data_path + 'val/'
test_path = DATA_HOME_DIR + 'test/'
saved_model_path = ROOT_DIR + 'models/'


# data
batch_size = 16


# model
nb_runs = 1
nb_aug = 3
epochs = 35
lr = 1e-4
clip = 0.001
archs = ["resnet101"]
MODEL = 'resnet101'
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
best_prec1 = 0
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    
    # switch to train mode
    model.train()
    
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('   * EPOCH {epoch} | Accuracy: {acc.avg:.3f} | Loss: {losses.avg:.3f}'.format(epoch=epoch,
                                                                                         acc=acc,
                                                                                         losses=losses))

    return acc.avg
def test(test_loader, model):
    csv_map = collections.defaultdict(float)
    
    # switch to evaluate mode
    model.eval()
    
    for aug in range(nb_aug):
        print("   * Predicting on test augmentation {}".format(aug + 1))
        
        for i, (images, filepath) in enumerate(test_loader):
            # pop extension, treat as id to map
            filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
            filepath = int(filepath)

            image_var = torch.autograd.Variable(images, volatile=True)
            y_pred = model(image_var)
            # get the index of the max log-probability
            smax = nn.Softmax()
            smax_out = smax(y_pred)[0]
            normal_prob = smax_out.data[0]
            pnemo_prob = smax_out.data[1]
            prob = pnemo_prob
            if normal_prob > pnemo_prob:
                prob = 1 - normal_prob
            prob = np.around(prob, decimals=4)
            prob = np.clip(prob, clip, 1-clip)
            csv_map[filepath] += (prob / nb_aug)

    sub_fn = '{0}epoch_{1}clip_{2}runs'.format(epochs, clip, nb_runs)
    
    for arch in archs:
        sub_fn += "_{}".format(arch)
        
    print("Writing Predictions to CSV...")
    with open(sub_fn + '.csv', 'w') as csvfile:
        fieldnames = ['id', 'label']
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'label'))
        for row in sorted(csv_map.items()):
            csv_w.writerow(row)
    print("Done.")
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
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
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr
    lr = lr * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in sorted(glob.glob(test_path + "*.jpg")):
            images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)
def shear(img):
    width, height = img.size
    m = random.uniform(-0.05, 0.05)
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), Image.AFFINE,
                        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                        Image.BICUBIC)
    return img
def main(mode="train", resume=False):
    
    global best_prec1
    
    for arch in archs:

        # create model
        print("=> Starting {0} on '{1}' model".format(mode, arch))
        model = models.__dict__[arch](pretrained=True)
        # Don't update non-classifier learned features in the pretrained networks
        for param in model.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        # Final dense layer needs to replaced with the previous out chans, and number of classes
        # in this case -- resnet 101 - it's 2048 with two classes

        model.fc = nn.Linear(2048, 2)

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            
        # optionally resume from a checkpoint
        if resume:
            if os.path.isfile(resume):
                print("=> Loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            else:
                print("=> No checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        # Data loading code
        traindir = split_train_path
        valdir = valid_path
        testdir = test_path

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_loader = data.DataLoader(
            datasets.ImageFolder(traindir,
                                 transforms.Compose([
#                                      transforms.Lambda(shear),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        val_loader = data.DataLoader(
            datasets.ImageFolder(valdir,
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        test_loader = data.DataLoader(
            TestImageFolder(testdir,
                            transforms.Compose([
#                                 transforms.Lambda(shear),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)
        
        
        if mode == "test":
            test(test_loader, model)
            return
        
        # define loss function (criterion) and pptimizer
        criterion = nn.CrossEntropyLoss().cuda()
        
        if mode == "validate":
            validate(val_loader, model, criterion, 0)
            return

        optimizer = optim.Adam(model.module.fc.parameters(), lr, weight_decay=1e-5)

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best Accuracy and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
main(mode="train")
main(mode="validate", resume='model_best.pth.tar')
main(mode="test", resume='model_best.pth.tar')
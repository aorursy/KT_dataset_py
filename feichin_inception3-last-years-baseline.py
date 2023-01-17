import os

import shutil

import time

import numpy as np



import torch

import torch.nn as nn

import torch.backends.cudnn as cudnn

import torch.optim

import torch.utils.data

#import torchvision.models as models



import urllib.request



import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo



import torch.utils.data as data

from PIL import Image

import os

import json

from torchvision import transforms

import random

import numpy as np
# uncomment this to work with last years baseline model

# performance will however be bad, with this commented out, we will just work

# on a pre trained image net model

# print('Downloading model files...', end="")

# url = 'http://vision.caltech.edu/~macaodha/inat2018/iNat_2018_InceptionV3.pth.tar'  

# urllib.request.urlretrieve(url, 'iNat_2018_InceptionV3.pth.tar')

# print('Done.')
class Params:

    # arch = 'inception_v3'

    num_classes = 1010

    workers = 8

    epochs = 4

    start_epoch = 0

    batch_size = 64  # might want to make smaller 

    lr = 0.0045

    lr_decay = 0.94

    epoch_decay = 4

    momentum = 0.9

    weight_decay = 1e-4

    print_freq = 100



    resume = 'iNat_2018_InceptionV3.pth.tar'    # path to trained model

    train_file = '../input/train2019.json'      # path to train file

    val_file = '../input/test2019.json'         # path to test file

    data_root_train = '../input/train_val2019/' # path to train images

    data_root_test = '../input/test2019/'       # path to test images

    op_file_name = 'submission.csv'             # submission filename



    # set evaluate to True to run the test set

    evaluate = True

    save_preds = True
# This code was copied from https://github.com/macaodha/inat_comp_2018/blob/master/inception.py



# Same as the version from the official start_epoch

# ttps://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

# Only change being that it can take variable sized inputs

# See line 122



__all__ = ['Inception3', 'inception_v3']





model_urls = {

    # Inception v3 ported from TensorFlow

    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',

}





def inception_v3(pretrained=False, **kwargs):

    r"""Inception v3 model architecture from

    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    if pretrained:

        if 'transform_input' not in kwargs:

            kwargs['transform_input'] = True

        model = Inception3(**kwargs)

        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))

        return model



    return Inception3(**kwargs)





class Inception3(nn.Module):



    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):

        super(Inception3, self).__init__()

        self.aux_logits = aux_logits

        self.transform_input = transform_input

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)

        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)

        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)

        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        self.Mixed_5b = InceptionA(192, pool_features=32)

        self.Mixed_5c = InceptionA(256, pool_features=64)

        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)

        self.Mixed_6c = InceptionC(768, channels_7x7=160)

        self.Mixed_6d = InceptionC(768, channels_7x7=160)

        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if aux_logits:

            self.AuxLogits = InceptionAux(768, num_classes)

        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)

        self.Mixed_7c = InceptionE(2048)

        self.fc = nn.Linear(2048, num_classes)



        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                import scipy.stats as stats

                stddev = m.stddev if hasattr(m, 'stddev') else 0.1

                X = stats.truncnorm(-2, 2, scale=stddev)

                values = torch.Tensor(X.rvs(m.weight.data.numel()))

                values = values.view(m.weight.data.size())

                m.weight.data.copy_(values)

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()



    def forward(self, x):

        if self.transform_input:

            x = x.clone()

            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5

            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5

            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        # 299 x 299 x 3

        x = self.Conv2d_1a_3x3(x)

        # 149 x 149 x 32

        x = self.Conv2d_2a_3x3(x)

        # 147 x 147 x 32

        x = self.Conv2d_2b_3x3(x)

        # 147 x 147 x 64

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # 73 x 73 x 64

        x = self.Conv2d_3b_1x1(x)

        # 73 x 73 x 80

        x = self.Conv2d_4a_3x3(x)

        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # 35 x 35 x 192

        x = self.Mixed_5b(x)

        # 35 x 35 x 256

        x = self.Mixed_5c(x)

        # 35 x 35 x 288

        x = self.Mixed_5d(x)

        # 35 x 35 x 288

        x = self.Mixed_6a(x)

        # 17 x 17 x 768

        x = self.Mixed_6b(x)

        # 17 x 17 x 768

        x = self.Mixed_6c(x)

        # 17 x 17 x 768

        x = self.Mixed_6d(x)

        # 17 x 17 x 768

        x = self.Mixed_6e(x)

        # 17 x 17 x 768

        if self.training and self.aux_logits:

            aux = self.AuxLogits(x)

        # 17 x 17 x 768

        x = self.Mixed_7a(x)

        # 8 x 8 x 1280

        x = self.Mixed_7b(x)

        # 8 x 8 x 2048

        x = self.Mixed_7c(x)

        # 8 x 8 x 2048

        x = F.adaptive_avg_pool2d(x, 1)

        #x = F.avg_pool2d(x, kernel_size=8)

        # 1 x 1 x 2048

        x = F.dropout(x, training=self.training)

        # 1 x 1 x 2048

        x = x.view(x.size(0), -1)

        # 2048

        x = self.fc(x)

        # 1000 (num_classes)

        if self.training and self.aux_logits:

            return x, aux

        return x





class InceptionA(nn.Module):



    def __init__(self, in_channels, pool_features):

        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)



        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)

        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)



        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)

        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)



        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)



    def forward(self, x):

        branch1x1 = self.branch1x1(x)



        branch5x5 = self.branch5x5_1(x)

        branch5x5 = self.branch5x5_2(branch5x5)



        branch3x3dbl = self.branch3x3dbl_1(x)

        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)



        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_pool = self.branch_pool(branch_pool)



        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionB(nn.Module):



    def __init__(self, in_channels):

        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)



        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)

        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)



    def forward(self, x):

        branch3x3 = self.branch3x3(x)



        branch3x3dbl = self.branch3x3dbl_1(x)

        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)



        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)



        outputs = [branch3x3, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionC(nn.Module):



    def __init__(self, in_channels, channels_7x7):

        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)



        c7 = channels_7x7

        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)

        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))

        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))



        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)

        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))

        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))



        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)



    def forward(self, x):

        branch1x1 = self.branch1x1(x)



        branch7x7 = self.branch7x7_1(x)

        branch7x7 = self.branch7x7_2(branch7x7)

        branch7x7 = self.branch7x7_3(branch7x7)



        branch7x7dbl = self.branch7x7dbl_1(x)

        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)

        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)

        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)

        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)



        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_pool = self.branch_pool(branch_pool)



        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionD(nn.Module):



    def __init__(self, in_channels):

        super(InceptionD, self).__init__()

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)



        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)



    def forward(self, x):

        branch3x3 = self.branch3x3_1(x)

        branch3x3 = self.branch3x3_2(branch3x3)



        branch7x7x3 = self.branch7x7x3_1(x)

        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)

        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)

        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)



        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]

        return torch.cat(outputs, 1)





class InceptionE(nn.Module):



    def __init__(self, in_channels):

        super(InceptionE, self).__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)



        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)

        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))

        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))



        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)

        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)

        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))

        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))



        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)



    def forward(self, x):

        branch1x1 = self.branch1x1(x)



        branch3x3 = self.branch3x3_1(x)

        branch3x3 = [

            self.branch3x3_2a(branch3x3),

            self.branch3x3_2b(branch3x3),

        ]

        branch3x3 = torch.cat(branch3x3, 1)



        branch3x3dbl = self.branch3x3dbl_1(x)

        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch3x3dbl = [

            self.branch3x3dbl_3a(branch3x3dbl),

            self.branch3x3dbl_3b(branch3x3dbl),

        ]

        branch3x3dbl = torch.cat(branch3x3dbl, 1)



        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_pool = self.branch_pool(branch_pool)



        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]

        return torch.cat(outputs, 1)





class InceptionAux(nn.Module):



    def __init__(self, in_channels, num_classes):

        super(InceptionAux, self).__init__()

        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)

        self.conv1 = BasicConv2d(128, 768, kernel_size=5)

        self.conv1.stddev = 0.01

        self.fc = nn.Linear(768, num_classes)

        self.fc.stddev = 0.001



    def forward(self, x):

        # 17 x 17 x 768

        x = F.avg_pool2d(x, kernel_size=5, stride=3)

        # 5 x 5 x 768

        x = self.conv0(x)

        # 5 x 5 x 128

        x = self.conv1(x)

        # 1 x 1 x 768

        x = x.view(x.size(0), -1)

        # 768

        x = self.fc(x)

        # 1000

        return x





class BasicConv2d(nn.Module):



    def __init__(self, in_channels, out_channels, **kwargs):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)



    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        return F.relu(x, inplace=True)
# The following was copied from https://github.com/macaodha/inat_comp_2018/blob/master/train_inat.py



# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py



best_prec3 = 0.0  # store current best top 3



def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()

    data_time = AverageMeter()

    losses = AverageMeter()

    top1 = AverageMeter()

    top3 = AverageMeter()



    # switch to train mode

    model.train()



    end = time.time()

    print('Epoch:{0}'.format(epoch))

    print('Itr\t\tTime\t\tData\t\tPrec@1\t\tPrec@3')

    for i, (input, im_id, target, tax_ids) in enumerate(train_loader):

        # measure data loading time

        data_time.update(time.time() - end)



        input = input.cuda()

        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input)

        target_var = torch.autograd.Variable(target)



        # compute output

        output = model(input_var)

        loss = criterion(output, target_var)



        # measure accuracy and record loss

        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))

        # losses.update(loss.data[0], input.size(0))

        top1.update(prec1[0], input.size(0))

        top3.update(prec3[0], input.size(0))



        # compute gradient and do SGD step

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        # measure elapsed time

        batch_time.update(time.time() - end)

        end = time.time()



        if i % args.print_freq == 0:

            print('[{0}/{1}]\t'

                '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'

                '{data_time.val:.2f} ({data_time.avg:.2f})\t'

                '{top1.val:.2f} ({top1.avg:.2f})\t'

                '{top3.val:.2f} ({top3.avg:.2f})'.format(

                i, len(train_loader), batch_time=batch_time,

                data_time=data_time, top1=top1, top3=top3))





def validate(val_loader, model, criterion, save_preds=False):

    batch_time = AverageMeter()

    losses = AverageMeter()

    top1 = AverageMeter()

    top3 = AverageMeter()



    # switch to evaluate mode

    model.eval()



    end = time.time()

    pred = []

    im_ids = []



    print('Validate:\tTime\t\tPrec@1\t\tPrec@3')

    for i, (input, im_id, target, tax_ids) in enumerate(val_loader):

        input = input.cuda()

        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)

        target_var = torch.autograd.Variable(target, volatile=True)



        # compute output

        output = model(input_var)

        # loss = criterion(output, target_var)



        if save_preds:

            # store the top K classes for the prediction

            im_ids.append(im_id.cpu().numpy().astype(np.int))

            _, pred_inds = output.data.topk(5,1,True,True)

            pred.append(pred_inds.cpu().numpy().astype(np.int))



        # measure accuracy and record loss

        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))

        # losses.update(loss.data[0], input.size(0))



        top1.update(prec1[0], input.size(0))

        top3.update(prec3[0], input.size(0))



        # measure elapsed time

        batch_time.update(time.time() - end)

        end = time.time()

        

        if i % args.print_freq == 0:

            print('[{0}/{1}]\t'

                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'

                  '{top1.val:.2f} ({top1.avg:.2f})\t'

                  '{top3.val:.2f} ({top3.avg:.2f})'.format(

                   i, len(val_loader), batch_time=batch_time,

                   top1=top1, top3=top3))



    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'

          .format(top1=top1, top3=top3))



    if save_preds:

        return top3.avg, np.vstack(pred), np.hstack(im_ids)

    else:

        return top3.avg





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    torch.save(state, filename)

    if is_best:

        print("\tSaving new best model")

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

    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr





def accuracy(output, target, topk=(1,)):

    """Computes the precision@k for the specified values of k"""

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
# The following was copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py



def default_loader(path):

    return Image.open(path).convert('RGB')



def load_taxonomy(ann_data, tax_levels, classes):

    # loads the taxonomy data and converts to ints

    taxonomy = {}



    if 'categories' in ann_data.keys():

        num_classes = len(ann_data['categories'])

        for tt in tax_levels:

            tax_data = [aa[tt] for aa in ann_data['categories']]

            _, tax_id = np.unique(tax_data, return_inverse=True)

            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))

    else:

        # set up dummy data

        for tt in tax_levels:

            taxonomy[tt] = dict(zip([0], [0]))



    # create a dictionary of lists containing taxonomic labels

    classes_taxonomic = {}

    for cc in np.unique(classes):

        tax_ids = [0]*len(tax_levels)

        for ii, tt in enumerate(tax_levels):

            tax_ids[ii] = taxonomy[tt][cc]

        classes_taxonomic[cc] = tax_ids



    return taxonomy, classes_taxonomic





class INAT(data.Dataset):

    def __init__(self, root, ann_file, is_train=True):



        # load annotations

        print('Loading annotations from: ' + os.path.basename(ann_file))

        with open(ann_file) as data_file:

            ann_data = json.load(data_file)



        # set up the filenames and annotations

        self.imgs = [aa['file_name'] for aa in ann_data['images']]

        self.ids = [aa['id'] for aa in ann_data['images']]



        # if we dont have class labels set them to '0'

        if 'annotations' in ann_data.keys():

            self.classes = [aa['category_id'] for aa in ann_data['annotations']]

        else:

            self.classes = [0]*len(self.imgs)



        # load taxonomy

        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']

                           #8142, 4412,    1120,     273,     57,      25,       6

        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)



        # print out some stats

        print('\t' + str(len(self.imgs)) + ' images')

        print('\t' + str(len(set(self.classes))) + ' classes')



        self.root = root

        self.is_train = is_train

        self.loader = default_loader



        # augmentation params

        self.im_size = [299, 299]  # can change this to train on higher res

        self.mu_data = [0.485, 0.456, 0.406]

        self.std_data = [0.229, 0.224, 0.225]

        self.brightness = 0.4

        self.contrast = 0.4

        self.saturation = 0.4

        self.hue = 0.25



        # augmentations

        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))

        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])

        self.flip_aug = transforms.RandomHorizontalFlip()

        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)

        self.tensor_aug = transforms.ToTensor()

        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)



    def __getitem__(self, index):

        path = self.root + self.imgs[index]

        im_id = self.ids[index]

        img = self.loader(path)

        species_id = self.classes[index]

        tax_ids = self.classes_taxonomic[species_id]



        if self.is_train:

            img = self.scale_aug(img)

            img = self.flip_aug(img)

            img = self.color_aug(img)

        else:

            img = self.center_crop(img)



        img = self.tensor_aug(img)

        img = self.norm_aug(img)



        return img, im_id, species_id, tax_ids



    def __len__(self):

        return len(self.imgs)
# The following was copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py



args = Params()



# load pretrained model

print("Using pre-trained inception_v3")

model = inception_v3(pretrained=True)

model.fc = nn.Linear(2048, args.num_classes)

model.aux_logits = False

model = model.cuda()



# define loss function (criterion) and optimizer

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr,

                            momentum=args.momentum,

                            weight_decay=args.weight_decay)



# optionally resume from a checkpoint

if args.resume:

    if os.path.isfile(args.resume):

        print("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)

        args.start_epoch = checkpoint['epoch']

        best_prec3 = checkpoint['best_prec3']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"

               .format(args.resume, checkpoint['epoch']))

    else:

        print("=> no checkpoint found at '{}'".format(args.resume))



    cudnn.benchmark = True



# data loading code

train_dataset = INAT(args.data_root_train, args.train_file,

                     is_train=True)

val_dataset = INAT(args.data_root_test, args.val_file,

                     is_train=False)



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,

                   shuffle=True, num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,

                                         batch_size=args.batch_size, shuffle=False,

                                         num_workers=args.workers, pin_memory=True)



for epoch in range(args.start_epoch, args.epochs):

    adjust_learning_rate(optimizer, epoch)



    # train for one epoch

    train(train_loader, model, criterion, optimizer, epoch)



    # evaluate on validation set

    prec3 = validate(val_loader, model, criterion, False)



    # remember best prec@1 and save checkpoint

    is_best = prec3 > best_prec3

    best_prec3 = max(prec3, best_prec3)

    save_checkpoint({

        'epoch': epoch + 1,

        #'arch': args.arch,

        'state_dict': model.state_dict(),

        'best_prec3': best_prec3,

        'optimizer' : optimizer.state_dict(),

    }, is_best)



if args.evaluate:

    prec3, preds, im_ids = validate(val_loader, model, criterion, True)

    # write predictions to file

    if args.save_preds:

        with open(args.op_file_name, 'w') as opfile:

            opfile.write('id,predicted\n')

            for ii in range(len(im_ids)):

                opfile.write(str(im_ids[ii]) + ',' + ' '.join([str(j) for j in preds[ii, :]]) + '\n')
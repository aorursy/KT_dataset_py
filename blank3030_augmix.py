# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pwd
from __future__ import print_function

import os
import shutil
import time

import augmentations

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import datasets
from torchvision import models
from torchvision import transforms
!nvidia-smi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
args = {
    'model': 'resnet50',
    'mixture_depth': -1,
    'epochs': 10,
    'learning_rate': 0.1,
    'batch_size': 32,
    'eval_batch_size': 16,
    'momentum': 0.9,
    'decay': 0.0001,
    'mixture_width': 3,
    'aug_severity': 1,
    'aug_prob_coeff': 1.0,
    'no-jsd': False,
    'save': '/kaggle/working',
    'resume': '',
    'evaluate': False,
    'print_freq': 10,
    'pretrained': False,
    'num_workers': 4
}


ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]
augmentations.IMAGE_SIZE = 224
lr = 0 

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
    global lr
    b = args['batch_size'] / 256.
    ep = [5, 7, 9]
    if epoch == 0:
        lr = args['learning_rate']*b
        print('Learning Rate: ',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch in ep:
        lr = lr / 10
        print('Learning rate:',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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

def aug(image, preprocess):
    ws = np.float32(
      np.random.dirichlet([args['aug_prob_coeff']] * args['mixture_width']))
    m = np.float32(np.random.beta(args['aug_prob_coeff'], args['aug_prob_coeff']))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args['mixture_width']):
        image_aug = image.copy()
        a = args['mixture_depth'] if args['mixture_depth'] > 0 else np.random.randint(
                1, 4)
        for _ in range(a):
            op = np.random.choice(augmentations.augmentations)
            image_aug = op(image_aug, args['aug_severity'])
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

class AugMixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, no_jsd):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                      aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

def train(net, train_loader, optimizer):
    net.train()
    data_ema = 0.
    batch_ema = 0.
    loss_ema = 0.
    acc1_ema = 0.
    acc5_ema = 0.

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # Compute data loading time
        data_time = time.time() - end
        optimizer.zero_grad()

        if args['no-jsd']:
            images = images.to(device)
            targets = targets.to(device)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        else:
            images_all = torch.cat(images, 0).to(device)
            targets = targets.to(device)
            logits_all = net(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
              logits_all, images[0].size(0))

          # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
              logits_clean, dim=1), F.softmax(
                  logits_aug1, dim=1), F.softmax(
                      logits_aug2, dim=1)

          # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking

        loss.backward()
        optimizer.step()

        # Compute batch computation time and update moving averages.
        batch_time = time.time() - end
        end = time.time()

        data_ema = data_ema * 0.1 + float(data_time) * 0.9
        batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
        loss_ema = loss_ema * 0.1 + float(loss) * 0.9
        acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
        acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9

        if i % args['print_freq'] == 0:
            print(
              'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
              '{:.3f} | Train Acc5 {:.3f}'.format(i, len(train_loader), data_ema,
                                                  batch_ema, loss_ema, acc1_ema,
                                                  acc5_ema))

    return loss_ema, acc1_ema, batch_ema

def test(net, test_loader):
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)

def main():
    torch.manual_seed(1)
    np.random.seed(1)

  # Load datasets
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
      [transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
      transforms.ToPILImage()])
    preprocess = transforms.Compose(
      [transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
  ])
    all_data = datasets.ImageFolder(root='../input/natural-images/data/natural_images')
    train_data_len = int(len(all_data)*0.8)
    valid_data_len = int((len(all_data) - train_data_len))
    train_dataset, val_dataset = random_split(all_data, [train_data_len, valid_data_len])
#     train_dataset.dataset.transform = train_transform
#     val_dataset.dataset.transform = test_transform
    train_dataset = AugMixDataset(train_dataset, preprocess, no_jsd = False)
    val_dataset = AugMixDataset(val_dataset, test_transform, no_jsd = True)
    train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args['batch_size'],
      shuffle=True,
      num_workers=args['num_workers'])
    val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args['eval_batch_size'],
      shuffle=False,
      num_workers=args['num_workers'])

    if args['pretrained']:
        print("=> using pre-trained model '{}'".format(args['model']))
        net = models.__dict__[args['model']](pretrained=True)
    else:
        print("=> creating model '{}'".format(args['model']))
        net = models.__dict__[args['model']](num_classes = 8)

    optimizer = torch.optim.SGD(
      net.parameters(),
      args['learning_rate'],
      momentum=args['momentum'],
      weight_decay=args['decay'])

  # Distribute model across all visible GPUs
    net = net.to(device)
    cudnn.benchmark = True

    start_epoch = 0

    if args['resume']:
        if os.path.isfile(args['resume']):
            checkpoint = torch.load(args['resume'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model restored from epoch:', start_epoch)

    if args['evaluate']:
        test_loss, test_acc1 = test(net, val_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Acc1 {:.3f}'.format(
            test_loss, 100 * test_acc1))

    if not os.path.exists(args['save']):
        os.makedirs(args['save'])
    if not os.path.isdir(args['save']):
        raise Exception('%s is not a dir' % args['save'])
        
    log_path = os.path.join(args['save'],
                          '{}_training_log.csv'.format(args['model']))
    with open(log_path, 'w') as f:
        f.write(
            'epoch,batch_time,train_loss,train_acc1(%),test_loss,test_acc1(%)\n')

    best_acc1 = 0
    print('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, args['epochs']):
        adjust_learning_rate(optimizer, epoch)
        
        train_loss_ema, train_acc1_ema, batch_ema = train(net, train_loader,
                                                      optimizer)
        test_loss, test_acc1 = test(net, val_loader)

        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)
        checkpoint = {
            'epoch': epoch,
            'model': args['model'],
            'state_dict': net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(args['save'], 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(args['save'], 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%0.3f,%0.6f,%0.2f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                batch_ema,
                train_loss_ema,
                100. * train_acc1_ema,
                test_loss,
                100. * test_acc1,
              ))

        print(
            'Epoch {:3d} | Train Loss {:.4f} | Test Loss {:.3f} | Test Acc1 '
            '{:.4f}'
            .format((epoch + 1), train_loss_ema, test_loss, 100. * test_acc1))


if __name__ == '__main__':
    main()



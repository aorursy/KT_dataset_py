# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import itertools

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
    
class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        csv = pd.read_csv(csv_file) 
        self.data = csv.drop('label', axis=1).values
        self.label = csv['label'].values
        del csv

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, mask = self.data[idx].reshape(1,28,28), self.label[idx]
        img, mask = np.asarray(img), np.asarray(mask)
        img, mask = torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask.astype(np.float32))
        img = img/255
        return img, mask
# analyze train set
train_set = MNISTDataset('/kaggle/input/digit-recognizer/train.csv')

train_size = len(train_set)
sample_idx = np.random.randint(0, len(train_set), dtype=np.int64)
imdata, imlabel = train_set[sample_idx]
print('Số lượng mẫu trong tập huấn luyện gốc: ', train_size)
print('Mẫu tại chỉ số: ', sample_idx) #Lấy ngẫu nhiên
print(' + Kiểu dữ liệu của mẫu: ', type(imdata))
print(' + Kiểu dữ liệu của nhãn: ', type(imlabel))

imdata = np.array(imdata)
print(' + Kích thước một mẫu dữ liệu: ', imdata.shape)
print(' + Giá trị nhỏ nhất trong array: ', imdata.min())
print(' + Giá trị lớn nhất trong array: ', imdata.max())
print(' + Giá trị trung bình trong array: ', imdata.mean())
print(' + Giá trị std trong array: ', imdata.std())
imdata = imdata.reshape(28,28)

plt.figure(figsize=(2,2))
plt.imshow(imdata,  cmap='gray')
plt.axis('off')
title = 'Nhãn: {:d}'.format(int(imlabel))
plt.title(title)
plt.show()
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import NoNorm

def show_random_image(dataset, nrows, ncols):
    _, axarr = plt.subplots(nrows, ncols, figsize=(10, 6))
    for r in range(nrows):
      for c in range(ncols):
        sample_idx = np.random.randint(0, len(dataset), dtype=np.int64)
        imdata, imlabel = dataset[sample_idx]
        imdata = imdata.numpy()
        imlabel = int(imlabel)
        imdata = imdata.reshape((28,28))
     
        axarr[r,c].imshow(imdata,  cmap='gray',norm=NoNorm())
        axarr[r,c].set_title(imlabel)
        plt.setp(axarr[r,c].get_xticklabels(), visible=False)
        plt.setp(axarr[r,c].get_yticklabels(), visible=False)
        axarr[r,c].grid(False)
    plt.show()

show_random_image(train_set, 5, 10) 
import matplotlib as mplt

def plot_dataset_stats(datasets, legends, class_names=None):
    # Thống kê số lượng dữ liệu các nhãn trong các tập dữ liệu
    colors = ['red', 'green', 'blue', 'maganta', 'cyan']
    num_classes = 0
    pos = []
    max_count = 0
    for idx in range(len(datasets)):
        cur_dataset = datasets[idx]
        cur_legend = legends[idx]
        labels = [cur_dataset[sid][1] for sid in range(len(cur_dataset))]
        label_hist = np.bincount(labels)
        num_classes = max(num_classes, len(label_hist))

        pos = 5.0*np.arange(num_classes) + idx*1.0
        plt.bar(pos, label_hist, align='center', 
              alpha=0.5, color=colors[idx], label=cur_legend)
        max_count = max(max_count, label_hist.max())
    if class_names is None:
        class_names = [str(idx) for idx in range(num_classes)]

    plt.xticks(pos, class_names)
    plt.ylabel('The number of samples')
    plt.xlabel('Label')
    plt.ylim(0, max_count + 500)
    plt.title('The number sample of each labels')
    plt.legend()
    plt.show()

train_dataset = MNISTDataset('/kaggle/input/digit-recognizer/train.csv')

# split train test => train_set and val_set
TRAIN_SIZE_PRC = 0.7 # train (70%) và val (30%)

TRAIN_SIZE = int(TRAIN_SIZE_PRC*len(train_dataset))
VALID_SIZE = len(train_dataset) - TRAIN_SIZE
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,(TRAIN_SIZE, VALID_SIZE))

plot_dataset_stats([train_dataset, valid_dataset],
                  ['Train_set', 'Val_set'])

print('Train set - size: ', TRAIN_SIZE)
labels = [train_dataset[sid][1] for sid in range(len(train_dataset))]
print('The number sample of each labels (0->9): ', np.bincount(labels))
print('Val set - size: ', VALID_SIZE)
labels = [valid_dataset[sid][1] for sid in range(len(valid_dataset))]
print('The number sample of each labels (0->9): ', np.bincount(labels))
# Create data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True, 
    num_workers=4, 
    pin_memory=True) 
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=128, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True)

def show_info(ds_name, dataset, loader):
    print('Tên tập dữ liệu:', ds_name)
    print('Số lượng mẫu trong tập:', len(dataset))
    print('Số lượng batch được chia:', len(loader))
    print('Thông tin của từng batch:')

    for batch_idx, (batch_data, batch_label) in enumerate(loader):
        IDX = '{:5d}'.format(batch_idx + 1)
        print(IDX, "input: ", batch_data.shape,"target: ", batch_label.shape)
        break

print('-'*80)
show_info('TẬP HUẤN LUYỆN', train_dataset, train_loader)
print('-'*80)
show_info('TẬP KIỂM THỬ', valid_dataset, valid_loader)
print('-'*80)
import torch.nn.functional as F
BEST_MODEL_FNAME = './best_model.data'
best_acc = 0

def train(train_loader, model, criterion, optimizer, epoch, use_cuda=False, print_freq=40):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.long().cuda()
        else:
            data = data
            target = target.long()
        
        scores = model(data)
        loss = criterion(scores, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = scores.float()
        scores = F.softmax(scores, dim=1)
        loss = loss.float()
        
        # measure precision and record loss
        prec1 = metric(scores.data, target.data)[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))
    
    return losses.avg, top1.avg

def validate(val_loader, model, criterion, use_cuda=True, print_freq=40):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    global best_acc

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda()
            input_var = data.cuda()
            target_var = target.long().cuda()
        else:
            input_var = data
            target_var = target.long()
        
        # compute output
        scores = model(input_var)
        loss = criterion(scores, target_var)

        scores = scores.float()
        scores = F.softmax(scores, dim=1)
        loss = loss.float()

        prec1 = metric(scores.data, target_var.data)[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
          
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    
    #save
    if top1.avg > best_acc:
          best_acc = top1.avg
          torch.save(model.state_dict(), BEST_MODEL_FNAME)
          print('Mô hình tốt hơn đã được tìm thấy => lưu lại')

    return losses.avg, top1.avg, best_acc

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
    lr = base_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def metric(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        
        self.conv2_drop = nn.Dropout2d(p=0.5)
        
        self.fc1 = nn.Linear(320, 50)
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        x = x.view(-1, 320)
        
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return x
  
model = Net()
print(model)

base_lr = 0.05
momentum = 0.9
weight_decay = 5e-4
nepochs = 120

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            base_lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
loss_lst = {}
loss_lst['train'] = []
loss_lst['val']   = []

prec_lst = {}
prec_lst['train'] = []
prec_lst['val']   = []

for epoch in range(0, nepochs):
    adjust_learning_rate(optimizer, epoch)
    if use_cuda:
        model.cuda()
        loss_func.cuda()
        loss_train, prec_train = train(train_loader, model, loss_func, optimizer, epoch, True, print_freq=70)
        loss_val, prec_val, best_avg = validate(valid_loader, model, loss_func, True, print_freq=50)
        
        loss_lst['train'].append(loss_train)
        loss_lst['val'].append(loss_val)
        prec_lst['train'].append(prec_train)
        prec_lst['val'].append(prec_val)
fig, ax = plt.subplots(2,1)
ax[0].plot(loss_lst['train'], color='b', label="Training loss")
ax[0].plot(loss_lst['val'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(prec_lst['train'], color='b', label="Training precision")
ax[1].plot(prec_lst['val'], color='r',label="Validation precision")
legend = ax[1].legend(loc='best', shadow=True)
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix of val set',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
x_val = np.array([])
pred_val = np.array([])
gt_val = np.array([])

print("Best precision of val phase:", best_avg)
for i, (data, target) in enumerate(valid_loader):
    target = target.numpy().astype(np.int)
    pred = model(data.cuda())
    pred = pred.float()
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred,1).cpu().numpy()
    
    x_val = np.concatenate((x_val, data.numpy()), axis=0) if x_val.shape[0] != 0 else data.numpy()
    pred_val = np.append(pred_val, pred).astype(np.int)
    gt_val = np.append(gt_val, target).astype(np.int)
    
confusion_mtx = confusion_matrix(gt_val, pred_val) 
plot_confusion_matrix(confusion_mtx, classes = range(10))
errors = (pred_val - gt_val != 0)
x_val_err = x_val[errors]
y_pred_err = pred_val[errors]
gt_err = gt_val[errors]

def display_errors(img_errors, pred_errors, gt_errors):
    n = 0
    nrows = 3
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, figsize=(10,10))
    for row in range(nrows):
        for col in range(ncols):
            index = np.random.randint(0, gt_errors.shape[0], dtype=np.int64)
            ax[row,col].imshow((img_errors[index]).reshape((28,28)), cmap='gray')
            ax[row,col].set_title("Pred label :{}\nTrue label :{}".format(pred_errors[index], gt_errors[index]))
            n += 1
            
    fig.tight_layout()
    plt.show()

display_errors(x_val_err, y_pred_err, gt_err)
class NetBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        
        self.drop = nn.Dropout2d(p=0.5)
        
        self.fc1 = nn.Linear(320, 50)
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu1(x)
        
        x = self.bn2(self.conv2(x))
        x = self.drop(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        x = x.view(-1, 320)
        
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return x
  
model = NetBN()
print(model)

base_lr = 0.05
momentum = 0.9
weight_decay = 5e-4
nepochs = 120

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            base_lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
loss_lst = {}
loss_lst['train'] = []
loss_lst['val']   = []

prec_lst = {}
prec_lst['train'] = []
prec_lst['val']   = []

for epoch in range(0, nepochs):
    adjust_learning_rate(optimizer, epoch)
    if use_cuda:
        model.cuda()
        loss_func.cuda()
        loss_train, prec_train = train(train_loader, model, loss_func, optimizer, epoch, True, print_freq = 50)
        loss_val, prec_val, best_avg = validate(valid_loader, model, loss_func, True, print_freq = 50)
        
        loss_lst['train'].append(loss_train)
        loss_lst['val'].append(loss_val)
        prec_lst['train'].append(prec_train)
        prec_lst['val'].append(prec_val)
x_val = np.array([])
pred_val = np.array([])
gt_val = np.array([])

print("Best precision of val phase:", best_avg)
for i, (data, target) in enumerate(valid_loader):
    target = target.numpy().astype(np.int)
    pred = model(data.cuda())
    pred = pred.float()
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred,1).cpu().numpy()
    
    x_val = np.concatenate((x_val, data.numpy()), axis=0) if x_val.shape[0] != 0 else data.numpy()
    pred_val = np.append(pred_val, pred).astype(np.int)
    gt_val = np.append(gt_val, target).astype(np.int)
    
confusion_mtx = confusion_matrix(gt_val, pred_val) 
plot_confusion_matrix(confusion_mtx, classes = range(10))
# load test set
test_dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').values

# prediction
test_set = torch.from_numpy(test_dataset.reshape(-1, 1, 28, 28)).to(torch.float32)
pred = model(test_set.cuda())
pred = pred.float()
pred = F.softmax(pred, dim=1)
pred = torch.argmax(pred,1).cpu().numpy()

# save result to submit
df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df['Label'] = pred
df.to_csv('submission.csv',line_terminator='\r\n', index=False)
!rm submission.csv
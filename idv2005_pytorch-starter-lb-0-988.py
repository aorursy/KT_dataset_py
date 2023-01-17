# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from PIL import Image

import torch

import torch.utils.data as data

from torchvision import datasets, models, transforms

from torch.autograd import Variable
data_dir = '../input'
# define custom dataset

def load_img(filename):

    img_path = data_dir + '/train/' + str(filename) + '.jpg'

    with open(img_path, 'rb') as f:

        with Image.open(f) as img_f:

            return img_f.convert('RGB').resize((320,320))

# define custom dataset

class MyDataSet(data.Dataset):

    def __init__(self, filename, training=True, validating=False, train_percent=0.85, transforms=None):

        df = pd.read_csv(filename)

        if training:

            split_index = (int)(df.values.shape[0]*train_percent)

            if validating:

                split_data = df.values[split_index:]

            else:

                split_data = df.values[:split_index]

            imgs = [None]*split_data.shape[0]

            labels = [None]*split_data.shape[0]

            for i, row in enumerate(split_data):

                fn, labels[i] = row

                imgs[i] = load_img(fn)

        else:

            imgs = [None]*df.values.shape[0]

            for i, row in enumerate(df.values):

                fn, _ = row

                imgs[i] = load_img(fn)

        self.imgs = imgs

        self.training = training

        self.transforms = transforms

        self.num = len(imgs)

        if self.training:

            self.labels = np.array(labels, dtype=np.float32)

    def __getitem__(self, index):

        if not self.transforms is None:

            img = self.transforms(self.imgs[index])

        if self.training:

            return img, self.labels[index]

        else:

            return img

    def __len__(self):

        return self.num
# define data augumentations

import random

def randomRotate(img):

    angel = random.randint(0,4) * 90

    return img.rotate(angel)

 

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomSizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.Lambda(lambda x: randomRotate(x)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test': transforms.Compose([

        transforms.Scale(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

}
def get_data_loader(filename=data_dir+'/train_labels.csv', training=True, validating=False, shuffle=True):

    if training and not validating:

        transkey = 'train'

    else:

        transkey = 'test'

    dset = MyDataSet(filename, training=training, validating=validating, transforms=data_transforms[transkey])

    loader = torch.utils.data.DataLoader(dset, batch_size=8, shuffle=shuffle, num_workers=4)

    loader.num = dset.num

    return loader
def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=8):

    lr = 0

    for param_group in optimizer.param_groups:

        lr = param_group['lr']

    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:

        lr = lr * 0.6

    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:

        print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:

            param_group['lr'] = lr

    return optimizer    
weight_file = 'best_model.pth'



def do_train(net, criterion, optimizer, lr_scheduler, epochs=100):

    data_loaders = {'train': get_data_loader(), 'valid': get_data_loader(validating=True)}

    best_model = net

    best_acc = 0

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs))

        for phase in ['train', 'valid']:

            if phase == 'train':

                optimizer = lr_scheduler(optimizer, epoch)

                net.train(True)

            else:

                net.train(False)

            running_loss = 0.

            running_corrects = 0

            for imgs, labels in data_loaders[phase]:

                imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())

                optimizer.zero_grad()

                outputs = net(imgs)

                preds = torch.ge(outputs.data, 0.5).resize_(labels.data.size())

                loss = criterion(outputs, labels)

                if phase == 'train':

                    loss.backward()

                    optimizer.step()

                running_loss += loss.data[0]

                running_corrects += torch.sum(preds.int() == labels.data.int())

            epoch_loss = running_loss / data_loaders[phase].num

            epoch_acc = running_corrects / data_loaders[phase].num

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                torch.save(net.state_dict(), weight_file)

                best_model = net

    print('Best validation accuracy: {:4f}'.format(best_acc))

    return best_model

                
import torch.nn as nn

def get_dense201():

    net = models.densenet201(pretrained=True)

    net.classifier = nn.Sequential(nn.Linear(net.classifier.in_features, 1), nn.Sigmoid())

    return net.cuda()

def train_net():

    net = get_dense201()

    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    do_train(net, criterion, optimizer, lr_scheduler)
def predict(net):

    loader = get_data_loader(filename=data_dir+'/sample_submission.csv', training=False, shuffle=False)

    preds = []

    net.eval()

    for i, img in enumerate(loader, 0):

        inputs = Variable(img.cuda())

        outputs = net(inputs)

        pred = outputs.data.cpu().tolist()

        for p in pred:

            preds.append(p)

    return np.array(preds)
def submit(preds, filename):

    df = pd.read_csv(data_dir + '/sample_submission.csv')

    df['invasive'] = preds

    print(df.head())

    df.to_csv(filename, index=False)
if True:

    train_net()



if True:

    net = get_dense201()

    net.load_state_dict(torch.load(weight_file))

    preds = predict(net)

    submit(preds, 'submission1.csv')
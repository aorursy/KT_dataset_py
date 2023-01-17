# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install torchsummary
import time

import copy

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from sys import stdout

from PIL import Image

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from torchsummary import summary

from torchvision import transforms

import torch.nn.functional as F

import matplotlib.pyplot as plt

%matplotlib inline
class CreateDatasetFromDataFrame(Dataset):

    def __init__(self, data_, data_dim_, transform=None):

        """

        import pandas DataFrame

        return torch Tensor.

        :param data_: pandas dataset

        :param transform: torch transforms

        """

        self.dt = list(data_.values)

        self.transform = transform

        self.labels = np.asarray([x[0] for x in self.dt])

        self.images = np.asarray([x[1:] for x in self.dt]).reshape((-1, data_dim_, data_dim_)).astype('float32')



    def __len__(self):

        return len(self.labels)



    def __getitem__(self, item):

        label = self.labels[item]

        image = self.images[item]

        if self.transform is not None:

            pil_image = Image.fromarray(np.array(image))

            image = self.transform(pil_image)

        return image, label

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 6, 5, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

        )

        self.classifier = nn.Sequential(

            nn.Linear(16 * 5 * 5, 120),

            nn.ReLU(),

            nn.Linear(120, 84),

            nn.ReLU(),

            nn.Linear(84, 10)

        )

    

    def forward(self, x):

        x = self.features(x)  

        x = x.view(x.size(0), -1)

        out = self.classifier(x)

        return out

    



class CNNWithDropout(nn.Module):

    def __init__(self):

        super(CNNWithDropout, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 6, 5, padding=2),

            nn.Dropout(0.2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5),

            nn.Dropout(0.2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

        )

        self.classifier = nn.Sequential(

            nn.Linear(16 * 5 * 5, 120),

            nn.ReLU(),

            nn.Linear(120, 84),

            nn.ReLU(),

            nn.Linear(84, 10)

        )

    

    def forward(self, x):

        x = self.features(x)  

        x = x.view(x.size(0), -1)

        out = self.classifier(x)

        return out



    

class CNNWithBN(nn.Module):

    def __init__(self):

        super(CNNWithBN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 6, 5, padding=2),

            nn.BatchNorm2d(6),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

        )

        self.classifier = nn.Sequential(

            nn.Linear(16 * 5 * 5, 120),

            nn.ReLU(),

            nn.Linear(120, 84),

            nn.ReLU(),

            nn.Linear(84, 10)

        )

    

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        out = self.classifier(x)

        return out    

    



class MyCNN(nn.Module):

    def __init__(self):

        super(MyCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),

            nn.BatchNorm2d(256),

            nn.ReLU(),

        )

        self.classifier = nn.Sequential(

            nn.Linear(256, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Linear(64, 10),

        )



    def forward(self, x):

        x = self.features(x)

        # global average pooling

        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        out = self.classifier(x)

        out = F.log_softmax(out, dim=1)

        return out
def train(model_, device_, data_loader):

    global model_recorder

    since = time.time()

    model_.train()

    running_loss = 0.0

    running_corrects = 0

    sample_size = 0

    data_set_size = len(data_loader.dataset)



    for ind, data in enumerate(data_loader):

        loop_time_start = time.time()

        y_ = data[1].type(torch.LongTensor).to(device_)

        x_ = data[0].to(device_)



        # zero the parameter gradients

        optimizer.zero_grad()



        # get predict

        y_hat = model(x_)

        _, predicts = torch.max(y_hat, 1)



        # calculate loss

        loss = criterion(y_hat, y_)



        # backpropagation

        loss.backward()

        optimizer.step()



        # statistics

        running_loss += loss.item() * x_.size(0)

        running_corrects += torch.sum(predicts == y_.data)



        # batch sum in one epoch

        sample_size += len(x_)



        # etismate epoch cost

        one_loop_time = time.time() - loop_time_start

        one_loop_time *= data_set_size / data_loader.batch_size

        loop_time_diff = one_loop_time * (1. - (sample_size / data_set_size))

        loop_time_diff = '{:.0f}h {:.0f}m {:.0f}s'.format(loop_time_diff // 3600,

                                                          loop_time_diff // 60,

                                                          loop_time_diff % 60)



        # print log

        stdout.write(

            "\r%s" %

            "Training: [{:5d}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}\tRemaining Time: {:10s}".format(

                sample_size,

                data_set_size,

                100. * sample_size / data_set_size,

                loss.item(),

                running_corrects.double() / sample_size,

                loop_time_diff

            )

        )

        stdout.flush()



    t_ = time.time() - since

    epoch_loss = running_loss / data_set_size

    epoch_acc = running_corrects.double() / data_set_size



    # recorder

    model_recorder['train_acc'] += [epoch_acc]

    model_recorder['train_loss'] += [epoch_loss]

    # get learning rate

    optimizer_state = optimizer.state_dict()['param_groups'][0]



    model_recorder['train_lr'] += [optimizer_state['lr']]

    print()

    print('Epoch Time Costs: {:.0f}m {:.0f}s'.format(t_ // 60, t_ % 60))

    print()

    print('Train Set:\t| Average Loss: {:3.4f}\t| Accuracy: {:3.4f}\t| Learning Rate: {}'.format(

        epoch_loss,

        epoch_acc,

        optimizer_state['lr']

    ))





def valid(model_, device_, data_loader, acc, wts):

    global epoch_no_improve, n_epoch_stop, stop_flag

    global model_recorder

    model_.eval()

    epoch_loss = 0

    correct = 0

    model_wts = wts

    with torch.no_grad():

        for data, target in data_loader:

            data, target = data.to(device_), target.to(device_)

            output = model_(data)

            epoch_loss += criterion(output, target).item() * data.size(0)

            predict = output.max(1, keepdim=True)[1]

            correct += predict.eq(target.view_as(predict)).sum().item()



        epoch_loss /= len(data_loader.dataset)

        epoch_acc = float(correct) / len(data_loader.dataset)



        # recorder

        model_recorder['valid_acc'] += [epoch_acc]

        model_recorder['valid_loss'] += [epoch_loss]

        print("Test Set:\t| Average Loss: {:.4f}\t| Accuracy: {:3.4f}\t|\n".format(epoch_loss, epoch_acc))



    if epoch_acc > acc:

        print('New high accuracy: {}'.format(epoch_acc))

        print()

        acc = epoch_acc

        model_wts = copy.deepcopy(model_.state_dict())

        epoch_no_improve = 0

    else:

        epoch_no_improve += 1

        if epoch_no_improve == n_epoch_stop:

            stop_flag = True

    return acc, model_wts





def training_model(epochs_=25):

    early_stop_init()



    print("{:20s} {:^15s} {:20s}".format('=' * 20, model.__class__.__name__, '=' * 20))

    print("{:20s} {:^15s} {:20s}".format('=' * 20, 'Start Training', '=' * 20))

    global model_recorder_dict, model_recorder

    # Create a empty recorder

    model_recorder = create_recorder_dict()



    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(epochs_):

        print("{:20s} {:^15s} {:20s}".format('-' * 20, 'Epoch: {}'.format(epoch + 1), '-' * 20))

        train(model, device, train_data_loader)

        best_acc, best_model_wts = valid(model, device, test_data_loader, best_acc, best_model_wts)

        if stop_flag:

            print('Epoch Not improve. Early Stop.')

            break



        # recorder

        model_recorder['epoch_list'] += [epoch]

        # reduce learning rate here, not in the train function

        exp_lr_scheduler.step()



    t_ = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(t_ // 60, t_ % 60))

    print('Best Test Accuracy: {:4f}'.format(best_acc))



    model_recorder_dict.update({model.__class__.__name__: model_recorder})

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model





def create_recorder_dict():

    return {'epoch_list': [], 'train_acc': [], 'train_loss': [], 'valid_acc': [], 'valid_loss': [], 'train_lr': []}





def early_stop_init():

    global epoch_no_improve, n_epoch_stop, stop_flag

    epoch_no_improve, n_epoch_stop, stop_flag = 0, 3, False

    pass
train_dt = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test_dt = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

print('Training Set Shape', train_dt.shape)

print('Testing Set Shape', test_dt.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_labels = train_dt['label'].unique().tolist()

print('class_labels:', [class_names[x] for x in class_labels])
# Config -------------------------------------------------

# gray = 1, rgb = 3

channels = 1



# input image dimension

img_dim = 28



# image resize dimension

data_dim = 28



# batch size

batch_size = 128



# epochs

num_epochs = 50



# gpu device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# model dir

model_dir = '/'



# Create Early Stop

epoch_no_improve, n_epoch_stop, stop_flag = 0, 7, False



# Create Model Recorder Dictionary

model_recorder_dict = {}

model_recorder = create_recorder_dict()
# when using RandomRotation, get Image error...

data_transforms = {

    'train': transforms.Compose([

        transforms.Resize((img_dim, img_dim)),

        transforms.RandomHorizontalFlip(),

#         transforms.RandomRotation(degrees=5),

        transforms.ToTensor(),

        transforms.Normalize([0.485], [0.229])

    ]),

    'test': transforms.Compose([

        transforms.Resize((img_dim, img_dim)),

        transforms.ToTensor(),

        transforms.Normalize([0.485], [0.229])

    ]),

}
train_data_loader = DataLoader(

    CreateDatasetFromDataFrame(

        train_dt,

        data_dim_=data_dim,

        transform=data_transforms['train']

    ),

    batch_size=batch_size,

    shuffle=True



)

test_data_loader = DataLoader(

    CreateDatasetFromDataFrame(

        test_dt,

        data_dim_=data_dim,

        transform=data_transforms['test']

    ),

    batch_size=batch_size,

    shuffle=False

)
model_list = [

    CNN(),

    CNNWithDropout(),

    CNNWithBN(),

    MyCNN(),

]

# Training Process ------------------------------------------

for m in model_list:

    model = m.to(device)



    summary(model, (channels, img_dim, img_dim))



    # Create Loss Function

    criterion = nn.CrossEntropyLoss().to(device)



    # Create Optimization Function

    optimizer = optim.Adam(model.parameters(), lr=0.001)



    # Create Learning Rate Schedule

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



    model = training_model(num_epochs)



    # Save Model

    torch.save(model, model_dir + model.__class__.__name__ + 'model.pkl')

    torch.save(model.state_dict(), model_dir + model.__class__.__name__ + 'model_params.pkl')

for k, v in model_recorder_dict.items():

    plt.plot(v['epoch_list'], v['train_acc'][:len(v['epoch_list'])], 'bo', label='Training accuracy')

    plt.plot(v['epoch_list'], v['valid_acc'][:len(v['epoch_list'])], 'b', label='Validation accuracy')

    

#     for x,y in zip(v['epoch_list'], v['train_acc'][:len(v['epoch_list'])]):

#         label = "{:.2f}".format(y)

#         plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center', va='top')  

    for x,y in zip(v['epoch_list'], v['valid_acc'][:len(v['epoch_list'])]):

        label = "{:.2f}".format(y)

        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center', va='bottom', 

                     arrowprops=dict(arrowstyle="-", connectionstyle="arc3")) 

    

    

    plt.title('{} Training and validation accuracy'.format(k))

    plt.legend()

    plt.figure()

    plt.show()

    plt.cla()

    plt.plot(v['epoch_list'], v['train_loss'][:len(v['epoch_list'])], 'bo', label='Training loss')

    plt.plot(v['epoch_list'], v['valid_loss'][:len(v['epoch_list'])], 'b', label='Validation loss')

#     for x,y in zip(v['epoch_list'], v['train_loss'][:len(v['epoch_list'])]):

#         label = "{:.2f}".format(y)

#         plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center', va='top') 

    for x,y in zip(v['epoch_list'], v['valid_loss'][:len(v['epoch_list'])]):

        label = "{:.2f}".format(y)

        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center', va='bottom', 

                     arrowprops=dict(arrowstyle="-", connectionstyle="arc3")) 

    plt.title('{} Training and validation loss'.format(k))

    plt.legend()

    plt.figure()

    plt.show()
# Save Model

torch.save(model, model_dir + 'model.pkl')

torch.save(model.state_dict(), model_dir + 'model_params.pkl')



print(model)
color_list = ['b', 'g', 'r', 'c', 'n']

color_ind = 0

for k, v in model_recorder_dict.items():

    print(k, '\t', v)

    plt.plot(v['epoch_list'], v['valid_acc'][:len(v['epoch_list'])], color_list[color_ind], label=k)

    color_ind += 1



plt.legend()

plt.show()
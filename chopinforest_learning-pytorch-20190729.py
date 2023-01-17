# This Python 3 environment comes with many helpful analytics libraries installed%matplotlib inline

# python libraties

import os, cv2,itertools

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from glob import glob

from PIL import Image



# pytorch libraries

import torch

from torch import optim,nn

from torch.autograd import Variable

from torch.utils.data import DataLoader,Dataset

from torchvision import models,transforms



# sklearn libraries

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



# to make the results are reproducible

np.random.seed(10)

torch.manual_seed(10)

torch.cuda.manual_seed(10)



print(os.listdir("../input"))
from glob import glob

data_dir = '../input'

all_image_path = glob(os.path.join(data_dir, '*', '*.png'))

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

lesion_type_dict = {

    '0': 'No DR',

    '1': 'Mild',

    '2': ' Moderate ',

    '3': 'Severe',

    '4': 'Proliferative DR',

  }
all_image_path[:5]
from tqdm import tqdm

import cv2

def compute_img_mean_std(image_paths):

    """

        computing the mean and std of three channel on the whole dataset,

        first we should normalize the image from 0-255 to 0-1

    """



    img_h, img_w = 224, 224

    imgs = []

    means, stdevs = [], []



    for i in tqdm(range(len(image_paths))):

        img = cv2.imread(image_paths[i])

        img = cv2.resize(img, (img_h, img_w))

        imgs.append(img)



    imgs = np.stack(imgs, axis=3)

    print(imgs.shape)



    imgs = imgs.astype(np.float32) / 255.



    for i in range(3):

        pixels = imgs[:, :, i, :].ravel()  # resize to one row

        means.append(np.mean(pixels))

        stdevs.append(np.std(pixels))



    means.reverse()  # BGR --> RGB

    stdevs.reverse()



    print("normMean = {}".format(means))

    print("normStd = {}".format(stdevs))

    return means,stdevs
#Return the mean and std of RGB channels

norm_mean,norm_std = compute_img_mean_std(all_image_path)
import pandas as pd

import os

data_dir = '../input/'

train_dir = data_dir + '/train_images/'

test_dir = data_dir + '/test_images/'

retina_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

retina_df.head()
retina_df['path'] = retina_df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

retina_df['exists'] = retina_df['path'].map(os.path.exists)

print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')

retina_df.head()
retina_df['diagnosis'].value_counts()
# Copy fewer class to balance the number of 5 classes

#df.loc['a', :]表示选取索引为‘a’的行



data_aug_rate = [0,4,0,5,4]

for i in range(5):

    if data_aug_rate[i]:

        retina_df=retina_df.append([retina_df.loc[retina_df['diagnosis'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)

retina_df['diagnosis'].value_counts()
df_train, df_val = train_test_split(

    retina_df, 

    test_size=0.15, 

    random_state=2019

)

df_train.head()
df_val.head()
df_train = df_train.reset_index()

df_val = df_val.reset_index()
# feature_extract is a boolean that defines if we are finetuning or feature extracting. 

# If feature_extract = False, the model is finetuned and all model parameters are updated. 

# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.

def set_parameter_requires_grad(model, feature_extracting):

    if feature_extracting:

        for param in model.parameters():

            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    # Initialize these variables which will be set in this if statement. Each of these

    #   variables is model specific.

    model_ft = None

    input_size = 0



    if model_name == "resnet":

        """ Resnet18, resnet34, resnet50, resnet101

        """

        model_ft = models.resnet50(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224





    elif model_name == "vgg":

        """ VGG11_bn

        """

        model_ft = models.vgg11_bn(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 224





    elif model_name == "densenet":

        """ Densenet121

        """

        model_ft = models.densenet121(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier.in_features

        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        input_size = 224



    elif model_name == "inception":

        """ Inception v3

        Be careful, expects (299,299) sized images and has auxiliary output

        """

        model_ft = models.inception_v3(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        # Handle the auxilary net

        num_ftrs = model_ft.AuxLogits.fc.in_features

        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs,num_classes)

        input_size = 299



    else:

        print("Invalid model name, exiting...")

        exit()

    return model_ft, input_size
# pytorch libraries

import torch

from torch import optim,nn

from torch.autograd import Variable

from torch.utils.data import DataLoader,Dataset

from torchvision import models,transforms



# resnet,vgg,densenet,inception

model_name = 'densenet'

num_classes = 5

feature_extract = False

# Initialize the model for this run

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Define the device:

device = torch.device('cuda:0')

# Put the model on the device:

model = model_ft.to(device)
#后面运行就不需要再求normMean

#normMean = [0.4452997, 0.2424191, 0.077126786]

#normStd = [0.25739512, 0.14617251, 0.08006624]

# define the transformation of the train images.

train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),

                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),

                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),

                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

# define the transformation of the val images.

val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),

                                    transforms.Normalize(norm_mean, norm_std)])
# Define a pytorch dataloader for this dataset

class APTOS(Dataset):

    def __init__(self, df, transform=None):

        self.df = df

        self.transform = transform



    def __len__(self):

        return len(self.df)



    def __getitem__(self, index):

        # Load data and get label

        X = Image.open(self.df['path'][index])

        y = torch.tensor(int(self.df['diagnosis'][index]))



        if self.transform:

            X = self.transform(X)



        return X, y
# Define the training set using the table train_df and using our defined transitions (train_transform)

training_set = APTOS(df_train, transform=train_transform)

train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)

# Same for the validation set:

validation_set = APTOS(df_val, transform=train_transform)

val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)
# we use Adam optimizer, use cross entropy loss as our loss function

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss().to(device)
# this function is used during training process, to calculation the loss and accuracy

class AverageMeter(object):

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
total_loss_train, total_acc_train = [],[]

def train(train_loader, model, criterion, optimizer, epoch):

    model.train()

    train_loss = AverageMeter()

    train_acc = AverageMeter()

    curr_iter = (epoch - 1) * len(train_loader)

    for i, data in enumerate(train_loader):

        images, labels = data

        N = images.size(0)

        # print('image shape:',images.size(0), 'label shape',labels.size(0))

        images = Variable(images).to(device)

        labels = Variable(labels).to(device)



        optimizer.zero_grad()

        outputs = model(images)



        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        prediction = outputs.max(1, keepdim=True)[1]

        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

        train_loss.update(loss.item())

        curr_iter += 1

        if (i + 1) % 100 == 0:

            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (

                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))

            total_loss_train.append(train_loss.avg)

            total_acc_train.append(train_acc.avg)

    return train_loss.avg, train_acc.avg
def validate(val_loader, model, criterion, optimizer, epoch):

    model.eval()

    val_loss = AverageMeter()

    val_acc = AverageMeter()

    with torch.no_grad():

        for i, data in enumerate(val_loader):

            images, labels = data

            N = images.size(0)

            images = Variable(images).to(device)

            labels = Variable(labels).to(device)



            outputs = model(images)

            prediction = outputs.max(1, keepdim=True)[1]



            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)



            val_loss.update(criterion(outputs, labels).item())



    print('------------------------------------------------------------')

    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))

    print('------------------------------------------------------------')

    return val_loss.avg, val_acc.avg
epoch_num = 10

best_val_acc = 0

total_loss_val, total_acc_val = [],[]

for epoch in range(1, epoch_num+1):

    loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)

    loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)

    total_loss_val.append(loss_val)

    total_acc_val.append(acc_val)

    if acc_val > best_val_acc:

        best_val_acc = acc_val

        print('*****************************************************')

        print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))

        print('*****************************************************')
# 保存整个网络

torch.save(model, 'model.pkl')
fig = plt.figure(num = 2)

fig1 = fig.add_subplot(2,1,1)

fig2 = fig.add_subplot(2,1,2)

fig1.plot(total_loss_train, label = 'training loss')

fig1.plot(total_acc_train, label = 'training accuracy')

fig2.plot(total_loss_val, label = 'validation loss')

fig2.plot(total_acc_val, label = 'validation accuracy')

plt.legend()

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

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
model.eval()

y_label = []

y_predict = []

with torch.no_grad():

    for i, data in enumerate(val_loader):

        images, labels = data

        N = images.size(0)

        images = Variable(images).to(device)

        outputs = model(images)

        prediction = outputs.max(1, keepdim=True)[1]

        y_label.extend(labels.cpu().numpy())

        y_predict.extend(np.squeeze(prediction.cpu().numpy().T))



# compute the confusion matrix

confusion_mtx = confusion_matrix(y_label, y_predict)

# plot the confusion matrix

plot_labels = [' No DR', 'Mild', 'Moderate', 'Severe', ' Proliferative DR']

plot_confusion_matrix(confusion_mtx, plot_labels)
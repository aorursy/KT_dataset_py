# install torchsummary

# Model summary in PyTorch similar to `model.summary()` in Keras

!pip install torchsummary
from __future__ import print_function



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim.lr_scheduler import StepLR



from torchvision import transforms

from torch.utils.data import Dataset, DataLoader



from torchsummary import summary

from tqdm import tqdm



import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10,5)
import tensorflow as tf



device_name = tf.test.gpu_device_name()



try:

    print(f"Found GPU at : {device_name}")

except:

    print("GPU device not found.")
import torch



if torch.cuda.is_available():

    device = torch.device("cuda")

    use_cuda = True

    print(f"Number of GPU's available : {torch.cuda.device_count()}")

    print(f"GPU device name : {torch.cuda.get_device_name(0)}")

else:

    print("No GPU available, using CPU instead")

    device = torch.device("cpu")

    use_cuda = False
import pandas as pd



#read train.csv and test.csv

train_df = pd.read_csv("../input/digit-recognizer/train.csv")

test_df = pd.read_csv("../input/digit-recognizer/test.csv")
#check shape of datasets

train_df.shape, test_df.shape
train_df.head()
from PIL import Image
class MyDataset(Dataset):

    

    def __init__(self, df, train=False, transform=None):

        """

        Args:

            df : pandas dataframe

            train : train=True, dataframe contains additional label column

            transform : function that  takes in an PIL image and returns a transformed version

        """

        self.df = df

        self.transform = transform

        self.train = train

        

        if train:

            self.y_data = torch.tensor(np.asarray(self.df['label'].values))

            self.X_data = np.asarray(self.df.drop('label', axis=1).values).reshape(self.df.shape[0], 28, 28)

        else:

            self.X_data = np.asarray(self.df.values).reshape(self.df.shape[0], 28, 28)

            

        

    def __getitem__(self, index):

        """

        Args:

            index : Index



        Returns:

            tuple: (image, target) where target is index of the target class if available.

        """

        

        image = self.X_data[index]

        image = Image.fromarray(np.uint8(image), mode='L')

        

        if self.transform:

            image = self.transform(image)



        if self.train:

            return image, self.y_data[index]

        else:

            return image

            

    

    def __len__(self):

        return self.df.shape[0]
#training data transformation

train_transforms = transforms.Compose([

                                       transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),

                                       transforms.RandomRotation((-7,7), fill=(1,)),

                                       transforms.ToTensor(),

                                       transforms.Normalize((0.1307,),(0.3081,))

])



#validation data transformation

val_transforms = transforms.Compose([

                                       transforms.ToTensor(),

                                       transforms.Normalize((0.1307,),(0.3081,)),

])



#testing data transformation

test_transforms = transforms.Compose([

                                       transforms.ToTensor(),

                                       transforms.Normalize((0.1307,),(0.3081,)),

])
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'])

train_df.shape, val_df.shape
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
# get train, val and test dataset

train_dataset = MyDataset(train_df, train=True, transform=train_transforms)

val_dataset = MyDataset(val_df, train=True, transform=val_transforms)

test_dataset = MyDataset(test_df, train=False, transform=test_transforms)
# get train, val and test dataloader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, **kwargs)
examples = iter(train_loader)
example_data, example_targets = examples.next()
fig = plt.figure()

for i in range(30):

    plt.subplot(10,3,i+1)

    plt.axis('off')

    plt.imshow(example_data[i][0].numpy().squeeze(), cmap='gray_r')
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        self.conv1block = nn.Sequential(

            nn.Conv2d(1, 8, 3),                            #(-1,28,28,1)>(-1,3,3,1,8)>(-1,26,26,8)>3

            nn.BatchNorm2d(8),

            nn.ReLU(),

            nn.Conv2d(8, 8, 3),                           #(-1,26,26,8)>(-1,3,3,8,8)>(-1,24,24,8)>5

            nn.BatchNorm2d(8),

            nn.ReLU(),

            nn.Conv2d(8, 10, 3),                           #(-1,24,24,8)>(-1,3,3,8,10)>(-1,22,22,10)>7

            nn.BatchNorm2d(10),

            nn.ReLU(),

        )



        self.pool1block = nn.Sequential(

            nn.MaxPool2d(2,2),                              #(-1,22,22,10)>(-1,11,11,10)>8

        )



        self.conv2block = nn.Sequential(

            nn.Conv2d(10, 16, 3),                           #(-1,11,11,10)>(-1,3,3,10,16)>(-1,9,9,16)>12

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Dropout(0.01),

            nn.Conv2d(16, 16, 3),                           #(-1,9,9,16)>(-1,3,3,16,16)>(-1,7,7,16)>16

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Dropout(0.01),

            nn.Conv2d(16, 16, 3),                           #(-1,7,7,16)>(-1,3,3,16,16)>(-1,5,5,16)>20

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Dropout(0.01),

        )



        self.avgpool = nn.AvgPool2d(5)                      #(-1,5,5,16)>(-1,1,1,16)>28

        self.conv3 = nn.Conv2d(16, 10, 1)                   #(-1,1,1,16)>(-1,1,1,16,10)>(-1,1,1,10)>28  



        

    def forward(self, x):

        x = self.conv1block(x)

        x = self.pool1block(x)

        x = self.conv2block(x)

        x = self.avgpool(x)

        x = self.conv3(x)

        x = x.view(-1, 10)

        return F.log_softmax(x)
model = Net().to(device)

summary(model, input_size=(1, 28, 28))
def regularize_loss(model, loss, decay, norm_value):

    """

    L1/L2 Regularization

    decay : l1/l2 decay value

    norm_value : the order of norm

    """

    r_loss = 0

    # get sum of norm of parameters

    for param in model.parameters():

        r_loss += torch.norm(param, norm_value)

    # update loss value

    loss += decay * r_loss



    return loss
def save_ckp(state, checkpoint_fpath):

    """

    state: checkpoint we want to save

    checkpoint_path: path to save model

    """

    f_path = checkpoint_fpath

    # save checkpoint data to the path given, checkpoint_path

    torch.save(state, f_path)

    

def load_ckp(checkpoint_fpath, model, optimizer):

    """

    checkpoint_path: path to save checkpoint

    model: model that we want to load checkpoint parameters into       

    optimizer: optimizer we defined in previous training

    """

    # load check point

    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model

    model.load_state_dict(checkpoint['state_dict'])

    # initialize optimizer from checkpoint to optimizer

    optimizer.load_state_dict(checkpoint['optimizer'])

    # get epoch

    epoch = checkpoint['epoch']

    # get val_max_acc

    val_max_acc = checkpoint['valid_max_acc']

    # get misclassified images

    misclassified_images = checkpoint['misclassified_images']

    # return model, optimizer, epoch, val_max_acc, misclassified_images

    return model, optimizer, epoch, val_max_acc, misclassified_images
from tqdm import tqdm

import numpy as np



class Model:

    def __init__(self, model, device, optimizer, l1_decay=0.0, l2_decay=0.0):

        """

        Args:

            model : network

            device : cuda or cpu

            optimizer : defined optimizer function

            l1_decay : regularization, l1 decay value

            l2_decay : regularization, l2 decay value

        """

        self.model = model

        self.device = device

        self.optimizer = optimizer



        self.train_losses = [] # store train losses per epoch

        self.val_losses = [] # store val losses per epoch

        self.train_acc = [] # store train acc per epoch

        self.val_acc = [] # store val acc per epoch



        self.misclassified_images = {} # store misclassified images for the best model



        self.l1_decay = l1_decay # l1 weight decay value

        self.l2_decay = l2_decay # l2 weight decay value



        self.minimum_test_loss = np.Inf

        self.maximum_test_acc = 0.0





    def train(self, train_loader, epoch, checkpoint_fpath = None):

        """

        Args:

            train_loader : pytorch train loader

            epoch : epoch value

            checkpoint_fpath : file path to store state dict of model

        """

        running_loss = 0.0

        running_correct = 0



        self.model.train()



        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            

            self.optimizer.zero_grad()

            

            output = self.model(data)

            loss = F.nll_loss(output, target)



            if self.l1_decay>0.0:

                loss += regularize_loss(self.model, loss, self.l1_decay, 1)

            if self.l2_decay>0.0:

                loss += regularize_loss(self.model, loss, self.l2_decay, 2)



            _, preds = torch.max(output.data, 1)

            loss.backward()

            self.optimizer.step()



            #calculate training running loss

            running_loss += loss.item()

            running_correct += (preds == target).sum().item()

            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')



        r_total_loss = running_loss/len(train_loader.dataset)

        r_total_acc = 100. * running_correct/len(train_loader.dataset)



        self.train_losses.append(r_total_loss)

        self.train_acc.append(r_total_acc)

        print("\n")

        print(f"  TRAIN avg loss: {r_total_loss:.4f} train acc: {r_total_acc:.4f}\n")



    def test(self, val_loader, epoch, checkpoint_fpath = None):

        """

        Args:

            val_loader : pytorch validation loader 

            epoch : epoch value

            checkpoint_fpath : file path to store state dict of best model

        """

        running_loss = 0.0

        running_correct = 0



        self.model.eval()

      

        with torch.no_grad():

            for data, target in val_loader:

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                running_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability



                is_correct = pred.eq(target.view_as(pred))

                misclass_indx = (is_correct==0).nonzero()[:,0]

                

                for indx in misclass_indx:

                

                    if str(epoch) not in self.misclassified_images:

                        self.misclassified_images[str(epoch)] = []

                

                    self.misclassified_images[str(epoch)].append({

                        "target" : target[indx],

                        "pred" : pred[indx],

                        "img" : data[indx]

                    })



                running_correct += pred.eq(target.view_as(pred)).sum().item()



        r_total_loss = running_loss/len(val_loader.dataset)

        r_total_acc = 100.*running_correct/len(val_loader.dataset)



        # check and save best model

        if(r_total_acc>=self.maximum_test_acc):

            self.maximum_test_acc = r_total_acc

            if checkpoint_fpath:

                self.save_checkpoint(epoch, checkpoint_fpath)

                print(f"  Best Model Saved!!!\n")

            else:

                print(f"  Couldn't save the model. Path not defined!!!\n")



        self.val_losses.append(r_total_loss)

        self.val_acc.append(r_total_acc)

        

        print("\n")

        print(f"  TEST avg loss: {r_total_loss:.4f} test acc: {r_total_acc:.4f}\n")





    def save_checkpoint(self, epoch, checkpoint_fpath):

        """

        Args:

            epoch : epoch value

            checkpoint_fpath : file path to store state dict of model

        """

        checkpoint = {

          'epoch' : epoch,

          'misclassified_images' : self.misclassified_images[str(epoch)],

          'valid_max_acc': self.maximum_test_acc,

          'state_dict': self.model.state_dict(),

          'optimizer': self.optimizer.state_dict(),

        }

        save_ckp(checkpoint, checkpoint_fpath)
import json

import os



def run(model_name, l1_decay, l2_decay):

    """

    Args:

        model_name : (string) model name

        l1_decay : l1 regularization value

        l2_decay : l2 regularization value

    """

    

    # initialize network

    net = Net().to(device)

    

    # initialize optimizer with a scheduler

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    

    # number of epochs

    EPOCHS = 25

    

    # initialize our Model parameters

    model = Model(net, device, optimizer, l1_decay, l2_decay)

    

    # define path for storing state dict for network

    MODEL_PATH = "./weights/model_{}.pt".format(model_name)

    

    # TRAINING AND EVALUATION PROCESS

    for epoch in range(1, EPOCHS+1):

        print(f"\nEPOCH : {epoch}\n")

        model.train(train_loader, epoch, MODEL_PATH)

        scheduler.step()

        model.test(val_loader, epoch, MODEL_PATH)



    # get train/test losses and accuracy scores for each epoch

    result = {f'{model_name}':{'train_losses':model.train_losses, 'val_losses':model.val_losses,

                            'train_acc':model.train_acc, 'val_acc':model.val_acc}}



  

    # store the result

    file_path = "./stats/acc_losses_{}.json".format(model_name)

    if not os.path.exists(file_path):

        with open(file_path, "w") as f:

            json.dump({'status':200}, f)

    

    with open(file_path) as f:

        data = json.load(f)

    

    data.update(result)

    

    with open(file_path, "w") as f:

        json.dump(data, f)
# create new directories to store state_dict and stats of our model

!mkdir stats

!mkdir weights
run("mnist_model_03", 0.0, 0.0005)
def get_misclassified_images(model_name):

    """

    Args:

        model_name : (string) model name used to store the model state_dict

    Returns:

        list : misclassified images

    """

    net = Net().to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    

    ckp_path = f"./weights/model_{model_name}.pt"

    

    model, optimizer, epoch, val_max_acc, misclassified_images = load_ckp(ckp_path, net, optimizer)

    

    return misclassified_images





def validation_stat(name, type_='acc'):

    """

    Args:

        name : file path for storing stats dict

        type : (string) acc / loss

    Returns:

        list : accuracy / loss scores for the model

    """

    

    with open(stats_file_path) as f:

        data = json.load(f)

        

    if type_=="acc":

        return data[f"{name}"]["val_acc"] 

    else:

        return data[f"{name}"]["val_losses"]
epoch_count = range(1, 26)



# plot accuracy or loss graph

def plot_graphs(name, type_='acc'):

    """

    Args:

        name : (string) model name

    """

    fig = plt.figure(figsize=(10,10))

    plt.plot(epoch_count, validation_stat(name, type_))

    

    plt.legend(name)

    plt.xlabel('Epoch')

    plt.ylabel(type_)

    plt.show();
# plot misclassified images



def plot_misclassified_images(name, n=25):

    """

    Args:

        name : (string) model name

    """

    figure = plt.figure(figsize=(10,10))

    

    images = get_misclassified_images(name)[:n]

    

    for i in range(1, n+1):

        plt.subplot(5,5,i)

        plt.axis('off')

        plt.imshow(images[i-1]["img"].cpu().numpy()[0], cmap='gray_r')

        plt.title("Predicted : {} \nActual : {}".format(images[i-1]["pred"][0].cpu().numpy(), images[i-1]["target"].cpu().numpy()))



    plt.tight_layout()
# model name

name = "mnist_model_03"

# stats file path where we have stores accuracy and loss scores

stats_file_path = "./stats/acc_losses_{}.json".format(name)
# plot accuracy graph

plot_graphs(name, 'acc')
# plot loss graph

plot_graphs(name, 'losses')
# plot misclassified images for our best model

plot_misclassified_images(name)
def test(model, optimizer, test_loader, checkpoint_fpath = None):

    """

    Args:

        model : network

        test_loader : pytorch test loader

        checkpoint_fpath : file path where state_dict of our network is stored

    Returns:

        list : predicted values

    """

    

    model, optimizer, epoch, val_max_acc, misclassified_images = load_ckp(checkpoint_fpath, model, optimizer)

    model.eval()

    preds = []

    

    with torch.no_grad():

        for data in test_loader:

            data = data.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)

            preds.extend(pred.cpu().numpy())

            

    return preds
# initialize our network

model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)



# get predictions

predictions = test(model, optimizer, test_loader, "./weights/model_mnist_model_03.pt")
# check size of predictions

len(predictions)
# read submission file

submission_df = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

submission_df.head(), submission_df.shape
# store predictions in Label column

submission_df['Label'] = predictions
# some preprocessing before submission

submission_df['Label'] = submission_df.Label.apply(lambda x : x[0])

submission_df.head()
submission_df.to_csv("submission_v03.csv", index=False)
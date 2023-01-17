!pip install torchsummary
!pip install --upgrade pip
!pip install Cython
# DATA LOADING & PRE-PROCESSING
import numpy as np
import pandas as pd
import os
import random
import pandas_profiling
from sklearn.model_selection import train_test_split

## VISUALIZATION
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

## DATA AUGMENTATION
import albumentations as A
import albumentations.pytorch as AP

## NEURAL NETWORK 
import torch
import torch.optim
import torchvision
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.utils.data import Dataset, random_split

train_df = pd.read_csv('../input/digit-recognizer/train.csv')
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
sample_df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train_df.describe()
num_trainsamples = len(train_df)
num_classes = len(set(train_df['label']))

num_testsamples = len(test_df)



print('Number of Training Samples', num_trainsamples)
print('Number of Test Samples', num_testsamples)

print('Number of Classes', num_classes)
mnist_grid = make_grid(torch.Tensor((train_df.iloc[[1,2,3,4,5,6,7,8], 1:].values/255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)
plt.rcParams['figure.figsize'] = (8, 1)
plt.imshow(mnist_grid.numpy().transpose(1,2,0))
plt.axis('off')
print(*list(train_df.iloc[np.random.randint(num_trainsamples, size=8), 0].values), sep = ', ')
labelcount = train_df['label'].value_counts()

with plt.style.context('fivethirtyeight'):
    plt.rcParams['figure.figsize']=(10,8)
    plt.bar(train_df['label'].value_counts().index, train_df['label'].value_counts(), color ='blue')
    plt.xlabel('Labels', fontsize = 13)
    plt.ylabel('Count', fontsize = 13)
    plt.xticks(train_df['label'].value_counts().index),
    plt.grid('on', axis='y', linestyle = '--', color='black')

# Split the Train Dataset
train_data = train_df.drop(columns='label').values.reshape((-1, 28, 28, 1)).astype(np.uint8)#[:,:,:,None]
train_label = train_df.label.values

# Test Dataset
test_data = test_df.values.reshape((-1,28,28,1)).astype(np.uint8)#[:,:,:,None]

# Split the Train Data into Train & Validation
train_data, valid_data, train_y, valid_y = train_test_split(train_data, train_label, test_size=0.3)


class MNIST(Dataset):
    """
    Custom MNIST Dataset
    """
    
    def __init__(self, data, labels=None,  image_transforms=[]):
        """
        Args:
            data = numpy array of size(28x28x1) 
            labels = numpy array 
            image_transforms = List of Image Transformations
        """
        self.data = data
        self.labels = labels
        self.dataset = 'train' if self.labels is None else 'test'
        self.image_transforms = image_transforms
            
    
    def __len__(self):
        return len(self.data)
            
    
    def __getitem__(self, idx):
        if self.labels is not None: # TRAIN
            return  self.image_transforms( self.data[idx]), self.labels[idx]
        else: # TEST
            return self.image_transforms( self.data[idx])
        
   

        
class DataLoaders:
  def __init__(self, 
              batch_size=512,
              shuffle=True,
              num_workers=4,
              pin_memory=True,
              seed=1):
  
    """
    Arguments:-
    batch_size: Number of images to be passed in each batch
    shuffle(boolean):  If True, then shuffling of the dataset takes place
    num_workers(int): Number of processes that generate batches in parallel
    pin_memory(boolean):
    seed: Random Number, this is to maintain the consistency
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')  # set device to cuda

    if use_cuda:
        
      torch.manual_seed(seed)
    
    self.dataLoader_args = dict(batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True ) if use_cuda else dict(batch_size=1, shuffle=True, num_workers = 1, pin_memory = True)


  def dataLoader(self, data):
    return torch.utils.data.DataLoader(data,**self.dataLoader_args)



def Data_To_Dataloader(trainset, testset, seed=1,batch_size=(128,512), num_workers=4, pin_memory=True):
    SEED = 1
    
    trainBS, validBS = batch_size
    cuda = torch.cuda.is_available()# CUDA?
    torch.manual_seed(SEED)  # For reproducibility

    if cuda:
        torch.cuda.manual_seed(SEED)
        
    trainloader_args = dict(shuffle=True, batch_size=trainBS, num_workers=num_workers, pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=128)
    validloader_args = dict(shuffle=True, batch_size=validBS, num_workers=num_workers, pin_memory=pin_memory) if cuda else dict(shuffle=True, batch_size=128)

    trainloader = torch.utils.data.DataLoader(trainset, **trainloader_args)
    testloader = torch.utils.data.DataLoader(testset, **validloader_args)

    return  trainloader, testloader
train_transforms = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Pad(6),
                                       transforms.RandomCrop(28),
                                       transforms.RandomRotation(7.0),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.1307,), std=(0.3081,))])


test_transforms = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

# :29400] [29400:42000]
# creating train, validation & test set
train_set = MNIST(train_data, train_y, image_transforms=train_transforms)
validatn_set = MNIST(valid_data, valid_y, image_transforms=test_transforms)
test_set = MNIST(test_data, image_transforms=test_transforms)
trainLoader, validLoader = Data_To_Dataloader(train_set, validatn_set, seed=1, batch_size=(128,512), num_workers=4, pin_memory=True)

# dataloader object to convert testset to testloader
d1 = DataLoaders() # dummy dataloader obj
testLoader = d1.dataLoader(test_set)
    
def plot_sample_data(img_transforms=[], row=10):
    num = len(img_transforms)
    img = train_df.iloc[row, 1:].values.reshape((28,28)).astype(np.uint8)[:,:,None]
    fig = plt.figure(figsize=(8,10))
    for i, transform in enumerate(img_transforms):
        transformed_img = transform( transforms.ToPILImage()(img))
        ax = plt.subplot(1, num, i+1)
        plt.tight_layout()
        plt.axis('off')
        ax.set_title(type(transform).__name__)
        ax.imshow(np.reshape(np.array(list(transformed_img.getdata())), (-1,28)), cmap='gray')
    plt.show()

img_transforms = [ transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                  transforms.RandomCrop(28), 
                  transforms.RandomHorizontalFlip(p=1),
                  transforms.Compose([ transforms.RandomRotation((-7.0, 7.0), fill=(1,)), transforms.RandomCrop(28)])]
plot_sample_data(img_transforms, row=21)
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Dropout2d(p=0.25),
                                    
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Dropout2d(p=0.25),
                                   
                                   
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2,2))) # 14
            
        self.avgpool = nn.AvgPool2d(14)
        
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x) # 14

        x = self.avgpool(x) # 1
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=-1)
        return x

mnist = MnistNet()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
mnist_model = mnist.to(device)

summary(mnist_model, input_size=(1, 28, 28))

# MODEL CONFIG
class ModelConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# MNIST NET
sgd_optimizer = torch.optim.SGD(mnist_model.parameters(), lr=0.01, momentum=0.95, nesterov=True)
reducelr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer)



model_config = ModelConfig(model=mnist_model, device=device, trainloader=trainLoader, 
                           testloader=validLoader, optimizer=sgd_optimizer,criterion = F.nll_loss, epochs=35)
class Trainer:
    def __init__(self, config):
        self.model = config.model
        self.device = config.device
        self.train_loader = config.trainloader
        self.test_loader = config.testloader
        self.optimizer = config.optimizer
        self.loss_criterion = config.criterion
        self.epoch = config.epochs
        
        # train & test variables
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        
        # samples
        self.correct_samples = []
        self.incorrect_samples = []
        
        # counter
        self.count = 0
        
    def train(self):
        self.count += 1
        self.lr = 0
        self.train_loss_value = 0.0
        self.correct_predictions = 0
        
        self.model.train()
        pbar = tqdm(self.train_loader)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
  
            self.optimizer.zero_grad()
            pred_value = self.model(data)
            
            loss = F.nll_loss(pred_value, target)
            loss.backward()
            self.optimizer.step()
            
            # train variables
            predictions = pred_value.argmax(dim=1, keepdim=True)
            self.correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()
            self.train_loss_value += self.loss_criterion(pred_value, target).item()
            
       
        # calculate loss and acc for each batch
        self.train_loss_value /= len(self.train_loader.dataset)
        self.train_acc_value = self.correct_predictions / len(self.train_loader.dataset)
         
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
           
        # append the same
        self.train_loss.append(self.train_loss_value)
        self.train_acc.append(self.train_acc_value*100)
        
        return round(self.train_acc_value*100,4), round(self.train_loss_value,4), round(self.lr)

    def test(self):
        self.model.eval()
        self.test_loss_value = 0.0
        self.correct_predictions = 0
        pbar = tqdm(self.test_loader)
        with torch.no_grad():
            for data, target in self.test_loader:
                img_batch = data
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                self.test_loss_value += self.loss_criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                self.correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                
                result = pred.eq(target.view_as(pred))

                if len(self.incorrect_samples) < 25:
                  for i in range(0, self.test_loader.batch_size):
                    if not list(result)[i]:
                      self.incorrect_samples.append({'prediction': list(pred)[i], 'label': list(target.view_as(pred))[i],'image': img_batch[i]})
                                        
                # to extract correct samples/classified images
                if len(self.correct_samples) < 25:
                  for i in range(0, self.test_loader.batch_size):
                    if  list(result)[i]:
                      self.correct_samples.append({'prediction': list(pred)[i], 'label': list(target.view_as(pred))[i],'image': img_batch[i]})
                    
               
        self.test_loss_value = self.test_loss_value/ len(self.test_loader.dataset)
        self.test_acc_value = self.correct_predictions / len(self.test_loader.dataset)
        self.test_loss.append(self.test_loss_value)
        self.test_acc.append(self.test_acc_value*100)
        
        return round(self.test_acc_value*100,4), round(self.test_loss_value,4), self.incorrect_samples, self.correct_samples
        
        
        
def evaluate(configs):
    t1 = Trainer(configs)
    for epoch in range(1, t1.epoch+1):
        print('-'*35)
        print('EPOCH:', epoch)
        train_acc, train_loss, lr = t1.train()
       
        test_acc, test_loss, incorrect_samples, correct_samples =t1.test()
        reducelr_scheduler.step(test_loss)
        print(f' Train Set: Loss:{train_loss} | Accuracy:{train_acc} || Validatn Set: Loss:{test_loss} | Accuracy:{test_acc}')
             

    return t1.test_acc, t1.train_acc, t1.test_loss, t1.train_loss, lr, incorrect_samples, correct_samples

test_acc, train_acc, test_loss, train_loss, lr, incorrect_samples, correct_samples = evaluate(model_config)
# Training & Validation Curves

def plot_curve(elements, title, y_label = 'Accuracy', Figsize = (8,8)):
    """
    elements: Contains Training and Testing variables of the Model like Accuracy or Loss
    title: Plot title
    y_label: Y-axis Label, Accuracy by default
    FigSize: Size of the Plot
    """
    with plt.style.context('fivethirtyeight'):
        fig = plt.figure(figsize=Figsize)
        ax = plt.subplot()
        for elem in elements:
            ax.plot(elem[0], label=elem[1])
            ax.set(xlabel='Epochs', ylabel=y_label)
            plt.title(title)
        ax.legend()
    plt.show()

accuracyElements = [(test_acc,"Validation Accuracy"),(train_acc,"Train Accuracy")]
lossElements = [(test_loss,"Validation Loss"),(train_loss,"Train Loss")]

for i in range(1):
  plot_curve(lossElements,'Training & Validation Loss of MnistNet', y_label='Loss')
  plot_curve(accuracyElements,'Training & Validation Accuracy of MnistNet')
def plot_modelvar(test_var, train_var, plt_title, vartype='Loss'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(1, len(test_var)), y=test_var, mode='lines+markers', name='test'))
#     ?                        marker=dict(color="indianred", line=dict(width=.5,color='rgb(0, 0, 0)'))))


    fig.add_trace(go.Scatter(x=np.arange(1, len(train_var)), y=train_var, mode='lines+markers', name='train'))
    fig.update_layout(xaxis_title="Epochs", yaxis_title=vartype,
                      title_text=str(vartype)+" vs. Epochs", template="plotly_white", paper_bgcolor="#f0f0f0", title=str(plt_title))
    fig.show()
    


plot_modelvar(test_loss, train_loss, 'Train & Validation Losses of MnistNet')
 
plot_modelvar(test_acc, train_acc,  'Train & Validation Accuracy of MnistNet', vartype='Accuracy')

# Confusion Matrix for the validation results
def test_images(model, testLoader, data, filename):
  """
  Arguments:
    model(str): ModelName
    testLoader: Data Loader for Test Images
    data(list): Misclassified/Correctly Classified Images
    filename(str): Return Image Save as
  """
  classes = (0,1,2,3,4,5,6,7,8,9) # class names in the dataset

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model = model.to(device)
  dataiter = iter(testLoader)
  count = 0

  # Initialize plot
  fig = plt.figure(figsize=(13,13))

  row_count = -1
  fig, axs = plt.subplots(5, 5, figsize=(10, 10))
  fig.tight_layout()

  for idx, result in enumerate(data):
    # If 25 samples have been stored, break out of loop
    if idx > 24:
      break

    rgb_image = np.transpose(result['image'], (1, 2, 0)) / 2 + 0.5
    label = result['label'].item()
    prediction = result['prediction'].item()

    # Plot image
    if idx % 5 == 0:
      row_count += 1

    axs[row_count][idx % 5].axis('off')
    axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
    axs[row_count][idx % 5].imshow(rgb_image.reshape(28,28))

  # save the plot
    plt.savefig(filename)

test_images(mnist_model, testLoader, incorrect_samples, 'misclassified.png')
test_images(mnist_model, testLoader, correct_samples, 'classified.png')
def predict_results(model, test_loader, device):
    
    model.eval()
    predictions = sample_df['Label'].values
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1).to('cpu').numpy()
            predictions[i*512:i*512+len(inputs)] = pred
    
    output = sample_df.copy()
    output['Label'] = predictions
    output.to_csv('submission4.csv', index=False)
    return output

output = predict_results(mnist_model, testLoader, device)
output
a=output.groupby('Label').count()
a
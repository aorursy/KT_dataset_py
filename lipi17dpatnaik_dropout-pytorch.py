# Import libraries
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

%matplotlib inline
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Image transformations
train_transformation = transforms.Compose([transforms.RandomResizedCrop(64),   #create 64x64 image
                                    transforms.RandomHorizontalFlip(),    #flipping the image horizontally
                                    transforms.ToTensor(),                 #convert the image to a Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  #normalize the image

test_transformation = transforms.Compose([transforms.RandomResizedCrop(64),   #create 64x64 image
#                                    transforms.RandomHorizontalFlip(),    #flipping the image horizontally
                                    transforms.ToTensor(),                 #convert the image to a Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  #normalize the image
!ls ../input/alien-vs-predator-images/data/train/
!ls ../input/alien-vs-predator-images/data/validation/
# Load training dataset
# We are using ImageFolder because of the folder structure
train_dataset = datasets.ImageFolder(root = '../input/alien-vs-predator-images/data/train/',
                                     transform = train_transformation)

# Load validation dataset
test_dataset = datasets.ImageFolder(root = '../input/alien-vs-predator-images/data/validation/',
                                    transform = test_transformation)
batch_size = 4
# Create a data loader for loading training dataset
train_load = torch.utils.data.DataLoader(dataset = train_dataset, 
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers=0
                                        )

# Create a data loader for loading testing dataset
test_load = torch.utils.data.DataLoader(dataset = test_dataset, 
                                         batch_size = batch_size,
                                         shuffle = False,
                                       num_workers=0
                                       )
#Show a batch of images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
# get some random training images
dataiter = iter(train_load)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
print("# of images in training dataset: {}".format(len(train_dataset)))
print("# of images in validation dataset: {}".format(len(test_dataset)))
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)        #Batch normalization
        self.relu = nn.ReLU()                 #RELU Activation
        self.lrelu = nn.LeakyReLU()           #Leaky ReLU activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)   #Maxpooling reduces the size by kernel size. 64/2 = 32
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    #Size now is 32/2 = 16
        
        #Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        self.fc1 = nn.Linear(in_features=8192, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
        self.droput = nn.Dropout(p=0.5)                    #Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)    #Since there were so many features, I decided to use 45 layers to get output layers. You can increase the kernels in Maxpooling to reduce image further and reduce number of hidden linear layers.
       
        
    def forward(self,x):
        out = self.cnn1(x)
        #out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        #out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1,8192)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        #Then we forward through our fully connected layer 
        out = self.fc1(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc5(out)
        return out
class CNN_dropout(nn.Module):
    def __init__(self):
        super(CNN_dropout,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)        #Batch normalization
        self.relu = nn.ReLU()                 #RELU Activation
        self.lrelu = nn.LeakyReLU()           #Leaky ReLU activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)   #Maxpooling reduces the size by kernel size. 64/2 = 32
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    #Size now is 32/2 = 16
        
        #Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        self.fc1 = nn.Linear(in_features=8192, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
        self.droput = nn.Dropout(p=0.5)                    #Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)    #Since there were so many features, I decided to use 45 layers to get output layers. You can increase the kernels in Maxpooling to reduce image further and reduce number of hidden linear layers.
       
        
    def forward(self,x):
        out = self.cnn1(x)
        #out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        #out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1,8192)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        #Then we forward through our fully connected layer 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
        #self.batchnorm1 = nn.BatchNorm2d(8)        #Batch normalization
        self.relu = nn.ReLU()                 #RELU Activation
        self.lrelu = nn.LeakyReLU()           #Leaky ReLU activation
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)   #Maxpooling reduces the size by kernel size. 64/2 = 32
        
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        #self.batchnorm2 = nn.BatchNorm2d(32)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2)    #Size now is 32/2 = 16
        
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        #self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    #Size now is 32/2 = 16
        
        #Flatten the feature maps. You have 32 feature mapsfrom cnn4. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        self.fc1 = nn.Linear(in_features=8192, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.fc4 = nn.Linear(in_features=500, out_features=100)
        self.fc5 = nn.Linear(in_features=100, out_features=50)
        self.fc6 = nn.Linear(in_features=50, out_features=2)    
       
        
    def forward(self,x):
        out = self.cnn1(x)
        #out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.cnn2(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn3(out)
        #out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.cnn4(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1,8192)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        #Then we forward through our fully connected layer 
        out = self.fc1(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        #out = self.droput(out)
        out = self.fc6(out)
        return out
import time
def train_model(lr, weight_decay=0, num_epochs = 200, model=CNN()):
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.to('cuda')
    loss_fn = nn.CrossEntropyLoss() 
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=weight_decay)

    #Define the lists to store the results of loss and accuracy
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    
    # Training the CNN
    for epoch in range(1,num_epochs+1): 
        #Reset these below variables to 0 at the begining of every epoch
        start = time.time()
        correct = 0
        iterations = 0
        iter_loss = 0.0

        model.train()                   # Put the network into training mode

        for i, (inputs, labels) in enumerate(train_load):

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            optimizer.zero_grad()            # Clear off the gradient in (w = w - gradient)
            outputs = model(inputs)         
            loss = loss_fn(outputs, labels)  
            iter_loss += loss.item()       # Accumulate the loss
            loss.backward()                 # Backpropagation 
            optimizer.step()                # Update the weights

            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1

        # Record the training loss
        train_loss.append(iter_loss/iterations)
        # Record the training accuracy
        train_accuracy.append((correct.item() / len(train_dataset)))

        #Testing
        loss = 0.0
        correct = 0
        iterations = 0
        iter_loss = 0.0

        
        with torch.no_grad():
            
            model.eval()                    # Put the network into evaluation mode

            for i, (inputs, labels) in enumerate(test_load):

                CUDA = torch.cuda.is_available()
                if CUDA:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                outputs = model(inputs)     
                loss = loss_fn(outputs, labels) # Calculate the loss
                iter_loss += loss.item()
                # Record the correct predictions for training data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()

                iterations += 1

        # Record the Testing loss
        test_loss.append(iter_loss/iterations)
        # Record the Testing accuracy
        test_accuracy.append((correct.item() / len(test_dataset)))
        stop = time.time()

        if (epoch%10 == 0):
            print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {:.3f}s'
                   .format(epoch, num_epochs, train_loss[-1], train_accuracy[-1], 
                           test_loss[-1], test_accuracy[-1], stop-start))
            
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    train_accuracy = np.array(train_accuracy)
    test_accuracy = np.array(test_accuracy)
    return model,train_loss,test_loss,train_accuracy,test_accuracy
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, labels, colors,
                       loss_legend_loc='upper center', acc_legend_loc='upper left', legend_font=15,
                       fig_size=(16, 8), sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2)):
    
    plt.rcParams["figure.figsize"] = fig_size
    plt.figure
    
    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])
    
    for i in range(len(train_loss)):
        x_train = range(len(train_loss[i]))
        x_val = range(len(val_loss[i]))
        
        min_train_loss = train_loss[i].min()
        
        min_val_loss = val_loss[i].min()
        
        plt.plot(x_train, train_loss[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN ({0:.4}): {1}".format(min_train_loss, labels[i]))
        plt.plot(x_val, val_loss[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID ({0:.4}): {1}".format(min_val_loss, labels[i]))
        
    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc=loss_legend_loc, prop={'size': legend_font})
    plt.title('Training and Validation Loss')
        
    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])
    
    for i in range(len(train_acc)):
        x_train = range(len(train_acc[i]))
        x_val = range(len(val_acc[i]))
        
        max_train_acc = train_acc[i].max() 
        
        max_val_acc = val_acc[i].max() 
        
        plt.plot(x_train, train_acc[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN ({0:.4}): {1}".format(max_train_acc, labels[i]))
        plt.plot(x_val, val_acc[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID ({0:.4}): {1}".format(max_val_acc, labels[i]))
        
    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc=acc_legend_loc, prop={'size': legend_font})
    plt.title('Training and Validation Accuracy')
    
    plt.show()
    
    return
# Without dropout
model_no_dropout,train_loss_1,val_loss_1,train_acc_1,val_acc_1 = train_model(lr = 0.01, weight_decay=0, num_epochs = 500, model=CNN())
plot_loss_accuracy(train_loss=[train_loss_1], 
                   val_loss=[val_loss_1], 
                   train_acc=[train_acc_1], 
                   val_acc=[val_acc_1], 
                   labels=['No Dropout'], 
                   colors=['blue'], 
                   loss_legend_loc='upper center', 
                   acc_legend_loc='upper left')
# With dropout
model_dropout,train_loss_2,val_loss_2,train_acc_2,val_acc_2 = train_model(lr = 0.01, weight_decay=0, num_epochs = 500, model=CNN_dropout())
plot_loss_accuracy(train_loss=[train_loss_1, train_loss_2], 
                   val_loss=[val_loss_1, val_loss_2], 
                   train_acc=[train_acc_1, train_acc_2], 
                   val_acc=[val_acc_1, val_acc_2], 
                   labels=['No Dropout', 'Dropout'], 
                   colors=['blue', 'orange'], 
                   loss_legend_loc='upper left', 
                   acc_legend_loc='best', 
                   legend_font = 12,
                   fig_size=(12, 24), 
                   sub_plot1=(2, 1, 1), 
                   sub_plot2=(2, 1, 2))
plot_loss_accuracy(train_loss=[train_loss_2], 
                   val_loss=[val_loss_2], 
                   train_acc=[train_acc_2], 
                   val_acc=[val_acc_2], 
                   labels=['Dropout'], 
                   colors=['blue'],  
                   loss_legend_loc='upper center', 
                   acc_legend_loc='upper left')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
import scipy.stats as stats
def plot_normal_dist(mu, sigma):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)
    return (x,y)
def plot_params(model_no_dropout, model_dropout):
    layer_num = 1
    for param,param2 in zip(model_dropout.parameters(),model_no_dropout.parameters()):
        param = param.data.cpu().numpy()
        param2 = param2.data.cpu().numpy()
        param = param.reshape(1,-1)[0]
        param2 = param2.reshape(1,-1)[0]
        mu2 = np.mean(param2)
        sigma2 = np.std(param2)
        print("No L2 -- Layer: {} -- Std dev: {} -- Mean: {}".format(layer_num,sigma2,mu2))
        mu = np.mean(param)
        sigma = np.std(param)
        print("With L2 -- Layer: {} -- Std dev: {} -- Mean: {}".format(layer_num,sigma,mu))
        plt.figure(figsize=(10,5))
        plt.title("Layer#{}".format(layer_num))
        x2,y2 = plot_normal_dist(mu2,sigma2)
        x,y = plot_normal_dist(mu,sigma)
        plt.plot(x,y,color="blue",label="Dropout")
        plt.plot(x2,y2,color="red", label="No Dropout")
        plt.legend()
        plt.show()
        layer_num += 1
plot_params(model_no_dropout,model_dropout)
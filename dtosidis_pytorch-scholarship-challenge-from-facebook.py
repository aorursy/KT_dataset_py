# Imports here
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os # accessing directory structure

from torch import nn, optim
from subprocess import check_output
from torchvision import datasets, transforms, models
print(os.listdir('../input'))
os.chdir('../input')
print(check_output(["ls", "../input"]).decode("utf8"))
data_dir = '../input/flower-data/flower_data/flower_data'
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
valid_dir = data_dir + '/valid'
# TODO: Define your transforms for the training and validation sets
data_transforms =  transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
traindatasets = datasets.ImageFolder(train_dir,transform=data_transforms)
testdatasets = datasets.ImageFolder(test_dir,transform=data_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(traindatasets, batch_size=32, shuffle=True)
testloaders = torch.utils.data.DataLoader(testdatasets, batch_size=32, shuffle=True)
images, labels = next(iter(trainloaders))
print(type(images))
print(images.shape)
print(labels.shape)
images, labels = next(iter(testloaders))
print(type(images))
print(images.shape)
print(labels.shape)

import json
print(os.listdir('../input/flowerjsonmap'))
with open('../input/flowerjsonmap/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# TODO: Build and train your network
model = models.vgg16_bn(pretrained = True)
print(model)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
                      nn.Linear(25088, 4096, bias=True),
                      nn.ReLU(),
                      nn.Dropout(p=0.5),
                      nn.Linear(4096, 4096, bias=True),
                      nn.ReLU(),
                      nn.Dropout(p=0.5),
                      nn.Linear(4096, 102, bias=True),
                      nn.LogSoftmax(dim=1))
# Define the loss
device='cuda'
model.cuda()
criterion = nn.NLLLoss()

from torch import optim

# Optimizers require the parameters to optimize and a learning rate
#optimizer = optim.SGD(model.parameters(), lr=0.003)
optimizer = optim.Adam(model.parameters(), lr = 0.003)
#print('Initial weights - ', model[0].weight)
epochs = 1#30
train_losses, test_losses = [], []
for e in range(epochs):
    print('Epoch {}/{}'.format(e, epochs - 1))
    # Each epoch has a training and validation phase
    running_loss = 0
    running_corrects = 0
    for images, labels in trainloaders:#[phase]:
        inputs = images.to(device)
        labels = labels.to(device)
        # TODO: Training pass
        optimizer.zero_grad()
        # forward
        # track history if only in train
        outputs = model(inputs)
        loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()
        #print('Gradient -', model.weight.grad)
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            for images, labels in testloaders:
                inputs = images.to(device)
                labels = labels.to(device)
                log_ps = model.forward(inputs)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                print(top_class) #the predicted class
                print(torch.exp(top_p)) # the predicted probability
                equals = top_class == labels.view(*top_class.shape)
                print(equals)
                print(equals.float().mean())
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()
        train_losses.append(running_loss/len(trainloaders))
        test_losses.append(test_loss/len(testloaders))
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),#(running_loss/len(trainloaders)),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),#test_loss/len(testloaders)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloaders)))

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt        
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
# TODO: Save the checkpoint 
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
model.class_to_idx = traindatasets.class_to_idx
model.cpu()
torch.save({'net': 'vgg16_bn',
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            'checkpoint.pth')
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net = checkpoint['net']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

load_model('checkpoint.pth')  
print(model)
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
# TODO: Display an image along with the top 5 classes
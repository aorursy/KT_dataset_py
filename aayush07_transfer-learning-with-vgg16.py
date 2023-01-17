import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
DATA_DIR = '../input/nepalicurrencynpr/Nepali Currency'
transform_train = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                     ])

transform_valid = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                     ])

training_dataset = datasets.ImageFolder(DATA_DIR+'/Train', transform = transform_train)
validation_dataset = datasets.ImageFolder(DATA_DIR+'/Test',transform = transform_valid)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=128, shuffle=False)
print(len(training_dataset))
print(len(validation_dataset))
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    image = image.clip(0,1)
    
    return image

classes = os.listdir(DATA_DIR+'/Train')
print(classes)
dataiter = iter(training_loader) #Converting our train folder to iterable so that it can be iterater through a batch of 20
images, labels = dataiter.next() #we get inputs for our model here
fig = plt.figure(figsize=(25,5))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title(classes[labels[idx].item()])
model = models.vgg16(pretrained=True)
model
#Freezing feature parameters and using sane as in VGG16. 
for param in model.features.parameters():
    param.requires_grad = False #feature layer does not require any gradient
import torch.nn as nn
n_inputs = model.classifier[6].in_features #tge i/p features of our classifier model
last_layer = nn.Linear(n_inputs, len(classes)) #new layer that we want to put in there
model.classifier[6] = last_layer #replacing last layer of vgg16 with our new layer
model.to(device) #put the model in device for higher procesing power
print(model.classifier[6].out_features)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#learning rate is very small as we have small number of datasets
epochs = 10 #Less because of small number of datasets
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    
    for inputs, labels in training_loader:
        #taking inputs and putting it in our model which is inside the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
    
    else:
        with torch.no_grad():
            for val_inputs, val_labels  in validation_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                
                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss = val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)
                
            epoch_loss = running_loss/len(training_loader.dataset)*100
            epoch_acc = running_corrects.float()/len(training_loader.dataset)*100 #we are now dividing the total loss of one epoch with the entire length of dataset to get the probability between 1 and 0
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc)
            val_epoch_loss = val_running_loss/len(validation_loader.dataset)*100
            val_epoch_acc = val_running_corrects.float()/len(validation_loader)
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)
            print('epoch :', (e+1))
            print('training loss: {:.4f}, training_acc {:.4f} '.format(epoch_loss, epoch_acc.item())) 
            print('validation loss: {:.4f}, validation acc {:.4f}'.format(val_epoch_loss, val_epoch_acc.item())) 
            
            
            
        
           
plt.style.use("ggplot")
plt.plot(running_loss_history, label='training loss')
plt.plot(val_running_loss_history, label='validation loss')
plt.legend()
plt.style.use("ggplot")
plt.plot(running_corrects_history, label='training accuracy')
plt.plot(val_running_corrects_history, label = 'validation accuracy')
plt.legend()

import PIL.ImageOps  # # from python imaging library we take this so we can preform operations on our image
import requests
from PIL import Image

dataiter = iter(validation_loader)
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)
output = model(images)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[idx]))
  ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])), color=("green" if preds[idx]==labels[idx] else "red"))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline
# Set the model to evaluate mode
model.eval()

# Get predictions for the test data and convert to numpy arrays for use with SciKit-Learn
print("Getting predictions from test set...")
truelabels = []
predictions = []
for data, target in validation_loader:
    for label in target.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model.cpu()(data).data.numpy().argmax(1):
        predictions.append(prediction) 

# Plot the confusion matrix
cm = confusion_matrix(truelabels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Shape")
plt.ylabel("True Shape")
plt.show()
torch.save(model.state_dict(), 'project-vgg16.pth')
project_name = 'Major Project'
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project_name, environment=None, outputs=['project-vgg16.pth'])

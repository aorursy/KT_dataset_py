from torchvision import transforms # To perform all the transforms on our data
from torchvision import datasets #Used to load the data from the folders
from torch.utils.data import DataLoader
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import time
from PIL import Image
import os
import matplotlib.pyplot as plt
!nvidia-smi
from shutil import copyfile
classes = ['drawing','hentai','neutral','porn','sexy']
for i in classes:
  files = os.listdir("../input/nsfw-image-classification/test/"+i)
  for j in files[:3000]:
    copyfile("../input/nsfw-image-classification/test/"+i+"/"+j, "../input/output/test_norm/"+i+"/"+j)
  print(i)
files = os.listdir("../input/output/test_norm/"+i)
print(len(files))
os.rmdir("../input/output/test_norm/.ipynb_checkpoints")
image_transforms = {
    'train':transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid':transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
test_directory = 'test_norm'

# Setting batch size for training
batch_size=128

#Number of classes for the data
num_classes = 5

#Loading the data from the folders into the variable 'data'
data = {
    'test': datasets.ImageFolder(root=test_directory,transform=image_transforms['test']),
}

test_data_size = len(data['test'])
test_data_loader = DataLoader(data['test'],batch_size=batch_size,shuffle=True)
idx_to_class = {v: k for k, v in data['test'].class_to_idx.items()}
print(idx_to_class)
# Set the train, test and validation directory
train_directory = 'batch-7/train'
test_directory = 'batch-7/test'
valid_directory = 'batch-7/valid'

# Setting batch size for training
batch_size=128

#Number of classes for the data
num_classes = 5

#Loading the data from the folders into the variable 'data'
data = {
    'train': datasets.ImageFolder(root=train_directory,transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=test_directory,transform=image_transforms['test']),
    'valid': datasets.ImageFolder(root=valid_directory,transform=image_transforms['valid'])
}

#Find out the size of the data
train_data_size = len(data['train'])
test_data_size = len(data['test'])
valid_data_size = len(data['valid'])

# Create iterators for the Data loaded using DataLoader module
train_data_loader = DataLoader(data['train'],batch_size=batch_size,shuffle=True)
test_data_loader = DataLoader(data['test'],batch_size=batch_size,shuffle=True)
valid_data_loader = DataLoader(data['valid'],batch_size=batch_size,shuffle=True)

#Printing the sizes of the sets
print(train_data_size,test_data_size,valid_data_size)
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)
# Load the pretrained resnet 50 model
resnet50 = models.resnet50(pretrained=True)
# We don't want to train the model with new values, so we make the existing values untrainable
for param in resnet50.parameters():
    param.requires_grad=False
    
# We want to add in a extra layer at the last with a 10 neuron output to classify the data
#Number of neurons in the last layer
fc_inputs = resnet50.fc.in_features

#Replacing last layer with our layers
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs,256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256,5),
    nn.LogSoftmax(dim=1)
)
model = torch.load("drive/My Drive/Image_class_NSFW_Stats/models/model_batch-9.pt")
# Define the optimizer and loss function
weights = torch.tensor([1,1,1,1,6])
loss_func = nn.NLLLoss(weight=weights.float().cuda())
optimizer = optim.Adam(model.parameters())
def train_and_validate(model,loss_criterion,optimizer,epochs=25):
    start = time.time()
    best_acc = 0.0
    history = []
    
    #Training the data
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch - {}/{}".format(epoch+1,epochs))
        
        #Set to training mode
        model.train()
        
        train_loss = 0.0
        train_acc = 0.0
        
        val_loss = 0.0
        val_acc = 0.0
        
        #Training
        for i,(inputs,labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            #lavde  mooditu kelu
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()*inputs.size(0)
            
            ret,predictions = torch.max(outputs.data,1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            acc = torch.mean(correct_counts.type(torch.cuda.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            
        #Validation
        with torch.no_grad():
            
            model.eval()
            
            for j,(inputs,labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_criterion(outputs,labels)
                
                val_loss += loss.item() * inputs.size(0)
                
                ret,predictions = torch.max(outputs.data,1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                
                acc = torch.mean(correct_counts.type(torch.cuda.FloatTensor))
                val_acc += acc.item()*inputs.size(0)
                
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
        
        avg_val_loss = val_loss/valid_data_size
        avg_val_acc = val_acc/valid_data_size
        
        history.append([avg_train_loss,avg_val_loss,avg_train_acc,avg_val_acc])
        
        epoch_end = time.time()
        print("Epoch : {},Training: Loss: {:.4f}, Accuracy:{:.4f}%\n\t\tValidation : Loss:{:.4f},Accuracy:{:.4f}".format(epoch+1,avg_train_loss,avg_train_acc*100,avg_val_loss,avg_val_acc*100))
        print("Time taken:"+str((epoch_end-epoch_start)))
        if(avg_val_acc>best_acc):
              best_acc = avg_val_acc
    return model,history   
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs=30

if torch.cuda.is_available():
    model.cuda()
print(torch.cuda.is_available())
    
model,history = train_and_validate(model,loss_func,optimizer,num_epochs)

# Save the model of the corresponding batch and its history for further use
torch.save(model,'drive/My Drive/Image_class_NSFW_Stats/models/model_batch-10.pt')
torch.save(history, 'drive/My Drive/Image_class_NSFW_Stats/history/history_batch-10.pt')
def predict(model, test_image_name):
     
    transform = image_transforms['test']
 
    test_image = Image.open(test_image_name)
    plt.imshow(test_image)
     
    test_image_tensor = transform(test_image)
 
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
     
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        print("Output class :  ", idx_to_class[topclass.cpu().numpy()[0][0]])
predict(model,"batch-1/test/neutral/01013.jpg")
model = torch.load("nsfw_classification.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([1,1,1,1,6])
loss_func = nn.NLLLoss(weight=weights.float().cuda())

y_true = torch.Tensor([]).cuda()
y_pred = torch.Tensor([]).cuda()

test_loss = 0.0
test_acc = 0.0
for j,(inputs,labels) in enumerate(test_data_loader):
  inputs = inputs.to(device)
  labels = labels.to(device)
  outputs = model(inputs)
  loss = loss_func(outputs,labels)
  y_true = torch.cat([y_true,labels.float()],dim=0)
  
  test_loss += loss.item() * inputs.size(0)

  ret,predictions = torch.max(outputs.data,1)
  y_pred = torch.cat([y_pred,predictions.float()],dim=0)
  correct_counts = predictions.eq(labels.data.view_as(predictions))

  acc = torch.mean(correct_counts.type(torch.cuda.FloatTensor))
  test_acc += acc.item()*inputs.size(0)

avg_test_loss = test_loss/test_data_size
avg_test_acc = test_acc/test_data_size
print(y_true,y_pred)
print(avg_test_acc)
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set_style('white') 
labels = ['drawing', 'hentai', 'neutral', 'porn', 'provocative']
x = confusion_matrix(y_true.tolist(),y_pred.tolist())
sns.heatmap(x,xticklabels=labels,yticklabels=labels,linecolor='white')
from sklearn.metrics import classification_report
print(classification_report(y_true.tolist(),y_pred.tolist()))
current = "7,"+str(train_data_size)+","+str(avg_test_acc*100)+"\n"
print(current)
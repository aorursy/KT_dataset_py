import numpy as np
import pandas as pd
import os
import json

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from PIL import Image
print(os.listdir("../input"))
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
batch_size = 20
n_workers = 0
data_dir = '../input/oxford-102-flower-pytorch/flower_data/flower_data/'
with open(data_dir + '/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
valid_dataset = datasets.ImageFolder(root = data_dir + '/valid', transform = test_transforms)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers = n_workers)

test_folder = '../input/test-flowers/test_data/test_data/test'
test_dataset = ImageFolderWithPaths(root = test_folder, transform = test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers = n_workers)

# Print the statistics:
print("Datasets Load has finished:")
print("\tNumber of validation images:{}".format(len(valid_dataset)))
print("\tNumber of test images:{}".format(len(test_dataset)))
class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_output)
        self.dropout = nn.Dropout(p = 0.3)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x))) 
        x = self.dropout(F.relu(self.fc2(x)))     
        x = F.log_softmax(self.fc3(x), dim=1)        
        return x
checkpoint_path = '../input/additional-training/checkpoint_105.pth'
checkpoint = torch.load(checkpoint_path)
model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
# Put the classifier on the pretrained network
n_inputs = model.classifier.in_features
last_layer = Classifier(n_inputs, 512, 256, len(cat_to_name))
model.classifier = last_layer

pretrained_dict = checkpoint['state_dict']
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model.load_state_dict(pretrained_dict)
model.to(device);
# Check again on validation set to see that we didn't "destroy" the model:
criterion = nn.NLLLoss()
valid_loss, accuracy  = 0, 0
model.eval()
with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)                    
        valid_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
predictions = {}
idx_to_class = {v: k for k, v in valid_dataset.class_to_idx.items()}
with torch.no_grad():
    for image, label, filename in test_loader:
        image, label = image.to(device), label.to(device)
        logps = model.forward(image)
        ps = torch.exp(logps)
        
        top_p, top_class = ps.topk(1, dim=1)
        class_pred = top_class.data.cpu().numpy()[0]        
        filename = filename[0].split('/')[-1]
        predictions[filename] = idx_to_class[class_pred.tolist()[0]]
sub = pd.DataFrame.from_dict(predictions, orient='index')
sub.index.names = ['file_name']
sub.columns = ['id']
sub.to_csv('submission_01.csv')
"""
def image_loader(image_name):
    #load image, returns cuda tensor
    image = Image.open(image_name)
    image = test_transforms(image)#.float()
    #image = autograd.Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)  #assumes that you're using GPU
    
# Define Path for Test Images:
test_dir = '../input/oxford-102-flower-pytorch/flower_data/flower_data/test/'
predictions = {}
filenames = next(os.walk(test_dir))[2]
for filename in filenames:
    image = image_loader(test_dir + "/" + filename)
    pred = model(image)
    ps = torch.exp(pred)
    top_p, top_class = ps.topk(1, dim=1)
    class_pred = top_class.data.cpu().numpy()[0]
    predictions[filename] = class_pred
"""
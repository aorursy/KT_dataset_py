import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

print(os.listdir("../input"))
print(f'\nPyTorch version {torch.__version__}\n')
print('cuda' if torch.cuda.is_available() else 'cpu')
print(os.listdir('../input/oxford-102-flower-pytorch/flower_data/flower_data/'))
#try data_dir = '../input/flower_data/flower_data/'
data_dir = '../input/oxford-102-flower-pytorch/flower_data/flower_data/'
#data_dir = '../input/flower_data/flower_data/'
class TestDataset(data.Dataset):
    '''
    Custom dataset class for test dataset which contains uncategorized images.
    The category index is set to 0 for all images (we don't need it).
    It also returns the filename of each image.
    '''
    def __init__(self, path, transform=None):
        self.path = path
        self.files = []
        for (dirpath, _, filenames) in os.walk(self.path):
            for f in filenames:
                if f.endswith('.jpg'):
                    p = {}
                    p['img_path'] = dirpath + '/' + f
                    self.files.append(p)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]['img_path']
        img_name = img_path.split('/')[-1]
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 0, img_name
    
    
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
    
# Image transformations
data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

# Validation dataset
val_dataset = datasets.ImageFolder(data_dir + 'valid', transform=data_transforms)

# Test dataset
test_dataset = TestDataset(data_dir + 'test', transform=data_transforms)

# Create the dataloaders
batch_size = 32
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def imshow(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((1, 2, 0))
    else:
        image = np.array(image).transpose((1, 2, 0))
    # Unnormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.imshow(image)
    ax.axis('off') 
        
    # Make a grid from batch
images, _, _ = next(iter(test_loader))
out = torchvision.utils.make_grid(images, nrow=8)
imshow(out)
# Change the classifier to the one you used.
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(25088, 12000)
        self.fc2 = nn.Linear(12000, 6000)
        self.fc3 = nn.Linear(6000, 1000)
        self.fc4 = nn.Linear(1000, 500)
        self.fc5 = nn.Linear(500, 102)

        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))

        # output so no dropout here
        x = self.fc5(x)
        
        return x
'''def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Put the classifier on the pretrained network
    model.classifier = Classifier()
    #loading the weights
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    return model

model = load_checkpoint('../input/checkpoint/checkpoint_ft_0_.pth')

# model.load_state_dict(torch.load(‘/home/workspace/checkpoint-3.pth’, map_location=lambda storage, loc: storage))
# If you used something other than 224x224 cropped images, set the correct size here
image_size = 224
# Values you used for normalizing the images. Default here are for
# pretrained models from torchvision.
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
print('done')
'''


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Here put the pretrained model that you used (in my case it's densenet161).
    model = models.vgg16(pretrained=False)
    
    # We freeze all the layers since we are not training the model here.
    for param in model.parameters():
        param.requires_grad = False
    
    try:
        #n_ftrs = model.classifier.in_features
        model.classifier = Classifier()
    except AttributeError:
        n_ftrs = model.fc.in_features
        model.fc = classifier(n_ftrs)
                              
    model.load_state_dict(checkpoint['state_dict'])  # your checkpoint's key may differ (e.g.'state_dict')
    model.eval()
    
    return model

model = load_checkpoint('../input/mydata/classifiervgg16.pth')  # use your path to checkpoint file
print(model.classifier)
def comp_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        running_acc = 0.0
        for ii, (images, labels) in enumerate(dataloader, start=1):
            if ii % 5 == 0:
                print('Batch {}/{}'.format(ii, len(dataloader)))
            images, labels = images.to(device), labels.to(device)
            logps = model(images)
            ps = torch.exp(logps)  # in my case the outputs are logits so I take the exp()
            equals = ps.topk(1)[1].view(labels.shape) == labels          
            running_acc += equals.sum().item()
        acc = running_acc/len(dataloader.dataset) 
        print(f'\nAccuracy: {acc:.5f}') 
        
    return acc
comp_accuracy(model, val_loader)
# The prediction of our model is an index which we need to convert back to the class label.
# For this, we will use the following mapping
idx_to_class = {val: key for key, val in val_dataset.class_to_idx.items()}
print(idx_to_class)
def predict(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()
    
    predictions = {}   
    with torch.no_grad():
        for ii, (images, _, img_names) in enumerate(dataloader, start=1):
            if ii % 5 == 0:
                print('Batch {}/{}'.format(ii, len(dataloader)))
            images = images.to(device)
            logps = model(images)
            ps = torch.exp(logps)
            
            # Top indices
            _, top_indices = ps.topk(1)
            top_indices = top_indices.detach().cpu().numpy().tolist()
    
            # Convert indices to classes
            top_classes = [idx_to_class[idx[0]] for idx in top_indices]
            
            for i, img_name in enumerate(img_names):
                predictions[img_name] = top_classes[i]
            
        print('\nCompleted')

    return predictions
predictions = predict(model, test_loader)
submission = pd.DataFrame(list(predictions.items()), columns=['file_name', 'id'])
submission.to_csv('submission.csv', index=False)
print(submission)

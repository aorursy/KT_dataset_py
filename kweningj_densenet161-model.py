# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import json
print(os.listdir("../input"))
ROOT_DIR = '../input'
print('root data dir:', os.listdir(ROOT_DIR))
FLOWER_DATA_DIR = os.path.join(ROOT_DIR, 'oxford-102-flower-pytorch',
                                  'flower_data', 'flower_data')
print('flower data dir:', os.listdir(FLOWER_DATA_DIR))
# print(os.listdir("../input/oxford-102-flower-pytorch/flower_data/flower_data/cat_to_name.json"))

# Any results you write to the current directory are saved as output.
# Load label mapping
LABEL_MAPPING_PATH = os.path.join(ROOT_DIR, 'oxford-102-flower-pytorch',
                                  'flower_data', 'flower_data',
                                  'cat_to_name.json')
with open(LABEL_MAPPING_PATH, 'r') as f:
    cat_to_name = json.load(f)

cat_to_name
# Kernel imports
import time
import json
%config InlineBackend.figure_format = 'retina'
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('CUDA is available!  Running on GPU ...')
else:
    print('CUDA is not available.  Running on CPU ...')
# Configure Densnet model
densenet = {
    'arch': 'densenet161_v2_unfreeze',
    'model': models.densenet161(pretrained=True),
    'reshape': {
        'classifier': nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2208, 102))
        ]))
    }
}
# Load model checkpoint
OUTPUT_DIR = os.path.join(ROOT_DIR, 'image-classifier-project-pytorch-challenge')
# Configure checkpoint model
if train_on_gpu:
    checkpoint = torch.load(f'{OUTPUT_DIR}/densenet161_v2_unfreeze_model_12.pt')
else:
    checkpoint = torch.load(f'{OUTPUT_DIR}/densenet161_v2_unfreeze_model_12.pt',
                            map_location='cpu')
print(checkpoint.keys())

# Fully configure achitecture
model = densenet['model']
model.classifier = densenet['reshape']['classifier']
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']
model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
model.classifier
# Data directory paths
TRAIN_DIR = os.path.join(FLOWER_DATA_DIR, 'train')
VALID_DIR = os.path.join(FLOWER_DATA_DIR, 'valid')

# Load train and test data
def load_data(input_size=224):
    '''Load data and crop to specified input_size.'''
    # Define your transforms for the training and validation sets
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    # Load the datasets with ImageFolder
    train_data = ImageFolder(TRAIN_DIR, transform=train_transforms)
    valid_data = ImageFolder(VALID_DIR, transform=valid_transforms)
    
    return train_data, valid_data
# Prepare loading test images
TEST_DIR = os.path.join(FLOWER_DATA_DIR, 'test')
test_image_files = os.listdir(TEST_DIR)
test_image_files[:10]

# Load data
_, valid_data = load_data()

# Obtain a batch of images
batch_size = 12
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(102))
class_total = list(0. for i in range(102))

criterion = nn.CrossEntropyLoss()

# Use GPU if available, attach device to model accordingly
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if train_on_gpu:
    device = 'cuda'
else:
    device = 'cpu'
print(f'Training model on {device}')
model.to(device)

model.eval() # prep model for evaluation
for data, target in valid_loader:
    data, target = data.to(device), target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(102):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img
def predict(image_path, model, top_num=5):
    # To cpu
    model.to('cpu')
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.softmax(model.forward(model_input), dim=1)
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers
# Predict each image in test folder
preds = list()
names = list()
probs = list()
for image_file in test_image_files:
    image_path = os.path.join(TEST_DIR, image_file)
    top_probs, top_labels, top_flowers = predict(image_path, model, top_num=1)
    prob = top_probs[0]
    label = top_labels[0]
    name = top_flowers[0]
    print(f'prediction for {image_file}: label = {label}, prob = {prob}, name = {name}')
    probs.append(prob)
    names.append(name)
    preds.append(label)
# Store results in a dataframe
results = {
    'file_name': test_image_files,
    'label': preds,
    'probability': probs,
    'name': names
}
prediction_results = pd.DataFrame(results)
prediction_results.head()
# Transform into submission format
submission_df = prediction_results.drop(['probability', 'name'], axis=1)
submission_df = submission_df.rename(columns={'label' : 'id'})
submission_df.head()
# Write results and submission to csv
prediction_results.to_csv('test_data_predictions.csv', sep=',', index=False)
submission_df.to_csv('densenet161_submission.csv', sep=',', index=False)

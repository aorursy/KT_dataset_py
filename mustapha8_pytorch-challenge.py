# Imports here
import os
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

import torchvision
from torchvision import datasets, models, transforms
import helper
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
from subprocess import check_output
print(check_output(["ls", "../input/flower-data/flower_data/flower_data/train/1/"]).decode("utf8"))
data_dir = '../input/flower-data/flower_data/flower_data'
train_dir = os.path.join(data_dir + '/train')
valid_dir = os.path.join(data_dir + '/valid')

# TODO: Define your transforms for the training and validation sets
data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                    
                                      transforms.RandomHorizontalFlip(),
                                      
                                      transforms.RandomRotation(0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                     
                                      ])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(data_dir, transform = data_transforms)
train_data= datasets.ImageFolder(train_dir, transform = data_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = data_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
batch_size=32
num_workers=0
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
train_loder = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = num_workers, shuffle =True)
valid_loder = torch.utils.data.DataLoader(valid_data, batch_size = batch_size , num_workers = num_workers, shuffle = True)
print(len(train_loder))
print(len(valid_loder))
#visualize some data
import helper
image = next(iter((train_data)))
helper.imshow(image[0,:]);
# Visualize some sample data

# obtain one batch of training images
dataiter = iter(train_loder)

images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    
import json
print(check_output(["ls","../input/json-file"]).decode("utf8"))
with open('../input/json-file/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    label_map = cat_to_name
    #print(cat_to_name)
    classes= []
    for  k,val in cat_to_name.items():
        classes.append(val)
    print(classes)
    
# TODO: Build and train your network
#i'm use the vgg16 trained model then i will freez the weights of features as they are already trained 
vgg = models.vgg16(pretrained=True)
for param in vgg.features.parameters():
    param.requires_grad = False
    
    
    

vgg.classifier[5].p=0.2
vgg.classifier[2].p=0.2
#move model to cuda if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
vgg.to(device)
print(vgg)

print(vgg)
#we need to change the out_features in classifier seq 
#the six layer in classiefier is the ouput so we change it to 102 from 1000 like so

vgg.classifier[6].out_features = 102

#we need to optimze only the pram of the classifier layer 
print(vgg)
optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.01,  momentum = 0.9)
criterion = nn.CrossEntropyLoss()
is_cuda = torch.cuda.is_available()

epochs = 30
for epoch in range(1,epochs + 1): 
    running_loss = 0.0
    vgg.train()
    for  batchi ,(data ,target) in enumerate(train_loder):
        if is_cuda:
            data,target = data.cuda(),target.cuda()
        optimizer.zero_grad()
        output = vgg(data)
        #loss = F.nll_loss(output,target)
        loss = criterion(output , target)
        loss.backward()
        optimizer.step()
        #running_loss += F.nll_loss(output,target,reduction='sum').item()
        running_loss += loss.item()*data.size(0)
        
        running_loss = running_loss / len(train_loder.dataset)
        #train_loss = running_loss / len(train_loder.dataset)
        if batchi % 32 == 31:
            print('Epoch %d, Batch %d loss: %.6f' % (epoch, batchi + 1, running_loss ))
            
            

        
#build a function for valid model
valid_loss = 0.0
running_correct = 0
class_correct = list(0. for i in range(102))
class_total = list(0. for i in range(102))
vgg.eval()
for batchi, (data, target) in enumerate(valid_loder):
    if is_cuda:
        data,target = data.cuda(),target.cuda()
    output = vgg(data)
    loss = criterion(output,target)
    valid_loss += loss.item()*data.size(0)
        
    #preds = output.data.max(dim=1,keepdim=True)[1]
    _, preds = torch.max(output, 1)
   
    #accuracy = 100 * running_correct/len(valid_loder.dataset)
        
    #if batchi % 2 == 1:
    #print(' Batch %d, loss: %.6f, acc: %.2d' % ( batchi + 1, valid_loss, accuracy ))

    correct_tensor =  preds.eq(target.data.view_as(preds))   
    correct = np.squeeze(correct_tensor.numpy()) if not is_cuda else np.squeeze(correct_tensor.cpu().numpy())
    #print(target.data)
   # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
# average test loss
#test_loss = test_loss/len(test_loader.dataset)
valid_loss = valid_loss/len(valid_loder.dataset)
print('valid Loss: {:.6f}\n'.format(valid_loss))
for i in range(102):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
        classes[i], 100 * class_correct[i] / class_total[i],
        np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)'  % (classes[i]))
print('\nValid Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


#builed the training model

# TODO: Save the checkpoint 
model =vgg
model.cuda()
model.class_to_idx = train_data.class_to_idx
checkpoint ={ 'Epoch': epoch,
          
           'model_optimizer_dict': optimizer.state_dict(),
           'class_to_idx' : model.class_to_idx ,
            'model_state_dict': model.state_dict()  
           }
torch.save(checkpoint, 'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
model.cuda()
filepath = 'checkpoint.pth'
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = vgg(checkpoint['Epoch'],
                             checkpoint['model_optimizer_dict'],
                             checkpoint['class_to_idx'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

import PIL
from PIL import Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #here some thing i changed
    image =np.transpose(image , (1, 2, 0))
    #image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
%matplotlib inline
import matplotlib.image as mpimg
#print('../input/flower-data/flower_data/flower_data/train/1/image_06734.jpg')
#img=mpimg.imread('../input/flower-data/flower_data/flower_data/train/1/image_06734.jpg')
imge = '../input/flower-data/flower_data/flower_data/train/1/image_06737.jpg'
_= imshow(process_image(imge))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.cuda()
def predict(image_path, model, topk = 5):
 #   ''' Predict the class (or classes) of an image using a trained deep learning model.
  #  '''
    
     # TODO: Implement the code to predict the class from an image file
    #probs, classes = predict.predict(image=imge, checkpoint='checkpoint.pth', labels='cat_to_name.json', gpu=True)
    #print(probs)
    #print(classes)
    model.eval()
    imege = process_image(image_path)
    image_tensor = torch.from_numpy(imege).type(torch.FloatTensor)
    input_img = image_tensor.unsqueeze(0) 
    d = input_img.cuda()
    probs = torch.exp(model.forward(d))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [label_map[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

# TODO: Display an image along with the top 5 classes

model.cuda()
def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('/')[6]
    title_ = label_map[flower_num]
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);
    # Make prediction
    probs, labs, flowers = predict(image_path, model) 
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()
image_path = imge
plot_solution(image_path, model)
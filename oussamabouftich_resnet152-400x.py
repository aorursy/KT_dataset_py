!pip install Pillow==5.3.0
!pip install image
!pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html  

!pip install --upgrade pip

import PIL
print(PIL.PILLOW_VERSION)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
import copy
import json
import os
from os.path import exists
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Organizing the dataset
data_dir = '../input/dataset-breks/data set'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
batch_size = 32
use_gpu = torch.cuda.is_available()
import json
with open('../input/cat-to-name-1json/cat_to_name (1).json', 'r') as f:
    cat_to_name = json.load(f)


# Définissez vos transformations pour les ensembles de formation et de validation
# Augmentation et normalisation des données pour la formation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
#Charger les jeux de données avec ImageFolder
data_dir = '../input/dataset-breks/data set'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])for x in ['train', 'valid']}
#print(image_datasets)
# À l'aide des jeux de données d'images et des trains, définissez les chargeurs de données
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
print(dataset_sizes)

class_names = image_datasets['valid'].classes

print(class_names)


model = models.alexnet(pretrained=True)
# Geler les paramètres
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict


# Remplacement du classificateur de modèle pré-formé par notre classificateur

model.classifier [6] = nn.Sequential ( 
                      nn.Linear (4096, 256), 
                      nn.ReLU (), 
                      nn.Dropout (0.5), 
                      nn.Linear (256, 2),                    
                      nn.LogSoftmax ( dim = 1))
print(model)

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_v = 0.0
    best_acc_T = 0.0
    best_loss_v= 1.0
    best_loss_T= 1.0
    loss_dict = {'train': [], 'valid': []}
    acc_dict = {'train': [], 'valid': []}

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
              

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() #L'appel de .backward()plusieurs fois accumule le gradient (par addition) pour chaque paramètre. C'est pourquoi vous devez appeler optimizer.zero_grad()après chaque .step()appel.
                        optimizer.step()#est effectue une mise à jour des paramètres basée sur le gradient actuel SGD

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                #print(labels)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            

           # copier en profondeur le modèle
            if phase == 'valid' :
                if epoch_acc > best_acc_v  :
                   best_acc_v = epoch_acc
                   best_model_wts = copy.deepcopy(model.state_dict())
                if best_loss_v > epoch_loss:
                   best_loss_v = epoch_loss    
            if phase == 'train' :
                if epoch_acc > best_acc_T:
                   best_acc_T = epoch_acc
                if best_loss_T > epoch_loss:
                   best_loss_T = epoch_loss    
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc_v))
    print('Best train  accuracy: {:4f}'.format(best_acc_T))
    print('valid losss: {:4f}'.format(best_loss_v))
    print('train losss: {:4f}'.format(best_loss_T))


    #   charger les meilleurs poids de modèle
    model.load_state_dict(best_model_wts)
    return model,loss_dict, acc_dict,time_elapsed

    
ress_loss = {'train': [], 'valid': []}
ress_acc = {'train': [], 'valid': []}
time_elapse=0
# Train a model with a pre-trained network
res_loss = {'train': [], 'valid': []}
res_acc = {'train': [], 'valid': []}
if use_gpu:
    print ("Using GPU: "+ str(use_gpu))
    model = model.cuda()
# NLLLoss because our output is LogSoftmax
criterion = nn.NLLLoss()
# Adam optimizer with a learning rate
optimizer = optim.SGD(model.classifier.parameters(), lr=0.006, momentum=0.9)
# Decay LR by a factor of 0.1 every 5 epochs 15
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

model_ft,loss_dict, acc_dict,time_elapsed = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=50)
res_loss = loss_dict
res_acc = acc_dict





time_elapse=time_elapse+time_elapsed
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapse // 60, time_elapse % 60))
ress_loss['train'].extend(loss_dict['train'])
ress_loss['test'].extend(loss_dict['test'])
ress_acc['train'].extend(acc_dict['train'])
ress_acc['test'].extend(acc_dict['test'])
print(ress_loss)
print(ress_acc)

from torch.utils.data import DataLoader
plt.rcParams["figure.figsize"] = (13,13)

res_loss = loss_dict
res_acc = acc_dict
plt.title("Loss")

plt.plot(ress_loss['train'],label='Training Loss')  
plt.plot(ress_loss['test'],label='Validation Loss')  

plt.legend()  
plt.show()  
from torch.utils.data import DataLoader
plt.rcParams["figure.figsize"] = (11,11)

res_loss = loss_dict
res_acc = acc_dict
plt.title("Accuracy")

plt.plot(ress_acc['train'],label='Training acc')  
plt.plot(ress_acc['test'],label='Validation acc')
plt.plot(ress_loss['test'],label='Validation Loss') 

plt.legend()  
plt.show()  

time_elapse=time_elapse+time_elapsed
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapse // 60, time_elapse % 60))
# Save the checkpoint 
num_epochs=100
model.class_to_idx = dataloaders['train'].dataset.class_to_idx
model.epochs = num_epochs
checkpoint = {'input_size': [3, 224, 224],
                 'batch_size': dataloaders['train'].batch_size,
                  'output_size': 2,
                  'state_dict': model.state_dict(),
                  'data_transforms': data_transforms,
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs,
                  'ress_loss': ress_loss,
                  'ress_acc': ress_acc 
             }
torch.save(checkpoint, 'resnet152.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.resnet152()
    
    # Our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 2
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          #('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(512, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Replacing the pretrained model classifier with our classifier
    model.fc = classifier
    
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']
# Get index to class mapping
loaded_model, class_to_idx = load_checkpoint('8960_checkpoint10.pth')

idx_to_class = { v : k for k,v in class_to_idx.items()}


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage
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
def predict(image_path, model, topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class
# Display an image along with the top 2 classes
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = img.split('/')[-2]
    img = Image.open(img)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    cancer_type = mapper[img_filename]
    
    ax1.set_title(cancer_type)
    ax1.imshow(img)
    ax1.axis('off')
    
    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()
img = '../input/data-breakhis/data images Breakhis Bresil/valid/MALIGNANT/SOB_M_DC-14-2523-100-023.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)
img = '../input/data-breakhis/data images Breakhis Bresil/valid/MALIGNANT/SOB_M_DC-14-2523-100-028.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)
img = '../input/data-breakhis/data images Breakhis Bresil/train/BENIGN/SOB_B_A-14-22549CD-100-007.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)
img = '../input/data-breakhis/data images Breakhis Bresil/train/BENIGN/SOB_B_A-14-22549CD-100-030.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)

img = '../input/data-breakhis/data images Breakhis Bresil/train/MALIGNANT/SOB_M_DC-14-10926-200-009.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)

img = '../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_A-14-22549AB-400-001.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)

img = '../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_A-14-22549AB-400-011.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)

img = '../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_A-14-22549CD-400-009.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)

img = '../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_F-14-14134-400-020.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)

img = '../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_F-14-21998EF-400-007.png'
p, c = predict(img, loaded_model)
print(p)
view_classify(img, p, c, cat_to_name)

validation_img_paths = ["../input/breakhis-400x/BreaKHis 400X/test/malignant/SOB_M_DC-14-16601-400-002.png",
                        "../input/breakhis-400x/BreaKHis 400X/test/malignant/SOB_M_DC-14-16716-400-018.png",
                        "../input/breakhis-400x/BreaKHis 400X/test/malignant/SOB_M_DC-14-17901-400-011.png",
                        "../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_F-14-21998EF-400-017.png",
                        "../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_F-14-23060AB-400-006.png",
                        "../input/breakhis-400x/BreaKHis 400X/test/benign/SOB_B_F-14-23222AB-400-008.png"]
img_list = [Image.open(img_path) for img_path in validation_img_paths]

validation_batch = torch.stack([data_transforms['valid'](img).to(device)
                                for img in img_list])
pred_logits_tensor = model(validation_batch)
pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title("{:.0f}% BENIGN, {:.0f}% MALIGNANT".format(100*pred_probs[i,0],
                                                          100*pred_probs[i,1]))
    ax.imshow(img)
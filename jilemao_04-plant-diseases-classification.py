import os # for working with files
import torch # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms # for transforming images into tensors 
from torchvision.utils import make_grid # for data checking
from torchvision.datasets import ImageFolder # for working with classes and images
%matplotlib inline
project_name = "Plant-Disease-Classification" # used by jovian
Data_Dir = "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
Train_Dir = Data_Dir + "/train"
Valid_Dir = Data_Dir + "/valid"
Diseases = os.listdir(Train_Dir)
print(Diseases)
print(len(Diseases))
plants = []
NumberOfDiseases = 0
for plant in Diseases:
    if plant.split('___')[0] not in plants:
        plants.append(plant.split('___')[0])
    if plant.split('___')[1] != 'healthy':
        NumberOfDiseases += 1
print(plants)
print(len(plants))
print(NumberOfDiseases)
# Number of images for each disease
nums = {}
for disease in Diseases:
    nums[disease] = len(os.listdir(Train_Dir + '/' + disease))
print(nums)
index = [n for n in range(38)]
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Plants/Diseases', fontsize=10)
plt.ylabel('No of images available', fontsize=10)
plt.xticks(index, Diseases, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')
add = 0
for val in nums.values():
    add += val
print(add)
# datasets for validation and training
train_ds = ImageFolder(Train_Dir, transform=transforms.ToTensor())
val_ds = ImageFolder(Valid_Dir, transform=transforms.ToTensor()) 
img, label = train_ds[0]
print(img.shape, label)
train_ds.classes

# for checking some images from training dataset
def show_image(image, label):
    print("Label :" + train_ds.classes[label] + "(" + str(label) + ")")
    plt.imshow(image.permute(1, 2, 0))
show_image(*train_ds[0])
show_image(*train_ds[70000])
show_image(*train_ds[30000])
random_seed = 7
torch.manual_seed(random_seed)
batch_size = 32
# DataLoaders for training and validation
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break
show_batch(train_dl) # Images for first batch of training
# for moving data into GPU
def get_default_device():
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        return len(self.dl)
device = get_default_device()
device
# Moving data into GPU
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    

class DiseaseClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        accur = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": accur}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))
        
# Architecture for training
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(DiseaseClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
            
        
model = to_device(ResNet9(3, len(train_ds.classes)), device) # defining the model and moving it to the GPU
model
# for training
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # scheduler for one cycle learniing rate
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # recording and updating learning rates
            lrs.append(get_lr(optimizer))
            sched.step()
            
    
        # validation
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history
    
%%time
history = [evaluate(model, val_dl)]
history
epochs = 1 #2
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
%%time
history += fit_OneCycle(epochs, max_lr, model, train_dl, val_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=1e-4, 
                             opt_func=opt_func)
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_ds.classes[preds[0].item()]
test_dir = "../input/new-plant-diseases-dataset/test"
test_ds = ImageFolder(test_dir, transform=transforms.ToTensor())
test_ds.classes
test_images = sorted(os.listdir(test_dir + '/test')) # since images in test folder are in alphabetical order
test_images
def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_ds.classes[preds[0].item()]
Image.open('../input/new-plant-diseases-dataset/test/test/AppleCedarRust1.JPG')
img, label = test_ds[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[0], ', Predicted:', predict_image(img, model))
img, label = test_ds[1]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[1], ', Predicted:', predict_image(img, model))
img, label = test_ds[2]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[2], ', Predicted:', predict_image(img, model))
img, label = test_ds[3]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[3], ', Predicted:', predict_image(img, model))
img, label = test_ds[4]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[4], ', Predicted:', predict_image(img, model))
img, label = test_ds[5]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[5], ', Predicted:', predict_image(img, model))
img, label = test_ds[6]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[6], ', Predicted:', predict_image(img, model))
img, label = test_ds[7]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[7], ', Predicted:', predict_image(img, model))
img, label = test_ds[8]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[8], ', Predicted:', predict_image(img, model))
img, label = test_ds[9]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[9], ', Predicted:', predict_image(img, model))
img, label = test_ds[10]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[10], ', Predicted:', predict_image(img, model))
img, label = test_ds[11]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[11], ', Predicted:', predict_image(img, model))
img, label = test_ds[12]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[12], ', Predicted:', predict_image(img, model))
img, label = test_ds[13]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[13], ', Predicted:', predict_image(img, model))
img, label = test_ds[14]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[14], ', Predicted:', predict_image(img, model))
img, label = test_ds[15]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[15], ', Predicted:', predict_image(img, model))
img, label = test_ds[16]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[16], ', Predicted:', predict_image(img, model))
img, label = test_ds[17]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[17], ', Predicted:', predict_image(img, model))
img, label = test_ds[18]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[18], ', Predicted:', predict_image(img, model))
img, label = test_ds[19]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[19], ', Predicted:', predict_image(img, model))
img, label = test_ds[20]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[20], ', Predicted:', predict_image(img, model))
img, label = test_ds[21]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[21], ', Predicted:', predict_image(img, model))
img, label = test_ds[22]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[22], ', Predicted:', predict_image(img, model))
img, label = test_ds[23]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[23], ', Predicted:', predict_image(img, model))
img, label = test_ds[24]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[24], ', Predicted:', predict_image(img, model))
img, label = test_ds[25]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[25], ', Predicted:', predict_image(img, model))
img, label = test_ds[26]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[26], ', Predicted:', predict_image(img, model))
img, label = test_ds[27]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[27], ', Predicted:', predict_image(img, model))
img, label = test_ds[28]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[28], ', Predicted:', predict_image(img, model))
img, label = test_ds[29]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[29], ', Predicted:', predict_image(img, model))
img, label = test_ds[30]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[30], ', Predicted:', predict_image(img, model))
img, label = test_ds[31]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[31], ', Predicted:', predict_image(img, model))
img, label = test_ds[32]
plt.imshow(img.permute(1, 2, 0))
print('Label:', test_images[32], ', Predicted:', predict_image(img, model))
torch.save(model.state_dict(), 'plantdiseaseclassification.pth')
# init the name mapping between test filenames and class names
name_mappings = {
    "AppleCedarRust":"Apple___Cedar_apple_rust",
    "AppleScab":"Apple___Apple_scab",
    "CornCommonRust":"Corn_(maize)___Common_rust_",
    "PotatoEarlyBlight":"Potato___Early_blight",
    "PotatoHealthy":"Potato___healthy",
    "TomatoEarlyBlight":"Tomato___Early_blight",
    "TomatoHealthy":"Tomato___healthy",
    "TomatoYellowCurlVirus":"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
import numpy as np
x=[]
y=[]

# start to predict looping in test_ds
for i in range(len(test_ds)):
    # store the image data and label name
    img, label = test_ds[i]
    # get the label name of the test image
    label1=name_mappings[test_images[i][:(len(test_images[i])-5)]]
    # predict according to the test data using model
    label2=predict_image(img, model)
    # append the label and the prediction result for furture use
    x.append(label1)
    y.append(label2)
    # show the expect label name and predicted label name
    print('Label:', label1, ', Predicted:', label2)

# generate the xn, yn from x, y (from label name to label index for confusion generating)
xn=[]
yn=[]
keys=list(set(x))
keys.sort()
for i in range(len(test_ds)):
    xn.append(keys.index(x[i]))
    try:
        yn.append(keys.index(y[i]))
    except:
        yn.append(0)
        pass
xn=np.transpose(xn)
yn=np.transpose(yn)
keys=np.transpose(keys)
import tensorflow as tf
import pandas as pd

# use the confusion_matrix function provided by tensorflow to generate confusion matrix
con_mat_1 = tf.math.confusion_matrix(labels=xn, predictions=yn).numpy()

# normalize the confusion matrix
con_mat_norm_1 = np.around(con_mat_1.astype('float') / con_mat_1.sum(axis=1)[:, np.newaxis], decimals=2)

# convert the nomalized confusion matrix for better view
con_mat_df_1 = pd.DataFrame(con_mat_norm_1,
                     index = keys, 
                     columns = keys)

# show the nomalized confusion matrix
con_mat_df_1
# convert the original confusion matrix for better view (using the case numbers)
con_mat_df_1_explain = pd.DataFrame(con_mat_1,
                     index = keys, 
                     columns = keys)

# show the unnomalized confusion matrix
con_mat_df_1_explain
import sklearn.metrics

# generate the clasification report by using the classification_report of sklearn package
report_1 = sklearn.metrics.classification_report(yn, xn,target_names=keys)

# print the report
print(report_1)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Imports
import csv
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
import os
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
train_tfms = tt.Compose([tt.Resize((250,250)), 
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor()])
val_tfms = tt.Compose([tt.Resize((250,250)), tt.ToTensor()])

writer = SummaryWriter('runs/classify-age-groups_experiment_1')

batch_size = 4
# PyTorch datasets
trainset = ImageFolder('/kaggle/input/agegroups/', train_tfms)
train_dl = DataLoader(trainset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valset = ImageFolder('/kaggle/input/agegroupsval/', val_tfms)
val_dl = DataLoader(valset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
classes = ('Adults', 'Teenagers', 'Toddler')
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
        break
show_batch(train_dl)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
#         print(f'Epoch {epoch},  train_loss: {result["train_loss"]}')
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(25088, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
        
model = to_device(ResNet9(3, 3), device)
model
# Load the TensorBoard notebook extension
# %load_ext tensorboard
# %reload_ext tensorboard
# %tensorboard --logdir logs
# # helper function
# def select_n_random(data, labels, n=100):
#     '''
#     Selects n random datapoints and their corresponding labels from a dataset
#     '''
#     assert len(data) == len(labels)

#     perm = torch.randperm(len(data))
#     return data[perm][:n], labels[perm][:n]

# # select random images and their target indices
# images, labels = select_n_random(trainset.data, trainset.targets)

# # get the class labels for each image
# class_labels = [classes[lab] for lab in labels]

# # log embeddings
# features = images.view(-1, 28 * 28)
# writer.add_embedding(features,
#                     metadata=class_labels,
#                     label_img=images.unsqueeze(1))
# writer.close()
for images, labels in train_dl:
    out = model(images)
    print(out)
    print(labels)
    break
for images, labels in val_dl:
    out = model(images)
    print(out)
    print(labels)
    break
# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    running_loss = 0.0
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
#         result = {}
        lrs = []
#         batch_acc = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
            running_loss += loss.item()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
history = [evaluate(model, val_dl)]
history
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.SGD
%%time
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
plot_losses(history)
plot_lrs(history)
plot_accuracies(history)
accuracy = 0
for images, labels in train_dl:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(make_grid(images[:64].cpu(), nrow=8).permute(1, 2, 0))
    labels = [label.cpu().numpy() for label in labels]
    print([classes[label] for label in labels])
    output = model(images)
    output = output.cpu().detach().numpy()
    predictions = [np.where(ot == np.max(ot))[0][0]  for ot in output]
    print([classes[pred] for pred in predictions])
    print(output)
    for i in range(4):
        if(labels[i] == predictions[i]):
            accuracy += 1      
    break
 
accuracy
# find accuracy
accuracy = 0
count = 0
for images, labels in train_dl:
    count += 1
    labels = [label.cpu().numpy() for label in labels]
    output = model(images)
    output = output.cpu().detach().numpy()
    predictions = [np.where(ot == np.max(ot))[0][0]  for ot in output]
    for i in range(4):
        if(labels[i] == predictions[i]):
            accuracy += 1      
 
print(count*4)
print(accuracy)
print(f'Accuracy: {accuracy/(count*4)}')
PATH = './resnet9.pth'
torch.save(model.state_dict(), PATH)
model = to_device(ResNet9(3, 3), device)
model.load_state_dict(torch.load(PATH))
test_path = '/kaggle/input/hackerearth-friendship-goal-deep-learning/Test Data/Test Data/'
img_list = os.listdir(test_path)
test_image = Image.open(test_path + img_list[0])
test_image
batch_size_test = 1
testset = ImageFolder('/kaggle/input/hackerearth-friendship-goal-deep-learning/Test Data/', train_tfms)
test_dl = DataLoader(testset, batch_size_test, shuffle=True, num_workers=3, pin_memory=True)
final_subs = pd.read_csv('/kaggle/input/hackerearth-friendship-goal-deep-learning/Final_Submission.csv')
row_count = 0
with open('submission_classify_age_groups_val.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Filename', 'Category'])
    for img, label in test_dl:
#         plt.imshow(make_grid(img.cpu()).permute(1, 2, 0))
        img = to_device(img, device)
        output_test = model(img)
        output_test = output_test.cpu().detach().numpy()
        predictions_test = [np.where(ot == np.max(ot))[0][0]  for ot in output_test]
        output_class = classes[predictions_test[0]]
        filename = final_subs.iloc[row_count]['Filename']
#         print(output_class) 
        csvwriter.writerow([filename, output_class])
        row_count += 1
#         break
print(f"Completed {row_count}")
# with lr scheduler getting train accuracy: 80 and test score: 43.03
# with lr scheduler and validation images accuracy: 69 and test score: 37.46
torch.save(model, PATH)
model = torch.load(PATH)
model.eval()
!pip install onnx onnxruntime
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import onnx
import onnxruntime
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# Lets convert into onxx model
# Input to the model
for image, label in test_dl:
    img = image.to(device)
    torch_out = model(img)
    torch.onnx.export(model, img, "age-group-clf.onnx", export_params=True,
                      opset_version=10,do_constant_folding=True,input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
    onnx_model = onnx.load("age-group-clf.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("age-group-clf.onnx")
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print(f'output: {ort_outs}')

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    break

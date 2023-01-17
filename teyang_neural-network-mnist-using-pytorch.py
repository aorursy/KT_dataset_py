import numpy as np 

import pandas as pd 



import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torchvision.datasets import MNIST

import torchvision.transforms as transforms

from torchvision.utils import make_grid

from torch.utils.data import random_split, DataLoader



import matplotlib.pyplot as plt

%matplotlib inline



from shutil import copyfile



# download MNIST data

dataset = MNIST(root='data/', download=True)
# Get the train and test set



dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())



test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

print(len(dataset), len(test_dataset))
# shape of the images



img_tensor, label = dataset[0]

print(img_tensor.shape, label)
# plot image



plt.imshow(img_tensor[0,:,:], cmap='gray')

print('Label:', label)
train_ds, val_ds = random_split(dataset, [50000, 10000])

len(train_ds), len(val_ds)
batch_size = 128



# shuffle so that batches in each epoch are different, and this randomization helps generalize and speed up training

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)

# val is only used for evaluating the model, so no need to shuffle

val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
for images, _ in train_loader:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))

    break
input_size = 28*28 # 784 weights to train, 1 for each pixel

num_classes = 10 # 10 outputs, 10 biases



# Logistic regression model

model = nn.Linear(input_size, num_classes)
print(model.weight.shape)

model.weight
print(model.bias.shape)

model.bias
class MnistModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(input_size, num_classes)

    

    # reshape/flatten input tensor when it is passed to model

    def forward(self, xb):

        xb = xb.reshape(-1, 784) # -1 so that it will work for different batch sizes

        out = self.linear(xb)

        return out



model = MnistModel()
print(model.linear.weight.shape, model.linear.bias.shape)

list(model.parameters())
# check that model class works



for images, labels in train_loader:

    print('images.shape:', images.shape)

    outputs = model(images)

    break

    

print('outputs.shape:', outputs.shape)

print('Sample outputs:\n', outputs[:2].data)
prob = torch.exp(outputs[0])/torch.sum(torch.exp(outputs[0]))

prob
torch.sum(prob)
probs = F.softmax(outputs, dim=1) # output shape is (128, 10), apply softmax to 10 class dim



# Look at sample probabilities

print('Sample probabilities:\n', probs[:2].data)



# Add up the probabilities of an output row

print('Sum:', torch.sum(probs[0]).item())
max_probs, preds = torch.max(probs, dim=1) # probs shape is (128, 10), apply max to 10 class dim; max returns largest element and index of it

print(preds)

print(max_probs)
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1) # probs shape is (128, 10), apply max to 10 class dim; max returns largest element and index of it

    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds
loss_fn = F.cross_entropy
# Loss of current batch of data



loss = loss_fn(outputs, labels) # pass outputs instead of preds as cross_entropy will apply softmax, labels will be converted to one-hot encoded vectors

print(loss)

class MnistModel(nn.Module):

    # this is the constructor, which creates an object of class MnistModel when called

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(input_size, num_classes)

    

    # reshape/flatten input tensor when it is passed to model

    def forward(self, xb):

        xb = xb.reshape(-1, 784) # -1 so that it will work for different batch sizes

        out = self.linear(xb)

        return out

    

    # this is for loading the batch of train image and outputting its loss, accuracy & predictions

    def training_step(self, batch):

        images,labels = batch

        out = self(images)                            # generate predictions

        loss = F.cross_entropy(out, labels)           # compute loss

        acc,preds = accuracy(out, labels)             # calculate accuracy

        return {'train_loss': loss, 'train_acc':acc}

       

    # this is for computing the train average loss and acc for each epoch

    def train_epoch_end(self, outputs):

        batch_losses = [x['train_loss'] for x in outputs]   # get all the batches loss

        epoch_loss = torch.stack(batch_losses).mean()       # combine losses

        batch_accs = [x['train_acc'] for x in outputs]      # get all the batches acc

        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies

        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}

    

    # this is for loading the batch of val/test image and outputting its loss, accuracy, predictions & labels

    def validation_step(self, batch):

        images,labels = batch

        out = self(images)                       # generate predictions

        loss = F.cross_entropy(out, labels)      # compute loss

        acc,preds = accuracy(out, labels)        # calculate accuracy and get predictions

        return {'val_loss': loss, 'val_acc':acc, 'preds':preds, 'labels':labels}

    

    # this is for computing the validation average loss and acc for each epoch

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]     # get all the batches loss

        epoch_loss = torch.stack(batch_losses).mean()       # combine losses

        batch_accs = [x['val_acc'] for x in outputs]        # get all the batches acc

        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}



    # this is for printing out the results after each epoch

    def epoch_end(self, epoch, train_result, val_result):

        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch+1, train_result['train_loss'], train_result['train_acc'], val_result['val_loss'], val_result['val_acc']))

    

    # this is for using on the test set, it outputs the average loss and acc, and outputs the predictions

    def test_prediction(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()                           # combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()                              # combine accuracies

        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]   # combine predictions

        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]   # combine labels

        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(), 'test_preds': batch_preds, 'test_labels': batch_labels}       

        
def evaluate(model, val_loader):

    outputs = [model.validation_step(batch) for batch in val_loader] # perform val for each batch

    return model.validation_epoch_end(outputs)                       # get the results for each epoch 



def fit(model, train_loader, val_loader, epochs, lr, opt_func=torch.optim.SGD):

    history = {}

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        

        # Training phase

        train_outputs = []

        for batch in train_loader:

            outputs = model.training_step(batch)              # compute loss and accuracy

            loss = outputs['train_loss']                      # get loss

            train_outputs.append(outputs)

            loss.backward()                                   # compute gradients

            optimizer.step()                                  # update weights 

            optimizer.zero_grad()                             # reset gradients to zero

        train_results = model.train_epoch_end(train_outputs)  # get the train average loss and acc for each epoch

            

        # Validation phase

        val_results = evaluate(model, val_loader)

        

        # print results

        model.epoch_end(epoch, train_results, val_results)

                

        # save results to dictionary

        to_add = {'train_loss': train_results['train_loss'], 'train_acc': train_results['train_acc'],

                 'val_loss': val_results['val_loss'], 'val_acc': val_results['val_acc']}

        for key,val in to_add.items():

            if key in history:

                history[key].append(val)

            else:

                history[key] = [val]

                

    return history





def test_predict(model, test_loader):

    outputs = [model.validation_step(batch) for batch in test_loader] # perform testing for each batch

    results = model.test_prediction(outputs)                          # get the results

    print('test_loss: {:.4f}, test_acc: {:.4f}'.format(results['test_loss'], results['test_acc']))

    return results['test_preds'], results['test_labels']
# Hyperparameters

lr = 0.001

num_epochs = 10



model = MnistModel()  

history = fit(model, train_loader, val_loader, num_epochs, lr)
# Plot Accuracy and Loss 

epochs=10



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,epochs+1))

ax1.plot(epoch_list, history['train_acc'], label='Train Accuracy')

ax1.plot(epoch_list, history['val_acc'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, epochs+1, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history['train_loss'], label='Train Loss')

ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, epochs+1, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
test_loader = DataLoader(test_dataset, batch_size=256)

preds,labels = test_predict(model, test_loader)
img_num = 100

img_tensor, label = test_dataset[img_num]

plt.imshow(img_tensor[0,:,:], cmap='gray')

print('Label:', label, 'Prediction:', preds[img_num])
# Evaluate Model Performance



# copy .py file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/model-evaluation-utils/model_evaluation_utils.py", dst = "../working/model_evaluation_utils.py")



from model_evaluation_utils import get_metrics



get_metrics(true_labels=labels,

            predicted_labels=preds)
idxs = torch.randint(0, len(test_dataset)+1, (10,)).data # select random test images indices



fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,12))

for c,i in enumerate(idxs):

    img_tensor, label = test_dataset[i]

    ax[c//5][c%5].imshow(img_tensor[0,:,:], cmap='gray')

    ax[c//5][c%5].set_title('Label: {}, Prediction: {}'.format(label, preds[i]), fontsize=25)

    ax[c//5][c%5].axis('off')

class MnistModel_NN(nn.Module):

    # this is the constructor, which creates an object of class MnistModel_NN when called

    def __init__(self, input_size, hidden_size, num_classes):

        super().__init__()

        # hidden layer

        self.linear1 = nn.Linear(input_size, hidden_size)

        # output layer

        self.linear2 = nn.Linear(hidden_size, num_classes)

    

    # reshape/flatten input tensor when it is passed to model

    def forward(self, xb):

        xb = xb.reshape(-1, 784) # -1 so that it will work for different batch sizes

        # Get intermediate outputs using hidden layer

        out = self.linear1(xb)

        # Apply activation function

        out = F.relu(out)

        # Get predictions using output layer

        out = self.linear2(out)

        return out

    

    # this is for loading the batch of train image and outputting its loss, accuracy & predictions

    def training_step(self, batch):

        images,labels = batch

        out = self(images)                            # generate predictions

        loss = F.cross_entropy(out, labels)           # compute loss

        acc,preds = accuracy(out, labels)             # calculate accuracy

        return {'train_loss': loss, 'train_acc':acc}

       

    # this is for computing the train average loss and acc for each epoch

    def train_epoch_end(self, outputs):

        batch_losses = [x['train_loss'] for x in outputs]   # get all the batches loss

        epoch_loss = torch.stack(batch_losses).mean()       # combine losses

        batch_accs = [x['train_acc'] for x in outputs]      # get all the batches acc

        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies

        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}

    

    # this is for loading the batch of val/test image and outputting its loss, accuracy, predictions & labels

    def validation_step(self, batch):

        images,labels = batch

        out = self(images)                       # generate predictions

        loss = F.cross_entropy(out, labels)      # compute loss

        acc,preds = accuracy(out, labels)        # calculate accuracy and get predictions

        return {'val_loss': loss.detach(), 'val_acc':acc, 'preds':preds, 'labels':labels} # detach extracts only the needed number, or other numbers will crowd memory

    

    # this is for computing the validation average loss and acc for each epoch

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]     # get all the batches loss

        epoch_loss = torch.stack(batch_losses).mean()       # combine losses

        batch_accs = [x['val_acc'] for x in outputs]        # get all the batches acc

        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}



    # this is for printing out the results after each epoch

    def epoch_end(self, epoch, train_result, val_result):

        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch+1, train_result['train_loss'], train_result['train_acc'], val_result['val_loss'], val_result['val_acc']))

    

    # this is for using on the test set, it outputs the average loss and acc, and outputs the predictions

    def test_prediction(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()                           # combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()                              # combine accuracies

        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]   # combine predictions

        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]   # combine labels

        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(), 'test_preds': batch_preds, 'test_labels': batch_labels}       

        
torch.cuda.is_available()
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

    

device = get_default_device()

device
def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)
for images, labels in train_loader:

    print(images.shape)

    images = to_device(images, device)

    print(images.device)

    break
class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device) # yield will stop here, perform other steps, and the resumes to the next loop/batch



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
train_loader = DeviceDataLoader(train_loader, device)

val_loader = DeviceDataLoader(val_loader, device)
for xb, yb in val_loader:

    print('xb.device:', xb.device)

    print('yb:', yb)

    break
# Hyperparameters

input_size = img_tensor.shape[1] * img_tensor.shape[2] #728

hidden_size = 128

lr = 0.1

num_epochs = 10



modelNN = MnistModel_NN(input_size, hidden_size, num_classes=10)  

to_device(modelNN, device) # move model parameters to the same device

history = fit(modelNN, train_loader, val_loader, num_epochs, lr)
# Plot Accuracy and Loss 

epochs=10



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,epochs+1))

ax1.plot(epoch_list, history['train_acc'], label='Train Accuracy')

ax1.plot(epoch_list, history['val_acc'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, epochs+1, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history['train_loss'], label='Train Loss')

ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, epochs+1, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
test_loader = DeviceDataLoader(test_loader, device)

preds,labels = test_predict(modelNN, test_loader)
# Evaluate Model Performance



get_metrics(true_labels=labels,

            predicted_labels=preds)
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,12))

for c,i in enumerate(idxs):

    img_tensor, label = test_dataset[i]

    ax[c//5][c%5].imshow(img_tensor[0,:,:], cmap='gray')

    ax[c//5][c%5].set_title('Label: {}, Prediction: {}'.format(label, preds[i]), fontsize=25)

    ax[c//5][c%5].axis('off')
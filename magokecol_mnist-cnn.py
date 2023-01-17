# Some variables that change the values when commited

# Before commiting this variable must be setted to True

COMMIT = True



if COMMIT:

    # show some information during execution

    verbose = 1

    # use a nice number of epochs to train models when commiting

    training_epochs = 45

    # use many models to generate an ensembled model

    # we will train the double of this value and then the best NETS value will be selected

    NETS = 6



else:

    # no commiting (so direct work on jupyter notebook)

    # show more information during execution

    verbose = 2

    # decrease the number of epochs when we are in front of t

    training_epochs = 15

    # use few models to generate an ensembled model    

    NETS = 2

    

# Split original training data for cross validation (20%)

test_sample = 0.2

# Global batch_size that will be used all over this jupyter notebook

batch_size = 100

# Global learning rate

learning_rate = 0.0003

# Every 'lr_step_size' the learning_rate will be modified to 'lr_decrement' proportion

lr_step_size, lr_decrement = 13, 0.3

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import matplotlib.pyplot as plt

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

import copy

import time

import seaborn as sn



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train.shape)

print(test.shape)
train.head()
# Original training labels

Y_train = train["label"].values

# Original training images (784 pixels values per row)

X_train = train.drop(labels = ["label"], axis = 1).values

# Original testing images (784 pixels values per row)

X_test = test.values



split_idx = int((1 - test_sample) * len(Y_train))

x_train, x_test = X_train[0:split_idx], X_train[split_idx:]

y_train, y_test = Y_train[0:split_idx], Y_train[split_idx:]
print(x_train.shape)

print(x_test.shape)
# training data are row images of 784 pixel integer values in range 0-255

print(sorted(pd.unique(x_train.flatten())))
# labels are integer values in range 0-9

print(sorted(pd.unique(y_train.flatten())))
# torchvision.transforms expect data in PIL format (2D arrays with values in range 0-255)

x_train = x_train.astype('uint8').reshape(-1, 28, 28)

x_test = x_test.astype('uint8').reshape(-1, 28, 28)

X_test = X_test.astype('uint8').reshape(-1, 28, 28)



print(x_train.shape)

print(x_test.shape)

print(X_test.shape)
class CustomDataset(Dataset):

    def __init__(self, data, transform=None):

        # data: tuple (X, Y)

        if len(data) not in [1,2]:

            raise "Not expected data shape"

        self.data = data

        self.transform = transform



    def __len__(self):

        return len(self.data[0])



    def __getitem__(self, idx):



        x = self.data[0][idx]

        

        if len(self.data) == 2:

            y = self.data[1][idx]

        else:

            y = torch.zeros(len(x))





        if self.transform:

            x = self.transform(x)

        

        x = x.double()



        return x, y
def create_sets(augmented=False, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.0)):

    # Rotate, translate and scale original data

    # There is only one line that makes the difference (RandomAffine)

    # If we comment this line we will eliminate data augmentation

    transform_train = transforms.Compose([

        transforms.ToPILImage(),

        transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale),

        transforms.ToTensor()

    ])



    # Augmentation must not be used on test images

    transform_test = transforms.Compose([

        transforms.ToPILImage(),

        transforms.ToTensor()

    ])

    

    if augmented:

        train_set = CustomDataset(data=(x_train, y_train), transform=transform_train)

    else:

        # transform_test does not transform the original training data

        train_set = CustomDataset(data=(x_train, y_train), transform=transform_test)        

    test_set =  CustomDataset(data=(x_test, y_test), transform=transform_test)

    

    return train_set, test_set

train_set, test_set = create_sets(augmented=True)



W, H = 10, 10



fig, axes = plt.subplots(H, W, figsize=(10,10))

# show the first H different images in train set

for h in range(H):

    # show W times the same image transformed in and by CustomDataset

    for w in range(W):

        img, label = train_set[h]

        axes[h,w].imshow(img.numpy().squeeze(), cmap='gray')

        axes[h,w].axis('off')

plt.show()
W, H = 12, 6

fig, axes = plt.subplots(H, W, figsize=(16,8))

for h in range(H):

    for w in range(W):

        img, label = train_set[h*H+w]

        axes[h,w].imshow(img.numpy().squeeze(), cmap='gray')

        axes[h,w].axis('off')

        

plt.show()
train_set, test_set = create_sets(augmented=False)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
class CNN(nn.Module):

    def __init__(self, name='', activation=None):

        super(CNN, self).__init__()            

        self.name = name

        self.activation = activation

        self.cv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, bias=True)

        self.pool = nn.MaxPool2d(2)

        self.fc_last = nn.Linear(in_features=5*13*13, out_features=10, bias=True)



    def forward(self, x):

        x = self.cv1(x)

        if self.activation is not None:

            x = self.activation(x)

        x = self.pool(x)

        x = x.view(-1, 5*13*13)

        x = self.fc_last(x)

        return x
model = CNN()

print(model)

print('{} has {} parameters.'.format(model.name, sum(p.numel() for p in model.parameters())))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available(): print('Thanks Kaggle! Guau!!!')

print(device)
def train(model, criterion, optimizer, scheduler, train_loader, test_loader, CNN=True, device='cpu', epochs=2, verbose=1):



    start = time.time()



    model.to(device)

    history = {}

    history['train_losses'] = []

    history['test_losses'] = []

    history['acc'] = []

    history['best_model'] = None

    history['best_epoch'] = -1

    history['best_acc'] = -1

    

    for epoch in range(epochs):

        model.train()

        train_losses = 0.0

        for i, (images, labels) in enumerate(train_loader):

            # Load images

            if CNN:

                images = images.view(-1, 1, 28, 28)

            else:

                images = images.view(-1, 784)

                

            images, labels = images.to(device), labels.to(device)

            

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            train_losses += loss.item()

            loss.backward()

            optimizer.step()

        

        else:



            with torch.no_grad(): #Turning off gradients to speed up

                model.eval()

                correct = 0.0

                total = 0.0

                test_losses = 0.0

                for images, labels in test_loader:

                    if CNN:

                        images = images.view(-1, 1, 28, 28)

                    else:

                        images = images.view(-1, 784)



                    images, labels = images.to(device), labels.to(device)

            

                    outputs = model(images)

                

                    loss = criterion(outputs, labels)

                    test_losses += loss.item()



                    _, predicted = torch.max(outputs.data, 1)



                    total += labels.size(0)

                    # Total correct predictions

                    correct += (predicted == labels).sum()



            accuracy = float(correct) / total

            train_losses = train_losses/len(train_loader)

            test_losses = test_losses/len(test_loader)

            if verbose > 1:

                # Print Loss

                print('Epoch: {}/{}. LR: {}. Train Loss: {:.4f}. Test Loss: {:.4f}. Accuracy: {:.4f}'.format(

                    epoch+1, epochs, model.scheduler.get_lr(), train_losses, test_losses, accuracy))

                        

            history['train_losses'].append(train_losses)

            history['test_losses'].append(test_losses)

            history['acc'].append(accuracy)



            if accuracy > history['best_acc']:

                history['best_acc'] = accuracy

                history['best_model'] = copy.copy(model)

                history['best_epoch'] = epoch

            

        scheduler.step()

    

    total_time = time.time() - start

    

    if verbose > 0:

        print('Best model found at epoch {} with accuracy {}'.format(

                history['best_epoch']+1,

                history['best_acc']

        ))

        n_params = sum(p.numel() for p in model.parameters())

        print('Training time required: {:.2f} secs'.format(total_time))

        print('Training time required per epoch: {:0.2f} secs'.format(total_time/epochs))

        print('Training time per model parameter: {:0.2f} msecs'.format(total_time*1000/n_params))

    

    return history['best_model'], history
# plot every model training and testing losses

def show_losses(history, COLS=None):

    n = len(history)

    if COLS is None:

        COLS = 2

        if n >= 9: COLS = 3

    ROWS = int(np.ceil(n/COLS))

    fig, axes = plt.subplots(ROWS, COLS, figsize=(16, ROWS*4))

    i = 1

    for name, h in history.items():

        ax = plt.subplot(ROWS, COLS, i)        

        ax.set_title(name + ' losses')

        ax.plot(h['train_losses'], label='train')

        ax.plot(h['test_losses'], label='test')

        ax.legend()

        i+=1

    plt.show()



# show all models accuracies in one plot

def show_accuracy(history):

    fig = plt.figure(figsize=(16,8))

    plt.title('Accuracy')

    accs = [(name, h['best_acc']) for name, h in history.items()]

    accs = sorted(accs, key=lambda item: item[1], reverse=True)

    for name, acc in accs:

        print('{}: {:0.3f}'.format(name, acc))

    for name, h in history.items():

        plt.plot(h['acc'], label=name)



    plt.legend()    

    plt.show()
activations = [('none', None),('relu', F.relu), ('tanh', torch.tanh), ('sigmoid', torch.sigmoid)]

models = []



for act_name, act_func in activations:

    name = 'CNN_simple_' + act_name

    model = CNN(name=name, activation=act_func).double()

    # use the model as some kind of toolsbox

    model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=lr_step_size, gamma=lr_decrement)

    models.append(model)
def show_params(models):

    for model in models:  

        print('{} has {} parameters.'.format(model.name, sum(p.numel() for p in model.parameters())))



# Show the number of trainable parameters of every model

show_params(models)
# This method will be helpful to train many models

def train_all(models, epochs=10, device=device, verbose=2):

    history = {}

    n = len(models)

    for i, model in enumerate(models):

        print('Training model {}/{}: {} ...'.format(i+1, n, model.name))

        best_model, h = train(model, criterion=model.criterion, optimizer=model.optimizer,

                              scheduler=model.scheduler, train_loader=train_loader,

                              test_loader=test_loader, CNN=True,

                              epochs=epochs, device=device, verbose=verbose)

        history[best_model.name] = h

    return history
#history = train_all(models, epochs=10, device=device, verbose=verbose)
#show_losses(history, COLS=2)
#show_accuracy(history)
class CNN(nn.Module):

    def __init__(self, name='', filters=32):

        super(CNN, self).__init__()

        self.name = name

        self.filters = filters

        # 1x28x28

        self.cv1 = nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=7, bias=True)

        self.bn1 = torch.nn.BatchNorm2d(filters)

        # filters x 22 x 22

        self.cv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1, bias=True)

        self.bn2 = torch.nn.BatchNorm2d(filters)

        # filters x 20 x 20

        self.pool = nn.MaxPool2d(2)

        # filters x 10 x 10

        self.cv3 = nn.Conv2d(in_channels=filters, out_channels=8, kernel_size=3, bias=True)

        self.bn3 = torch.nn.BatchNorm2d(8)

        # filters x 8 x 8

        self.fc = nn.Linear(in_features=8*8*8, out_features=64, bias=True)

        # 1 x 64

        self.fc_last = nn.Linear(in_features=64, out_features=10, bias=True)



    def forward(self, x):

        x = F.relu(self.cv1(x))

        x = self.bn1(x)

        

        x = F.relu(self.cv2(x))

        x = self.bn2(x)

        

        x = self.pool(x)

        

        x = F.relu(self.cv3(x))

        x = self.bn3(x)

        

        x = x.view(-1, self.fc.in_features)

        x = F.relu(self.fc(x))

        x = self.fc_last(x)

        x = F.softmax(x, dim=1)

        return x
#filters = [32, 64, 128]

filters = [128]

weight_decays = [0.0, 0.005, 0.001]



models = []



for nf in filters:

    for l2 in weight_decays:

        name = 'CNN_filters_{}_L2_{}'.format(nf, l2)

        model = CNN(name=name, filters=nf).double()

        # use the model as some kind of toolsbox

        model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

        model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=l2, amsgrad=False)

        model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=lr_step_size, gamma=lr_decrement)

        models.append(model)
show_params(models)
history_2 = train_all(models, epochs=training_epochs, device=device, verbose=verbose)
show_losses(history_2, COLS=3)
show_accuracy(history_2)
for i, model in enumerate(models):

    torch.save(model, './CNN_{:02d}.pth'.format(i))
#ls -la
#models = []

#for i in range(6):

#    models.append(torch.load('./CNN_{:02d}.pth'.format(i)))

results = []

confusion_matrix = torch.zeros(10, 10)

with torch.no_grad():

    for images, labels in test_loader:



        images = images.view(-1, 1, 28, 28)

        #images = images.view(-1, 784)



        images, labels = images.to(device), labels.to(device)

        r = torch.zeros((len(labels), 10)).double().to(device)

        # sum 10 predictions

        for model in models:

            model.eval()

            model.to(device)

            outputs = model(images)

            r += outputs

        

        # select top 1 prediction ()

        _, preds = torch.max(r, 1)

        # save prediction

        results.append(preds.cpu().numpy())        

        

        for t, p in zip(labels.view(-1), preds.view(-1)):

            confusion_matrix[t.long(), p.long()] += 1



    results = np.array(results).flatten()



print('Confusion Matrix')

print(confusion_matrix)

print('Accuracy per class')

print(confusion_matrix.diag()/confusion_matrix.sum(1))

print('Accuracy: {:.4f}'.format(confusion_matrix.diag().sum()/confusion_matrix.sum().sum().item()))
fig = plt.figure(figsize=(10,8))

#plt.imshow(confusion_matrix, cmap='binary');

sn.heatmap(confusion_matrix, annot=True);
results = np.array(results).flatten()

errors = results - test_set.data[1]

errors = [idx for idx, value in enumerate(errors) if value != 0]

print('Total errors: {}'.format(len(errors)))
total_show = min(120, len(errors))

# See errors

W, H = 12, 6

H = total_show // W

if W*H < total_show:

    H += 1

fig, axes = plt.subplots(H, W, figsize=(16,H*2))

i = 0

for h in range(H):

    for w in range(W):

        if i == total_show: break        

        idx = errors[i]        

        img, label = test_set[idx]

        axes[h,w].set_title('P:{} ({})'.format(results[idx], test_set.data[1][idx]))

        axes[h,w].imshow(img.numpy().squeeze(), cmap='gray')

        axes[h,w].axis('off')

        i += 1

        

plt.show()
train_set, test_set = create_sets(augmented=True, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.0))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
models = []

weight_decay = 0.001



# train the double of NETS models

for i in range(2*NETS):

    name = 'CNN_{}'.format(str(i))

    model = CNN(name=name, filters=128).double()

    # use the model as some kind of toolsbox

    model.criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    model.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False)

    model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=lr_step_size, gamma=lr_decrement)

    models.append(model)
show_params(models)
history_3 = train_all(models, epochs=training_epochs, device=device, verbose=verbose)
show_losses(history_3, COLS=3)
show_accuracy(history_3)
# reorder models based on model best acc

models = [history_3[k]['best_model'] for k in sorted(history_3, key=lambda m: history_3[m]['best_acc'], reverse=True)]

# select best NETS models

models = models[0:NETS]
results = []

confusion_matrix = torch.zeros(10, 10)

with torch.no_grad():

    for images, labels in test_loader:



        images = images.view(-1, 1, 28, 28)

        #images = images.view(-1, 784)



        images, labels = images.to(device), labels.to(device)

        r = torch.zeros((len(labels), 10)).double().to(device)

        # sum 10 predictions

        for model in models:

            model.eval()

            model.to(device)

            outputs = model(images)

            r += outputs

        

        # select top 1 prediction ()

        _, preds = torch.max(r, 1)

        # save prediction

        results.append(preds.cpu().numpy())

        

        for t, p in zip(labels.view(-1), preds.view(-1)):

            confusion_matrix[t.long(), p.long()] += 1



    results = np.array(results).flatten()



print('Confusion Matrix')

print(confusion_matrix)

print('Accuracy per class')

print(confusion_matrix.diag()/confusion_matrix.sum(1))

print('Accuracy: {:.4f}'.format(confusion_matrix.diag().sum()/confusion_matrix.sum().sum().item()))
fig = plt.figure(figsize=(10,8))

#plt.imshow(confusion_matrix, cmap='binary');

sn.heatmap(confusion_matrix, annot=True);
results = np.array(results).flatten()

errors = results - test_set.data[1]

errors = [idx for idx, value in enumerate(errors) if value != 0]

print('Total errors: {}'.format(len(errors)))
total_show = min(120, len(errors))

# See errors

W, H = 12, 6

H = total_show // W

if W*H < total_show:

    H += 1

fig, axes = plt.subplots(H, W, figsize=(16,H*2))

i = 0

for h in range(H):

    for w in range(W):

        if i == total_show: break        

        idx = errors[i]        

        img, label = test_set[idx]

        axes[h,w].set_title('P:{} ({})'.format(results[idx], test_set.data[1][idx]))

        axes[h,w].imshow(img.numpy().squeeze(), cmap='gray')

        axes[h,w].axis('off')

        i += 1

        

plt.show()
for i, model in enumerate(models):

    torch.save(model, './CNN_ensemble_{:02d}.pth'.format(i))
# prepare original test to make predictions

no_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor()

])



testset =  CustomDataset(data=(X_test, np.zeros(len(test))), transform=no_transform)

testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
results = []



with torch.no_grad():

    for images, labels in testloader:

        if CNN:

            images = images.view(-1, 1, 28, 28)

        else:

            images = images.view(-1, 784)



        images, labels = images.to(device), labels.to(device)

        r = torch.zeros((len(labels), 10)).double().to(device)

        for model in models:

            model.to(device)

            outputs = model(images)

            r += outputs

        results.append(torch.argmax(r, 1).to('cpu').numpy())



    results = np.array(results).flatten()
results = [int(i) for i in results]

results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, len(results)+1), name = "ImageId"), results], axis = 1)

submission.to_csv("submission.csv", index=False)
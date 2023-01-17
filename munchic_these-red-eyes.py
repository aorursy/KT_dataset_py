import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# custom dataset wrapper
class RetinasDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.retina_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.retina_df.iloc[idx, 0]) + '.jpeg')
        if not os.path.isfile(img_name):
            return None
        
        image = Image.open(img_name)
        label = int(self.retina_df.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
            
        sample = {'image': image, 'label': label}
        return sample
        
    def __len__(self):
        return len(self.retina_df)


transform_viz = transforms.Compose([
    transforms.Resize((256, 256)),
])
viz = RetinasDataset(csv_file='../input/diabetic-retinopathy-detection/trainLabels.csv',
                         root_dir='../input/diabetic-retinopathy-detection', transform=transform_viz)
rows = 4
cols = 4
fig = plt.figure(figsize=(8, 8))

for i in range(1, rows * cols + 1):
    fig.add_subplot(rows, cols, i)
    plt.imshow(viz[i]['image'])
plt.show()
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

normalize = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
                           # mean and std for 3 channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = RetinasDataset(csv_file='../input/diabetic-retinopathy-detection/trainLabels.csv',
                         root_dir='../input/diabetic-retinopathy-detection', transform=normalize)

np.random.seed(42)
batch_size = 400
val_split = .2
data_sz = len(dataset)

# splitting the data into training and validation sets
indices = list(range(data_sz))
split = int(np.floor(val_split * data_sz))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# random batch samplers
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# some images that are in trainLabels.csv do not exist :(
def rm_na(batch): 
    batch = list(filter(lambda x : x is not None, batch))
    data = [item['image'] for item in batch]
    
    if len(data) > 0:
        data = torch.stack(data).cuda()
        target = torch.Tensor([item['label'] for item in batch]).long().cuda()
        return [data, target]
    
    return None
    

# batch feeders
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=rm_na)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=rm_na)
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = ToyNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0008, momentum=0.07)

for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        if data != None:
            inputs, labels = data
        else:
            continue

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, backprop, update weights
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[Epoch %d, Batch %d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), '/kaggle/working/../input/toy-cnn.pt')
import os
print(os.listdir("/kaggle/working/../input/testset"))
from sklearn.metrics import f1_score

def gen_answers(test_csv, root_dir='../input', transform=None):
    testset = RetinasDataset(csv_file=test_csv, root_dir=root_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=train_sampler, collate_fn=rm_na)
    
    with torch.no_grad():
        for data in test_loader:
            if data != None: 
                inputs, labels = data
                outputs = net(inputs)
                print(labels)
            else:
                print("hi", end=" ") # hm, Kaggle doesn't provide test images in the 1000 imgs :(
        
raw_test_pred = gen_answers('../input/testset/retinopathy_solution.csv', transform=normalize)
all_labels = torch.Tensor().cuda()
all_pred = torch.Tensor().cuda()

with torch.no_grad():
    for data in val_loader:
        inputs, labels = data
        outputs = net(inputs)
        values, pred_labels = outputs.max(1)
        
        all_labels = torch.cat((all_labels.long(), labels)).cuda()
        all_pred = torch.cat((all_pred.long(), pred_labels)).cuda()
        
        print("Actual labels:", labels)
        print("Predicted labels:", pred_labels)
print(f1_score(all_labels.cpu(), all_pred.cpu(), average='micro'))
# calculate metrics globally by counting the total true positives, false negatives and false positives.
print(f1_score(all_labels.cpu(), all_pred.cpu(), average='weighted'))
# calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
print(np.mean(np.array(abs(all_labels - all_pred).cpu())))
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
from torchvision import models

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
import os
print(os.path.isfile('../input/diabetic-retinopathy-detection/1_left.jpeg'))
print(os.path.isfile('../input/diabetic-retinopathy-detection/10_left.jpeg'))

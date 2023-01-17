import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from skimage import io
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import math
import os
from tqdm.notebook import tqdm
from torch.autograd import Variable
class FireModule(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=s1x1, kernel_size=1, stride=1)
        self.expand1x1 = nn.Conv2d(in_channels=s1x1, out_channels=e1x1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(in_channels=s1x1, out_channels=e3x3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.squeeze(x))
        x1 = self.expand1x1(x)
        x2 = self.expand3x3(x)
        x = F.relu(torch.cat((x1, x2), dim=1))
        return x
class SqueezeNet(nn.Module):
    def __init__(self, out_channels):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fire2 = FireModule(in_channels=96, s1x1=16, e1x1=64, e3x3=64)
        self.fire3 = FireModule(in_channels=128, s1x1=16, e1x1=64, e3x3=64)
        self.fire4 = FireModule(in_channels=128, s1x1=32, e1x1=128, e3x3=128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fire5 = FireModule(in_channels=256, s1x1=32, e1x1=128, e3x3=128)
        self.fire6 = FireModule(in_channels=256, s1x1=48, e1x1=192, e3x3=192)
        self.fire7 = FireModule(in_channels=384, s1x1=48, e1x1=192, e3x3=192)
        self.fire8 = FireModule(in_channels=384, s1x1=64, e1x1=256, e3x3=256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fire9 = FireModule(in_channels=512, s1x1=64, e1x1=256, e3x3=256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=12, stride=1)
        # We don't have FC Layers, inspired by NiN architecture.
        
    def forward(self, x):
        # First max pool after conv1
        x = self.max_pool1(self.conv1(x))
        # Second max pool after fire4
        x = self.max_pool2(self.fire4(self.fire3(self.fire2(x))))
        # Third max pool after fire8
        x = self.max_pool3(self.fire8(self.fire7(self.fire6(self.fire5(x)))))
        # Final pool (avg in this case) after conv10
        x = self.avg_pool(self.conv10(self.fire9(x)))
        return torch.flatten(x, start_dim=1)
x = torch.randn(2, 3, 224, 224)
squeezenet = SqueezeNet(out_channels=1000)
out = squeezenet(x)
out.shape
# Number of parameters before pruning (will be exactly equal to that given in paper if out_channels=1000)
sum(p.numel() for p in squeezenet.parameters() if p.requires_grad)
def generate_dictionaries(root_dir="../input/fruits/fruits-360/Training/", train_val_split=0.8):
    train_ids = []
    train_val_dict = {}
    labels = {}
    for root, dirs, files in os.walk(root_dir):
        if(len(dirs) == 0):
            continue
        classes = dirs
        class2num = [i for i in range(len(classes))]
        for cl in class2num:
            for file in os.listdir(os.path.join(root_dir, classes[cl])):
                train_ids.append('_'.join((classes[cl], file)))
                labels['_'.join((classes[cl], file))] = cl
    random.shuffle(x=train_ids)
    train_val_dict['train'] = train_ids[:int(math.ceil(train_val_split*len(train_ids)))]
    train_val_dict['val'] = train_ids[int(math.ceil(train_val_split*len(train_ids))):]
    return train_val_dict, labels, classes
class FruitsDataset(Dataset):
    def __init__(self, list_ids, labels, idx2class, root_dir, transforms=None):
        self.list_ids = list_ids
        self.labels = labels
        self.root_dir = root_dir
        self.idx2class = idx2class
        self.transforms = transforms
    
    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self, index):
        file_name = self.list_ids[index]
        label = torch.tensor(self.labels[file_name])
        # To get image path, join root dir, class folder name, and file_name
        img_path = os.path.join(self.root_dir, file_name.split("_")[0], "_".join(file_name.split("_")[1:]))
        image = io.imread(img_path)
        if self.transforms:
            image = self.transforms(image)
        return [image, label]
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    # Don't forget to toggle to eval mode!
    model.eval()
    
    with torch.no_grad():
        for data, targets in tqdm(loader):
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        print("Correct: {}, Total: {}, Accuracy: {}".format(num_correct, num_samples, int(num_correct) / int(num_samples)))
    # Don't forget to toggle back to model.train() since you're done with evaluation
    model.train()
if __name__ == '__main__':
    
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 16
    EPOCHS = 10
    NUM_CLASSES = 131
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_val_dict, labels, classes = generate_dictionaries()
    train_data = FruitsDataset(list_ids=train_val_dict['train'], 
                               labels=labels, 
                               idx2class=classes, 
                               root_dir="../input/fruits/fruits-360/Training", 
                               transforms=transform_img)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    val_data = FruitsDataset(list_ids=train_val_dict['val'], 
                             labels=labels, 
                             idx2class=classes, 
                             root_dir="../input/fruits/fruits-360/Training", 
                             transforms=transform_img)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    
    squeezenet = SqueezeNet(NUM_CLASSES)
    squeezenet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(squeezenet.parameters(), lr=LEARNING_RATE)
#     data, targets = next(iter(train_loader))
    for epoch in tqdm(range(EPOCHS)):
        losses = []
        with tqdm(total=len(train_val_dict['train']) // BATCH_SIZE) as pbar:
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device=device)
                targets = targets.to(device=device)

                scores = squeezenet(data)
                loss = criterion(scores, targets)
                losses.append(loss)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#                 print(loss.item())
                pbar.update(1)
        print("Cost at epoch {} is {}".format(epoch, sum(losses) / len(losses)))
        print("Calculating Validation Accuracy...")
        check_accuracy(val_loader, squeezenet)
        print("Calculating Train Accuracy...")
        check_accuracy(train_loader, squeezenet)
!pip install torchviz graphviz
from torchviz import make_dot
from graphviz import Source
x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False).to(device)
out = squeezenet(x)
model_arch = make_dot(out)
Source(model_arch).render("../working/squeezenet_architecture")
# Download visualization from right side panel under '/kaggle/working'
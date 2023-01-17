data_from = "kaggle"

if data_from == "colab":

    from google.colab import files

    files.upload()

    !pip install -q kaggle

    !mkdir -p ~/.kaggle

    !cp kaggle.json ~/.kaggle/

    !chmod 600 ~/.kaggle/kaggle.json

    !mkdir /content/cactus/

    !mkdir /content/cactus/data/

    !kaggle competitions download -c aerial-cactus-identification

    !mv test.zip train.zip train.csv sample_submission.csv /content/cactus/data/

    import zipfile

    def unzip(path):

        with zipfile.ZipFile(path,"r") as z:

            z.extractall('.')

    train_zip = "/content/cactus/data/train"

    unzip(train_zip+".zip")

    test_zip = "/content/cactus/data/test"

    unzip(test_zip+".zip")

    train_dir = "/content/train"

    test_dir = "/content/test"

    train_labels_path = "/content/cactus/data/train.csv"

    data_folder_path = "/content"

    csv_path = "/content/cactus/data/sample_submission.csv" 

elif data_from == "local":

    data_folder_path = "aerial-cactus-identification"

    train_dir = "aerial-cactus-identification/train"

    test_dir = "aerial-cactus-identification/test"

    train_labels_path = "aerial-cactus-identification/train.csv"

    csv_path = "aerial-cactus-identification/sample_submission.csv" 

elif data_from == "kaggle":

    import zipfile

    def unzip(path):

        with zipfile.ZipFile(path,"r") as z:

            z.extractall('.')

    train_zip = "../input/aerial-cactus-identification/train"

    unzip(train_zip+".zip")

    test_zip = "../input/aerial-cactus-identification/test"

    unzip(test_zip+".zip")

    train_dir = "/kaggle/working/train"

    test_dir = "/kaggle/working/test"

    train_labels_path = "../input/aerial-cactus-identification/train.csv"

    data_folder_path = "/kaggle/working"

    csv_path = "../input/aerial-cactus-identification/sample_submission.csv" 

    
import os

from datetime import datetime



import pandas as pd

import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from tqdm.notebook import tqdm

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, models

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
torch.manual_seed(100)

np.random.seed(100)
class CactusDataset2(Dataset):

    def __init__(self, labels_csv_path, root_dir, transform=None):

        """

        Inputs:

          labels_csv_path (string): path to csv containing images' names and labels

          root_dir (string): path to directory with images

          transform (callable, optional): Optional transform to be applied on images"""



        df = pd.read_csv(labels_csv_path)

        self.labels = df['has_cactus']

        # Read data to memory

        self.images = []

        print("Reading images to memory")

        for _, img_name in tqdm(df['id'].items()):

            img_path = os.path.join(root_dir, img_name)

            image = plt.imread(img_path)

            self.images.append(image)

            

        if transform is None:

            self.transform = transforms.Compose([transforms.ToTensor(),

                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        else:

            self.transform = transform



    def __len__(self):

        return len(self.labels)



    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.item()

        label = self.labels.iloc[idx].astype(np.float32).reshape(-1)

        label = torch.tensor(label).long()

        image = self.images[idx]

        

        image = self.transform(image)

        return image, label
resize = 128

transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(resize),

                                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),

                                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),

                                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(1, 1.1)),

                                transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(resize), transforms.ToTensor(),

                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



dataset_train = CactusDataset2(train_labels_path, train_dir, transform=transform)

dataset_val = CactusDataset2(train_labels_path, train_dir, transform=test_transform)

testset = CactusDataset2(csv_path, test_dir, transform=test_transform)



# Split training dataset to train and validation



batch_size = 64



num_train = len(dataset_train)

valid_percent = 0.01

valid_size = round(valid_percent * num_train)

train_size = num_train - valid_size

indices = list(range(num_train))

np.random.shuffle(indices)

train_idx, valid_idx = indices[:train_size], indices[train_size:]

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)





train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=4)

valid_loader = DataLoader(dataset_val, batch_size=batch_size, sampler=valid_sampler, num_workers=4)

test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
# Test the augmentation

indices = np.random.choice(list(range(len(dataset_val))), 16)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))

for i, idx in enumerate(indices):

    img, label = dataset_val[idx]

    img = 0.22*img + 0.45

    img = img.numpy().transpose(1, 2, 0)

    axes.flatten()[i].imshow(img)

    label = "Cactus" if label==1 else "Garbage"

    axes.flatten()[i].set_title(str(label))

    axes.flatten()[i].set_axis_off()

fig.suptitle("Validation Sample")

# plt.tight_layout()



indices = np.random.choice(list(range(len(dataset_train))), 16)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))

for i, idx in enumerate(indices):

    img, label = dataset_train[idx]

    img = 0.22*img + 0.45

    img = img.numpy().transpose(1, 2, 0)

    axes.flatten()[i].imshow(img)

    label = "Cactus" if label==1 else "Garbage"

    axes.flatten()[i].set_title(str(label))

    axes.flatten()[i].set_axis_off()

fig.suptitle("Training Sample")

# plt.tight_layout()
# Credit fast.ai https://github.com/fastai/fastai/blob/master/fastai/layers.py#L176

class AdaptiveConcatPool2d(nn.Module):

    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz=None):

        super().__init__()

        "Output will be 2*sz or 2 if sz is None"

        self.output_size = sz or 1

        self.ap = nn.AdaptiveAvgPool2d(self.output_size)

        self.mp = nn.AdaptiveMaxPool2d(self.output_size)



    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

    

class TransferHead(nn.Module):

    def __init__(self, in_features):

        """in_features is the number of channels in the last conv layer of the base model"""

        super().__init__()

        self.avg = AdaptiveConcatPool2d()

        self.in_features = in_features

        self.layer1 = nn.Sequential(nn.BatchNorm1d(2*in_features), nn.Dropout(0.25),

                                    nn.Linear(2*in_features, 512), nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.BatchNorm1d(512), nn.Dropout(0.5),

                                    nn.Linear(512, 2))

    

    def forward(self, x):

        out = self.avg(x)

        out = out.view(-1, self.in_features*2)

        out = self.layer1(out)

        out = self.layer2(out)

        return out

    

class TransCactusTron(nn.Module):

    def __init__(self, freeze=True):

        super().__init__()

        base_arch = models.densenet161(pretrained=True)

        n_channels = base_arch.classifier.in_features

        self.body = nn.Sequential(base_arch.features, nn.ReLU(inplace=True))

        self.head = TransferHead(n_channels)

        

        for x in self.body.parameters():

            x.requires_grad = False

        for x in self.body.modules():

            if isinstance(x, nn.modules.batchnorm._BatchNorm):

                x.bias.requires_grad = True

                x.weight.requires_grad = True

                x.reset_running_stats()

                

    

    def forward(self, x):

        out = self.body(x)

        return self.head(out)

    

    def predict(self, loader, device=torch.device('cpu')):

        y_pred = torch.tensor([])

        with torch.no_grad():

            for data in loader:

                img = data[0].to(device)

                curr_pred = self(img)

                y_pred = torch.cat((y_pred, curr_pred.cpu().detach()))

        softmax = nn.Softmax(dim=1)

        y_pred = softmax(y_pred)

        return y_pred[:, 1].numpy()

    

    def unfreeze(self):

        for x in self.body.parameters():

            x.requires_grad = True

    



find_lr = True



if find_lr:

    # According to https://arxiv.org/abs/1506.01186

    num_steps = len(train_loader)

    lrs = np.logspace(-8, 0, num=num_steps)

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    criterion = nn.CrossEntropyLoss(reduction='mean')

    model = TransCactusTron()

    model.train()

    model.to(device)



    optimizer = torch.optim.Adam(model.parameters() ,lr=lrs[0])

    train_loss = []

    best_loss = np.inf

    i = 0



    model.train()

    for img, label in tqdm(train_loader):

        img, label = img.to(device), label.to(device).squeeze()

        optimizer.zero_grad()

        predicted_label = model(img)

        loss = criterion(predicted_label, label)

        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())

        if loss.item() < best_loss:

            best_loss = loss.item()



        if loss.item() > 20*best_loss:

            print("Loss diverged")

            break



        optimizer.param_groups[0]['lr'] = lrs[i]

        i += 1



    # exponential smoothing

    train_loss = pd.Series(train_loss)

    train_loss_smooth = train_loss.ewm(alpha=0.02).mean()

    plt.plot(lrs[:len(train_loss)], train_loss_smooth)

    plt.grid()

    plt.xscale('log')
####################

# HYPER PARAMETERS #

####################



lr = 3e-2

epochs = 10



device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

criterion = nn.CrossEntropyLoss(reduction='mean')



tensorboard = False



############

# Training #

############



if tensorboard == True:

    time = datetime.now().strftime("%d%m-%H%M")

    train_id = f"{time}_lr={lr}_epochs={epochs}_wd={wd}_Trans_difflr_Adam"

    writer = SummaryWriter(log_dir=f"runs/{train_id}")

    



model = TransCactusTron()

model.to(device)



global_step = 0



print(f'\nSTART TRAINING\n')

parameters = [{'params': model.body.parameters()},

              {'params': model.head.parameters()}]

optimizer = torch.optim.Adam(parameters, lr=lr)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[lr/10, lr],

                                                epochs=epochs, steps_per_epoch=len(train_loader))



for epoch in tqdm(range(epochs), desc='Epochs'):

    train_loss = 0.0

    valid_loss = 0.0

    start_time = datetime.now()



    #########

    # Train #

    #########

    i = 0

    model.train()

    for img, label in tqdm(train_loader, desc="Iteration"):

        img, label = img.to(device), label.to(device).squeeze()

        optimizer.zero_grad()

        predicted_label = model(img)

        loss = criterion(predicted_label, label)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        scheduler.step()



    ############

    # Validate #

    ############



    model.eval()

    with torch.no_grad():

        for img, label in valid_loader:

            img, label = img.to(device), label.to(device).squeeze()

            predicted_label = model(img)

            loss = criterion(predicted_label, label)

            valid_loss += loss.item()



    ###########    

    # Logging #

    ###########



    avg_train_loss = train_loss/len(train_loader)

    avg_valid_loss = valid_loss/len(valid_loader)

    delta = datetime.now()-start_time

    print(f"Epoch: {epoch}\tTrain Loss: {avg_train_loss:.6f}\tVal Loss: {avg_valid_loss:.6f}\t Time:{delta}")

    if tensorboard == True:

        writer.add_scalar("Loss/Train", avg_train_loss, global_step=global_step)

        writer.add_scalar("Loss/Validation", avg_valid_loss, global_step=global_step)

        writer.add_scalars("Loss/Cross",{"Train": avg_train_loss,

                                                       "Validation": avg_valid_loss},

                           global_step=global_step)

        global_step += 1

    torch.save(model.state_dict(), f'model.pt')





print("FINISHED TRAINING")



if tensorboard == True:

    writer.close()
df = pd.read_csv(csv_path)

y_pred = np.zeros(len(testset))

model.eval()

y_pred = model.predict(test_loader, device)



df['has_cactus'] = y_pred

df.to_csv("submission.csv", index=False)



df.head(15)
if data_from == "kaggle":

    import shutil

    shutil.rmtree('/kaggle/working/train')

    shutil.rmtree('/kaggle/working/test')

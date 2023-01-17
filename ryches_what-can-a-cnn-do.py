import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preconv1 = nn.Conv2d(3, 2, 1)
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.preconv1(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x_max = x.max(axis = 2)[0]
        x_min = (-x).max(axis = 2)[0]
        
        x_max = (x_max - 1)*128
        x_min = (-x_min+1)*128
        
        x = torch.cat((x_max, x_min), axis = -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = F.elu(self.fc5(x))
        x = F.elu(self.fc6(x))
        x = F.elu(self.fc7(x))
        x = self.fc8(x)
        
        return torch.squeeze(x)
net = Net()
net = net.to("cuda")
class resnet_model(nn.Module):
    def __init__(self):
        super(resnet_model, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.backbone(x)
        return x.squeeze()


resnet = resnet_model().to("cuda")
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
sample = np.zeros((3, 128, 128))

class measure_dataset(Dataset):
    def __init__(self, size = (128, 128), data_size = 10000):
        self.size = size
        self.data_size = data_size
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        sample = np.zeros((3, self.size[0], self.size[1]))
        sample[1, :, :] = np.arange(0, self.size[0])[None, :]/(self.size[0] + 1)
        sample[2, :, :] = np.arange(0, self.size[1])[:, None]/(self.size[1] + 1)
        
        first_x = np.random.randint(0, self.size[0])
        second_x = np.random.randint(0, self.size[0])
        first_y = np.random.randint(0, self.size[1])
        second_y = np.random.randint(0, self.size[1])
        
        sample[0, first_y, first_x] = 1
        sample[0, second_y, second_x] = -1
        return sample, np.sqrt((first_x - second_x)**2 + (first_y - second_y)**2)
        
        
trn_dataset = measure_dataset()
trn_dataloader = DataLoader(trn_dataset, batch_size=512, drop_last = True)
for x,y in trn_dataloader:
    break
x.shape
dual_channel = np.zeros((128, 128, 3))
dual_channel[:, :, 0] = x[0, 0] * x[0, 1]
dual_channel[:, :, 1] = x[0, 0] * x[0, 2]
np.unique(dual_channel[:, :, 0])
np.unique(dual_channel[:, :, 1])
plt.imshow(dual_channel[:, :, 0])
plt.imshow(x[0].permute(1, 2, 0))
print(y[0])
y[0]
plt.imshow(x[0, 2])
def train_model(trn_dataloader, model, epochs = 10):
    criterion = nn.L1Loss()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    running_loss = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trn_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(torch.float32).to("cuda")
            labels = labels.to(torch.float32).to("cuda")
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.append(loss.cpu().detach().numpy())
        print("epoch", str(epoch) + ":",  np.mean(np.array(running_loss[-100:]).reshape(-1,)))
    return running_loss
net.preconv1.weight.data[0, 0] = 1
net.preconv1.weight.data[0, 1] = 1
net.preconv1.weight.data[0, 2] = 0

net.preconv1.weight.data[1, 0] = 1
net.preconv1.weight.data[1, 1] = 0
net.preconv1.weight.data[1, 2] = 1

net.preconv1.bias.data[:] = 0
net.preconv1.requires_grad_ = False
net.preconv1.weight.data
net.preconv1.bias.data
plt.imshow(x[0, 0])
net(x[0:1].to("cuda"))
running_loss = train_model(trn_dataloader, net, 100)
running_loss = train_model(trn_dataloader, resnet, 100)
def validate_model(trn_dataloader, model):
    val_criterion = nn.L1Loss(reduction = 'none')
    running_loss = []
    running_labels = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(trn_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(torch.float32).to("cuda")
            labels = labels.to(torch.float32).to("cuda")
            preds = model(inputs)
            loss = val_criterion(preds, labels)
            running_loss.append(loss.cpu().detach().numpy())
            running_labels.append(labels.cpu().detach().numpy())
    return np.array(running_loss).reshape(-1,), np.array(running_labels).reshape(-1,)
running_loss, running_labels = validate_model(trn_dataloader, net)
plt.scatter(running_labels, running_loss)
running_loss, running_labels = validate_model(trn_dataloader, resnet)
plt.scatter(running_labels, running_loss)
running_loss = train_model(trn_dataloader, net, 100)
x.shape
x = x.to(torch.float32).to('cuda')
layers = []
with torch.no_grad():
    for i, module in enumerate(net.modules()):
        try:
            if i == 0:
                continue
            print(module)
            if len(layers) == 0:
                layers.append(module(x))
            else:
                layers.append(module(layers[-1]))
        except:
            print("failed layer")
from mpl_toolkits.axes_grid1 import ImageGrid
with torch.no_grad():
    for i, layer in enumerate(layers):
        layer = layer[0]
        layer = layer[:32]
        print("layer" + str(i))
        fig = plt.figure(figsize=(len(layer)*4, len(layer)*4))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(int(np.ceil(len(layer)/8)), 8),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, layer.cpu().detach().numpy()):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
        print(im.shape)
        plt.show()
big_trn_dataset = measure_dataset((256, 256))
big_trn_dataloader = DataLoader(big_trn_dataset, batch_size=64, drop_last = True)
running_loss, running_labels = validate_model(big_trn_dataloader, net)
plt.scatter(running_labels, running_loss)
running_loss, running_labels = validate_model(big_trn_dataloader, resnet)
plt.scatter(running_labels, running_loss)
def validate_model(trn_dataloader, model):
    val_criterion = nn.L1Loss(reduction = 'none')
    running_loss = []
    running_labels = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(trn_dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(torch.float32).to("cuda")
            labels = labels.to(torch.float32).to("cuda")
            preds = model(inputs)
            loss = val_criterion(preds, labels)
            running_loss.append(loss.cpu().detach().numpy())
            running_labels.append(labels.cpu().detach().numpy())
            predictions.append(preds.cpu().detach().numpy())
    return np.array(running_loss).reshape(-1,), np.array(running_labels).reshape(-1,), np.array(predictions).reshape(-1,)
_, _, predictions = validate_model(big_trn_dataloader, resnet)
predictions_df = pd.DataFrame(predictions)
print(predictions_df.describe())
predictions_df.hist(bins = 20)
plt.title("distribution of predictions of resnet")
_, _, predictions = validate_model(big_trn_dataloader, net)
predictions_df = pd.DataFrame(predictions)
print(predictions_df.describe())
predictions_df.hist(bins = 20)
plt.title("distribution of predictions of custom CNN")
sml_trn_dataset = measure_dataset((110, 110))
sml_trn_dataloader = DataLoader(sml_trn_dataset, batch_size=64, drop_last = True)
running_loss, running_labels, predictions = validate_model(sml_trn_dataloader, resnet)
plt.scatter(running_labels, running_loss)

predictions_df = pd.DataFrame(predictions)
print(predictions_df.describe())
predictions_df.hist(bins = 20)
plt.title("distribution of predictions of resnet")
running_loss, running_labels, predictions = validate_model(sml_trn_dataloader, net)
plt.scatter(running_labels, running_loss)

predictions_df = pd.DataFrame(predictions)
print(predictions_df.describe())
predictions_df.hist(bins = 20)
plt.title("distribution of predictions of custom CNN")
losses = []
for i in range(64, 256, 10):
    sml_trn_dataset = measure_dataset((i, i))
    sml_trn_dataloader = DataLoader(sml_trn_dataset, batch_size=64, drop_last = True)
    running_loss, running_labels, predictions = validate_model(sml_trn_dataloader, resnet)
    plt.scatter(running_labels, running_loss)

    predictions_df = pd.DataFrame(predictions)
    print("image size:", str(i))
    print(predictions_df.describe())
    predictions_df.hist(bins = 20)
    plt.title("distribution of predictions of resnet")
    plt.show()
    losses.append(np.mean(running_loss))
    
plt.plot(list(range(64, 256, 10)), losses)
losses = []
for i in range(110, 256, 10):
    sml_trn_dataset = measure_dataset((i, i))
    sml_trn_dataloader = DataLoader(sml_trn_dataset, batch_size=64, drop_last = True)
    running_loss, running_labels, predictions = validate_model(sml_trn_dataloader, net)
    plt.scatter(running_labels, running_loss)

    predictions_df = pd.DataFrame(predictions)
    print("image size:", str(i))
    print(predictions_df.describe())
    predictions_df.hist(bins = 20)
    plt.title("distribution of predictions of custom NN")
    plt.show()
    losses.append(np.mean(running_loss))
    

plt.plot(list(range(110, 256, 10)), losses)

import numpy as np

import pandas as pd

from PIL import Image

from pathlib import Path

import random, math, cv2

from tqdm import tqdm_notebook as tqdm



import torch

from torch import nn

from torchvision.datasets import MNIST

from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from torchvision.transforms import Compose, ToTensor



import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore")
MNIST_DATA_DIR = Path('/kaggle/working')

MODEL_FILE = Path('best_source_weights_mnist.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



batch_size = 64

epochs = 5
def visualize_digits(dataset, k=80, mnistm=False, cmap=None, title=None):

    

    ncols = 20

    indices = random.choices(range(len(dataset)), k=k)

    nrows = math.floor(len(indices)/ncols)

    

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols,nrows+0.4), gridspec_kw=dict(wspace=0.1, hspace=0.1), subplot_kw=dict(yticks=[], xticks=[]))

    axes_flat = axes.reshape(-1)

    fig.suptitle(title, fontsize=20)

    

    for list_idx, image_idx in enumerate(indices[:ncols*nrows]):

        ax = axes_flat[list_idx]

        image = dataset[image_idx][0]

        image = image.numpy().transpose(1, 2, 0)

        ax.imshow(image, cmap=cmap)

        

class GrayscaleToRgb:

    """Convert a grayscale image to rgb"""

    def __call__(self, image):

        image = np.array(image)

        image = np.dstack([image, image, image])

        return Image.fromarray(image)
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(3, 10, kernel_size=5),

            nn.MaxPool2d(2),

            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=5),

            nn.MaxPool2d(2),

            nn.Dropout2d(),

        )

        

        self.classifier = nn.Sequential(

            nn.Linear(320, 50),

            nn.ReLU(),

            nn.Dropout(),

            nn.Linear(50, 10),

            nn.LogSoftmax(),

        )



    def forward(self, x):

        features = self.feature_extractor(x)

        features = features.view(x.shape[0], -1)

        logits = self.classifier(features)

        return logits
source_model = Net().to(device)

if MODEL_FILE.exists():

    source_model.load_state_dict(torch.load(MODEL_FILE))



train_dataset = MNIST(MNIST_DATA_DIR / 'mnist', train=True, download=True, transform=Compose([GrayscaleToRgb(), ToTensor()]))

test_dataset = MNIST(MNIST_DATA_DIR / 'mnist', train=False, download=True, transform=Compose([GrayscaleToRgb(), ToTensor()]))



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)



source_optim = torch.optim.Adam(source_model.parameters(), lr=0.002)

criterion = nn.NLLLoss()
visualize_digits(dataset=train_dataset, k=120, cmap='gray', title='Sample MNIST Images')
train_losses, train_accuracies, train_counter = [], [], []

test_losses, test_accuracies = [], []

test_counter = [idx*len(train_loader.dataset) for idx in range(0, epochs+1)]



def train(epoch):

    train_loss, train_accuracy = 0, 0

    source_model.train()

    tqdm_bar = tqdm(train_loader, desc=f'Training Epoch {epoch} ', total=int(len(train_loader)))

    for idx, (images, labels) in enumerate(tqdm_bar):

        images, labels = images.to(device), labels.to(device)

        source_optim.zero_grad()

        with torch.set_grad_enabled(True):

            outputs = source_model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            source_optim.step()

        train_loss += loss.item()

        train_losses.append(loss.item())

        outputs = torch.argmax(outputs, dim=1).type(torch.FloatTensor).to(device)

        train_batch_accuracy = torch.mean((outputs == labels).type(torch.FloatTensor)).item()

        train_accuracy += train_batch_accuracy

        train_accuracies.append(train_batch_accuracy)

        tqdm_bar.set_postfix(train_loss=(train_loss/(idx+1)), train_accuracy=train_accuracy/(idx+1))

        train_counter.append(idx*batch_size + images.size(0) + epoch*len(train_dataset))



def test():

    test_loss, test_accuracy = 0, 0

    source_model.eval()

    tqdm_bar = tqdm(test_loader, desc=f'Testing ', total=int(len(test_loader)))

    for idx, (images, labels) in enumerate(tqdm_bar):

        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():

            outputs = source_model(images)

            loss = criterion(outputs, labels)

        test_loss += loss.item()

        outputs = torch.argmax(outputs, dim=1).type(torch.FloatTensor).to(device)

        test_accuracy += torch.mean((outputs == labels).type(torch.FloatTensor)).item()

        tqdm_bar.set_postfix(test_loss=(test_loss/(idx+1)), test_accuracy=test_accuracy/(idx+1))

    test_losses.append(test_loss/len(test_loader))

    test_accuracies.append(test_accuracy/len(test_loader))

    if np.argmax(test_accuracies) == len(test_accuracies)-1:

        torch.save(source_model.state_dict(), 'best_source_weights_mnist.pth')

        

test()

for epoch in range(epochs):

    train(epoch)

    test()
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_counter, y=train_losses, mode='lines', name='Train loss'))

fig.add_trace(go.Scatter(x=test_counter, y=test_losses, marker_symbol='star-diamond', 

                         marker_color='orange', marker_line_width=1, marker_size=9, mode='markers', name='Test loss'))

fig.update_layout(

    width=1000,

    height=500,

    title="Train vs. Test Loss",

    xaxis_title="Number of training examples seen",

    yaxis_title="Negative Log Likelihood loss"),

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_counter, y=train_accuracies, mode='lines', name='Train loss'))

fig.add_trace(go.Scatter(x=test_counter, y=test_accuracies, marker_symbol='star-diamond', 

                         marker_color='orange', marker_line_width=1, marker_size=9, mode='markers', name='Test Accuracy'))

fig.update_layout(

    width=1000,

    height=500,

    title="Train vs. Test Accuracy",

    xaxis_title="Number of training examples seen",

    yaxis_title="Accuracy")

fig.show()
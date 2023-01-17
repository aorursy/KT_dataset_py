import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms
BATCH_SIZE = 50

EPOCHS = 30

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([

    transforms.RandomResizedCrop(150),

    transforms.ToTensor(),

    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])

 

dataset_train = datasets.ImageFolder('/kaggle/input/cat-and-dog/training_set/training_set', transform)

dataset_test = datasets.ImageFolder('/kaggle/input/cat-and-dog/test_set/test_set', transform)



train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)

        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3)

        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3)

        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 128, 3)

        self.max_pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(6272, 512)

        self.fc2 = nn.Linear(512, 1)

        

    def forward(self, x):

        in_size = x.size(0)

        x = self.conv1(x)

        x = F.relu(x)

        x = self.max_pool1(x)

        x = self.conv2(x)

        x = F.relu(x)

        x = self.max_pool2(x)

        x = self.conv3(x)

        x = F.relu(x)

        x = self.max_pool3(x)

        x = self.conv4(x)

        x = F.relu(x)

        x = self.max_pool4(x)

        # Expand

        x = x.view(in_size, -1)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        x = torch.sigmoid(x)

        return x
model = ConvNet().to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

 

def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device).float().unsqueeze(dim = 1) # unsqueeze instead of reshape

        optimizer.zero_grad()

        output = model(data)

        loss = F.binary_cross_entropy(output, target)

        loss.backward()

        optimizer.step()

        if(batch_idx+1)%10 == 0: 

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),

                100. * (batch_idx+1) / len(train_loader), loss.item()))
def test(model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device).float().unsqueeze(dim = 1)

            output = model(data)

            # print(output)

            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item() # summing the loss

            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)

            correct += pred.eq(target.long()).sum().item()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))
for epoch in range(1, EPOCHS + 1):

    train(model, DEVICE, train_loader, optimizer, epoch)

    test(model, DEVICE, test_loader)
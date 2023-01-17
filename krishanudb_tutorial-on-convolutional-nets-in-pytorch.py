import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
input_size = [28, 28]
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001
traindata = torchvision.datasets.MNIST(root="./data", train = True, transform=transforms.ToTensor(), download = True)
testdata = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download = True)
print(traindata.train_data.size())
print(testdata.test_data.size())
trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=batch_size, shuffle=False)
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding = 2),
                                   nn.BatchNorm2d(6),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=3, padding = 2),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding = 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride = 2))
        self.output = nn.Linear(4 * 4 * 32, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(-1, 4 * 4 * 32)
        out = self.output(out)
        return out

model = ConvNet(10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 200 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(trainloader), loss.item()))
with torch.no_grad():
    correct = 0
    total = 0
    for i, (image, labels) in enumerate(testloader):
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total = labels.size(0)
        
print("Prediction accuracy of the above model is: {}".format(float(correct) / float(total)))
        
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
num_epochs = 2
learning_rate = 0.001
transform = transforms.Compose([transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()])
# Download the CIFAR 10 dataset
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, batch_size=100)
testloader = torch.utils.data.DataLoader(dataset=test_data, shuffle=False, batch_size=100)
# Please note that the commented print statements were just to help figure out the dimensions properly. Remove the comment tag from the 
# print statements in order to see the actual dimensions of the tensors.



def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    """This is the residual block class which will contain one block with the skip connections. Please keep in
    mind the dimensions of the tensors.
    Objects of this class will be called by the main ResNet class while initiating the network"""
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride) # Stride can be one or two. If one, downsampling is not done
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels) # The stride is one irrespective of anything. No downsampling here.
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # The downsample layer as decided in the ResNet class
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
#         print(self.conv1)
        out = self.bn1(out)
        out = self.relu(out)
#         print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
#             print("Downsampling Done")
#             print(self.downsample)
            residual = self.downsample(x)
#         print(out.shape, "\t", residual.shape)
        out += residual # Adding the residual tensor to the main tensor
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0]) # Residual Layer 1 is constructed
        self.layer2 = self.make_layer(block, 32, layers[0], 2) # Residual layer 2 with 32 channels and stride 2 is constructed. Downsampling happens because of the stride
        self.layer3 = self.make_layer(block, 64, layers[1], 2) # Residual layer 3 with 64 channels and stride 2 is constructed. Downsampling happens.
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels): # IF stride != 2 downsampling happens. 
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels)) # Downsampling layer to downsample the residuals is constructed
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # Residual blocks added with/wiothout downsampling depending upon stride and channel sizes
        self.in_channels = out_channels # The number of output channels of one block is the number of input blocks of the next
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(* layers) # The layers list(with all the residual blocks are added to the sequence)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = ResNet(ResidualBlock, [2, 2, 2, 2])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
total_step = len(trainloader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 250 == 0:
            print("Epoch: {}/{};\tStep: {}/{};\tLoss: {}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
    if (epoch + 1) % 4 == 0:
        curr_lr = curr_lr / 3.
        update_lr(optimizer, curr_lr)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * float(correct) / float(total)))

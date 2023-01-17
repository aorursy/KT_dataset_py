import torch
from torch import nn

# input tensor: 64*1000
input_tensor = torch.randn(64,1000)

# linear layer: 1000 inputs and 100 output units
linear_layer = nn.Linear(1000, 100)

# output of the linear layer
output = linear_layer(input_tensor)
print(output.size())
from torch import nn

# define a two-layer model
model = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(), # Activation function
    nn.Linear(5, 1)
    )

print(model)
from torch import nn
import torch.nn.functional as F

# Define the model
class Model(nn.Module):
    
    # specify the layers
    def __init__(self):
        super(Model, self).__init__()
        # 1 input image channel, 20 output channels, 5x5 square convolution, 1x1 stride
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # 20 input channels, 50 output channels, 5x5 square convolution, 1x1 stride
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # 4x4 from image dimension, 500 output units
        self.fc1 = nn.Linear(50*4*4, 500)
        # 500 input units, 10 output units
        self.fc2 = nn.Linear(500, 10)
        
    # define the forward method
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    # calculate number of flat features
    def num_flat_features(self, x):
        size = x.size()[1:] # C * W * H: 50*4*4
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

# create an object of Model
model = Model()
print(model)
!pip install torchsummary
from torchsummary import summary
summary(model, input_size=(1, 28, 28))

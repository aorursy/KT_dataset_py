from google.colab import drive
drive.mount('/content/drive')
import torch
import torch.nn as nn
# target output size of 5x7
m = nn.AdaptiveAvgPool2d((1,1))
input = torch.randn(1, 64, 8, 9)
output = m(input)
output.size()
#m = nn.MaxPool2d((3, 2), stride=(2, 1))
m = nn.MaxPool2d((3), stride=(2))
input = torch.randn(20, 16, 50, 50)
output = m(input)
output.size()
n = nn.MaxPool2d((2))
input = torch.randn(20, 16, 50, 50)
output = n(input)
output.size()
m = nn.Linear(2, 10)
m
input = torch.randn(3, 4)
print(input.size())
print(input)
#output = m(input)
#print(output.size())
# Do not run, it's part of model's script.

def forward(self, xb):
        print('xb:', xb.size())
        x = self.network(xb)
        print('x1:', x.size())
        x = self.classifier1(x) 
        print('x2:', x.size())
        x = self.classifier2(x) 
        print('x3:', x.size())
        return x

!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project='what-i-learned-on-zero-to-gans-certification-course')

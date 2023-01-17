%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
mnist_pwd = "data"

batch_size= 256
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])



trainset = MNIST(mnist_pwd, train=True, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)



testset = MNIST(mnist_pwd, train=False, download=True, transform=transform)

testloader = DataLoader(testset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)

        self.fc2 = nn.Linear(50, 10)



    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
from ignite.engine import create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Loss, Accuracy

from ignite.contrib.handlers import FastaiLRFinder, ProgressBar
device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = nn.NLLLoss()

model = Net()

model.to(device)  # Move model before creating optimizer

optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})



lr_finder = FastaiLRFinder()

to_save={'model': model, 'optimizer': optimizer}

with lr_finder.attach(trainer, to_save, diverge_th=1.5) as trainer_with_lr_finder:

    trainer_with_lr_finder.run(trainloader)

    

trainer.run(trainloader, max_epochs=10)



evaluator = create_supervised_evaluator(model, metrics={"acc": Accuracy(), "loss": Loss(nn.NLLLoss())}, device=device)

evaluator.run(testloader)



print(evaluator.state.metrics)
lr_finder.plot()
lr_finder.lr_suggestion()
optimizer.param_groups[0]['lr'] = lr_finder.lr_suggestion()



trainer.run(trainloader, max_epochs=10)

evaluator.run(testloader)

print(evaluator.state.metrics)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torch.optim as optim



from torch.utils.data import DataLoader



import torchvision.datasets as datasets  

import torchvision.transforms as transforms
num_classes = 10

learning_rate = 1e-3

batch_size = 1024

num_epochs = 5
model = torchvision.models.vgg16(pretrained=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To use the GPU if available



model.to(device)
class Identity(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, x):

        return x
for prm in model.parameters():

    prm.requires_grad = False
model.avgpool = Identity()



model.classifier = nn.Sequential(

    nn.Linear(512, 100), 

    nn.ReLU(), 

    nn.Linear(100, num_classes)

)





model.to(device)
train_dataset = datasets.CIFAR10(root="/kaggle/working/dataset/", train=True, transform=transforms.ToTensor(), download=True)



train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):

    losses = []



    for batch_idx, (data, targets) in enumerate(train_loader):

        

        # If GPU is active, alter the data accordingly

        data = data.to(device=device)

        targets = targets.to(device=device)



        # forward

        scores = model(data)

        loss = criterion(scores, targets)



        losses.append(loss.item())

        

        # backward

        optimizer.zero_grad()

        loss.backward()



        # gradient descent or adam step

        optimizer.step()



    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")
def check_accuracy(loader, model):

    

    if loader.dataset.train:

        print("Checking accuracy on training data")

    else:

        print("Checking accuracy on test data")



    num_correct = 0

    num_samples = 0

    model.eval()



    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device)

            y = y.to(device=device)



            scores = model(x)

            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum()

            num_samples += predictions.size(0)



        print(

            f"Got {num_correct} / {num_samples} cases correctly with accuracy {float(num_correct)/float(num_samples)*100:.2f}"

        )



    model.train()





check_accuracy(train_loader, model)
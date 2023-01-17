import torch

import torchvision

from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

%matplotlib inline

import torchvision.transforms as transforms
dataset=MNIST(root='data/', download=True, transform=transforms.ToTensor())
test_dataset = MNIST(root='data/', train=False)

from torch.utils.data import random_split



train_ds , val_ds = random_split(dataset,[50000,10000])
from torch.utils.data import DataLoader
batch_size = 128



train_loader = DataLoader(train_ds, batch_size, shuffle=True)

val_loader = DataLoader(val_ds, batch_size)
import torch.nn as nn
input_size=28*28



num_classes=10



mode=nn.Linear(input_size,num_classes)
for images, labels in train_loader:

    print(labels)

    print(images.shape)

    outputs = model(images)

    break
class MnistModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(input_size, num_classes)

        

    def forward(self, xb):

        xb = xb.reshape(-1, 784)

        out = self.linear(xb)

        return out

    

model = MnistModel()
print(model.linear.weight.shape, model.linear.bias.shape)

list(model.parameters())
print(model.linear.weight.shape, model.linear.bias.shape)

list(model.parameters())
for images, labels in train_loader:

    outputs = model(images)

    break



print('outputs.shape : ', outputs.shape)

print('Sample outputs :\n', outputs[:2].data)
import torch.nn.functional as F
# Apply softmax for each output row

probs = F.softmax(outputs, dim=1)



# Look at sample probabilities

print("Sample probabilities:\n", probs[:2].data)



# Add up the probabilities of an output row

print("Sum: ", torch.sum(probs[0]).item())
max_probs, preds = torch.max(probs, dim=1)

print(preds)

print(max_probs)
def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
accuracy(outputs, labels)
loss_fn = F.cross_entropy

# Loss for current batch of data

loss = loss_fn(outputs, labels)

print(loss)
class MnistModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(input_size, num_classes)

        

    def forward(self, xb):

        xb = xb.reshape(-1, 784)

        out = self.linear(xb)

        return out

    

    def training_step(self, batch):

        images, labels = batch 

        out = self(images)                  # Generate predictions

        loss = F.cross_entropy(out, labels) # Calculate loss

        return loss

    

    def validation_step(self, batch):

        images, labels = batch 

        out = self(images)                    # Generate predictions

        loss = F.cross_entropy(out, labels)   # Calculate loss

        acc = accuracy(out, labels)           # Calculate accuracy

        return {'val_loss': loss, 'val_acc': acc}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    

model = MnistModel()
def evaluate(model, val_loader):

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    history = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Training Phase 

        for batch in train_loader:

            loss = model.training_step(batch)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # Validation phase

        result = evaluate(model, val_loader)

        model.epoch_end(epoch, result)

        history.append(result)

    return history
result0 = evaluate(model, val_loader)

result0
history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)
history3 = fit(5, 0.001, model, train_loader, val_loader)
history4 = fit(50, 0.001, model, train_loader, val_loader)
# Replace these values with your results

history = [result0] + history1 + history2 + history3 + history4

accuracies = [result['val_acc'] for result in history]

plt.plot(accuracies, '-x')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Accuracy vs. No. of epochs');
# Define test dataset

test_dataset = MNIST(root='data/', 

                     train=False,

                     transform=transforms.ToTensor())
def predict_image(img, model):

    xb = img.unsqueeze(0)

    yb = model(xb)

    _, preds  = torch.max(yb, dim=1)

    return preds[0].item()
img, label = test_dataset[0]

plt.imshow(img[0], cmap='gray')

print('Label:', label, ', Predicted:', predict_image(img, model))
test_loader = DataLoader(test_dataset, batch_size=256)

result = evaluate(model, test_loader)

result
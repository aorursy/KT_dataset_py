import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
# The dataset has 5 classes of flowers
# Looking at the classes
daisy = (Image.open('../input/flowers-recognition/flowers/daisy/100080576_f52e8ee070_n.jpg'), 'daisy')
dandelion = (Image.open('../input/flowers-recognition/flowers/dandelion/10043234166_e6dd915111_n.jpg'), 'dandelion')
rose = (Image.open('../input/flowers-recognition/flowers/rose/10090824183_d02c613f10_m.jpg'), 'rose')
sunflower = (Image.open('../input/flowers-recognition/flowers/sunflower/1008566138_6927679c8a.jpg'), 'sunflower')
tulip = (Image.open('../input/flowers-recognition/flowers/tulip/100930342_92e8746431_n.jpg'), 'tulip')
             
flowers = [daisy, dandelion, rose, sunflower, tulip]

fig, ax = plt.subplots(1, 5, figsize=(15, 3))
i = 0
for flower, name in flowers:
    ax[i].imshow(flower)
    ax[i].set_title(name)
    ax[i].axis('off')
    i += 1
# Loading data

transform = transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

dataset = datasets.ImageFolder('/kaggle/input/flowers-recognition/flowers/flowers', transform=transform)


len_train_set = int(0.8*len(dataset))
len_test_set = len(dataset) - len_train_set

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_test_set])

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=4)
# Importing and adjusting model
pretrained_net = models.vgg11(pretrained=True)

# Freezing the layers on the pretrained model
for param in pretrained_net.parameters():
    param.requires_grad = False

# Let's take a look at the network architecture
print(pretrained_net)
# Since we have 5 classes of flowers, we need to change the output layer
pretrained_net.classifier[6] = nn.Linear(4096, 5)
# Device adjustments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_net = pretrained_net.to(device)
def train(model, epochs, lr=0.001, print_during_epoch=True):
    # Optimizer and loss function

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training model
    data_size = len(train_dataset)
    losses = []
    
    for epoch in range(epochs):
        examples_processed = 0
        current = 0
        correct_pred = 0

        for batch, target in train_dataloader:
            X, y = batch.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss)
            
            _ ,pred_labels = torch.max(out, axis=1)
            correct_pred += (pred_labels == y).sum().item()

            current += 1
            examples_processed += len(batch)
            if not (current % 10) and print_during_epoch:
                print(f"Epoch {epoch+1}: percentage: {100*examples_processed/data_size:.2f}% | loss: {loss:.2f} | accuracy: {100*correct_pred/examples_processed:.2f}%")

        print(f"Epoch {epoch+1} | Percentage: 100%", end='')
        print(f" | Accuracy: {100*correct_pred/examples_processed:.2f}%")
    
    return model, losses
# Fine tuning the pretrained model
fine_tuned_net, fine_tuned_losses = train(pretrained_net, lr=1e-3, epochs=3)
def plot_losses(losses):
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.show()
plot_losses(fine_tuned_losses)
def evaluate_test_accuracy(model):
    test_dataset_size = len(test_dataset)
    correct_pred = 0
    for batch, target in test_dataloader:
        X, y = batch.to(device), target.to(device)
        out = model(X)

        _, pred_labels = torch.max(out, axis=1)
        correct_pred += (pred_labels == y).sum().item()

    print(f"Test accuracy: {100*correct_pred/test_dataset_size:.2f}%") 
# Accuracy on test set
evaluate_test_accuracy(fine_tuned_net)
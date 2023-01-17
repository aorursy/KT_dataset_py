import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# Reading the training data
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df.head()
# Splitting data and converting to tensor

from sklearn.model_selection import train_test_split
y = df.pop('label') # Getting target
X = df # Renaming data

# Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_test = scaler.fit_transform(X_test.values)

# Transforming to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)


# Setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Creating the neural network architecture

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*5*5, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(-1, 128*5*5)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
        
def train(net, X, y, epochs, batch_size, print_during_epoch=True):
    
    data_size = len(X)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)
    losses = []
    
    for epoch in range(epochs):
        processed = 0
        correct_pred = 0
        count = 0
    
        while processed < data_size:
            end = min(processed + batch_size, data_size-1) # Final index of the batch
            batch = X[processed:end].to(device).view(-1, 1, 28, 28)
            target = y[processed:end].to(device)
        
            optimizer.zero_grad() # We need to zero the accumulated gradients
            out = net(batch) # Forward pass
            loss = criterion(out, target)
            loss.backward() # Backpropagation
            optimizer.step()
            
            losses.append(loss)
            
            processed += batch_size
            
            # Keeping track of the correct predictions
            _, pred = torch.max(out, axis=1)
            correct_pred += (pred == target).sum().item()
        
            if not (count % 100) and print_during_epoch:
                print(f"Epoch {epoch+1} | Percentage: {100*processed/data_size:.2f}%", end='')
                print(f" | Accuracy: {100*correct_pred/processed:.2f}%")
            count += 1
        
        print(f"Epoch {epoch+1} | Percentage: 100%", end='')
        print(f" | Accuracy: {100*correct_pred/processed:.2f}%")
        
    return net, losses
def plot_losses(losses):
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.show()
net, losses = train(NN(), X_train, y_train, 10, 512, print_during_epoch=False)
plt.style.use('ggplot')
plot_losses(losses)
# Getting accuracy on test set

test_out = net(X_test.view(-1, 1, 28, 28).to(device))
_, pred = torch.max(test_out, axis=1)
correct_pred = (pred == y_test.to(device)).sum()
print(f"Test accuracy: {100*correct_pred/len(X_test):.2f}%")
# Training the network with the entire dataset

X_total = torch.cat((X_train, X_test), 0)
y_total = torch.cat((y_train, y_test), 0)
final_net, final_losses = train(NN(), X_total, y_total, 10, 128, print_during_epoch=False)
plot_losses(final_losses)
# Generating submission file
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_data_scaled = scaler.fit_transform(test_data.values)
test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float).view(-1, 1, 28, 28).to(device)
_, pred = torch.max(final_net(test_data_tensor), axis=1)

pred_dict = {'ImageId': range(1, len(pred)+1), 'Label': pred.cpu()}
submission_df = pd.DataFrame(pred_dict)
print(submission_df.head())
submission_df.to_csv('submission.csv', index=False)
# Let's take a look at some predictions
n = 4
fig, ax = plt.subplots(n, n, figsize=(10, 10))
plt.axis('off')
for i in range(n*n):
    image_array = test_data.iloc[i].values.reshape(28, 28)
    image_pred = pred[i]
    i_ax = ax[i//n][i%n]
    i_ax.imshow(image_array)
    i_ax.set_title(f"Predicted class: {image_pred}")
    i_ax.axis('off')
    
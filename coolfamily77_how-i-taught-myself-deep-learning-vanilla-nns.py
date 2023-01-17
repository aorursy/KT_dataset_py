# Import torch
import torch

# Create some tensors
x = torch.empty(5, 3)
print(x, '\n')

y = torch.ones(5, 3, dtype=torch.long)
print(y, '\n')

z = torch.tensor([[0, 1, 2], [3, 4, 5]])
print(z)
# Imports
import torch
import torch.nn as nn                               # to access build-in functions to build the NN
import torch.nn.functional as F                     # to access activation functions
from torchvision import datasets, transforms        # to access the MNIST dataset
import torch.optim as optim                         # to build out optimizer

import numpy as np
import matplotlib.pyplot as plt # for plotting
%matplotlib inline
import seaborn as sns
import sklearn.metrics
# Load in the data from torchvision datasets 
# train=True to access training images and train=False to access test images
# We also transform to Tensors the images
mnist_train = datasets.MNIST('data', train = True, download = True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST('data', train = False, download = True, transform=transforms.ToTensor())
# How the object looks:
print('Structure of train data:', mnist_train, '\n')
print('Structure of test data:', mnist_test, '\n')
print('Image on index 0 shape:', list(mnist_train)[0][0].shape)
print('Image on index 0 label:', list(mnist_train)[0][1])
# Check a sample of the images
# We need to import again, because our train and test data are Tensors already
sample = datasets.MNIST('data', train=True, download=True)

plt.figure(figsize = (16, 3))
for k, (image, label) in enumerate(sample):
    if k >= 16:
        break
    plt.subplot(2, 8, k+1)
    plt.imshow(image)
# Creating the Network:
class MNISTClassifier(nn.Module):                           # nn.Module is a subclass from which we inherit
    def __init__(self):                                     # Here you define the structure
        super(MNISTClassifier, self).__init__()             
        self.layers = nn.Sequential(nn.Linear(28*28, 50),   # Create first layer: from 784 neurons to 50
                                    nn.ReLU(),              # Call activation function
                                    nn.Linear(50, 20),      # Second layer: from 50 neurons to 20
                                    nn.ReLU(),              # Call Activation function
                                    nn.Linear(20, 10))      # Last layer: from 20 neurons to 10
        # 10 because we have 10 categories of numbers from which we need to pick 1
        # If we would have wanted to classify images labeled "dog", "cat", "crocodile",
           # the final layer would have had 3 neurons.
        
    def forward(self, image, prints=False):                 # Function where you take the image though the FNN
        if prints: print('Image shape:', image.shape)
        image = image.view(-1, 28*28)                       # Flatten image: from [1, 28, 28] to [784]
        if prints: print('Image reshaped:', image.shape)
        out = self.layers(image)                            # Create Log Probabilities
        if prints: print('Out shape:', out.shape)
        
        return out
torch.manual_seed(1) # set the random seed
np.random.seed(1) #set random seed in numpy

# Selecting 1 image with its label
image_example, label_example = mnist_train[0]
print('Image shape:', image_example.shape)
print('Label:', label_example, '\n')

# Creating an instance of the model
model_example = MNISTClassifier()
print(model_example, '\n')

# Creating the log probabilities
out = model_example(image_example, prints=True)
print('out:', out, '\n')

# Choose maximum probability and then select only the label (not the prob number)
prediction = out.max(dim=1)[1]
print('prediction:', prediction)
# Creating LOSS and Optimizer instances

# Loss is the function that calculates how far is the prediction from the true value
criterion = nn.CrossEntropyLoss()
print('Criterion:', criterion, '\n')

# Using this loss the Optimizer computes the gradients of each neuron and updates the weights
optimizer = optim.SGD(model_example.parameters(), lr=0.005, momentum=0.9)
print('Optimizer:', optimizer)
# Let's also look at how many parameters (weights and biases) are updating during 1 single backpropagation
# Parameter Understanding
for i in range(6):
    print(i+1, ':', list(model_example.parameters())[i].shape)
torch.manual_seed(1) # set the random seed
np.random.seed(1) #set random seed in numpy

print('Log Probabilities:', out)
print('Actual value:', torch.tensor(label_example).reshape(-1))

# Clear gradients - always needs to be called before backpropagation
optimizer.zero_grad()
# Compute loss
loss = criterion(out, torch.tensor(label_example).reshape(-1))
print('Loss:', loss)
# Compute Gradients
loss.backward()
# Update weights
optimizer.step()

# After this 1 iteration the weights have updated once
# Create trainloaders for train and test data
# We put shuffle=True so the images shuffle after every epoch
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=60, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10, shuffle=True)

# Inspect Trainloader
print('trainloader object:', train_loader, '\n')

# Select First Batch
imgs, labels = next(iter(train_loader))

print('Object shape:', imgs.shape)      # [60, 1, 28, 28]: 60 images of size [1, 28, 28]
print('Label values:', labels)          # actual labels for the 60 images
print('Total Inages:', labels.shape)    # 60 labels in total
torch.manual_seed(1) # set the random seed
np.random.seed(1) #set random seed in numpy

n = 0
for k, (images, labels) in enumerate(train_loader):
    # Stop after 3 iterations
    if k >= 3:
        break
    
    print('========== Batch', k, ':')
    # Prediction:
    out = model_example(images)
    print('out shape:', out.shape)
    
    # Update weights (or parameters):
    loss = criterion(out, labels)
    print('loss:', loss)
    
    print('Optimizing...')
    # Computes the gradient of current tensor
    loss.backward()
    # Performs a single optimization step.
    optimizer.step()
    # Clears the gradients of all optimized
    optimizer.zero_grad()
    print('Done.')    
    
    if k<2: print('\n')
# Instantiate 2 variables for total cases and correct cases
correct_cases = 0
total_cases = 0

# Sets the module in evaluation mode (VERY IMPORTANT)
model_example.eval()

for k, (images, labels) in enumerate(train_loader):
    # Just show first 3 batches accuracy
    if k >= 3: break
    
    print('==========', k, ':')
    out = model_example(images)
    print('Out:', out.shape)
    
    # Choose maximum probability and then select only the label (not the prob number)
    prediction = out.max(dim = 1)[1]
    print('Prediction:', prediction.shape)
    
    # Number of correct cases - we first see how many are correct in the batch
            # then we sum, then convert to integer (not tensor)
    correct_cases += (prediction == labels).sum().item()
    print('Correct:', correct_cases)
    
    # Total cases
    total_cases += images.shape[0]
    print('Total:', total_cases)
    
    
    if k < 2: print('\n')
        

print('Average Accuracy after 3 iterations:', correct_cases/total_cases)
def get_accuracy(model, data, batchSize = 20):
    '''Iterates through data and returnes average accuracy per batch.'''
    # Sets the model in evaluation mode
    model.eval()
    
    # Creates the dataloader
    data_loader = torch.utils.data.DataLoader(data, batch_size=batchSize)
    
    correct_cases = 0
    total_cases = 0
    
    for (images, labels) in iter(data_loader):
        # Is formed by 20 images (by default) with 10 probabilities each
        out = model(images)
        # Choose maximum probability and then select only the label (not the prob number)
        prediction = out.max(dim = 1)[1]
        # First check how many are correct in the batch, then we sum then convert to integer (not tensor)
        correct_cases += (prediction == labels).sum().item()
        # Total cases
        total_cases += images.shape[0]
    
    return correct_cases / total_cases
def train_network(model, train_data, test_data, batchSize=20, num_epochs=1, learning_rate=0.01, weight_decay=0,
                 show_plot = True, show_acc = True):
    
    '''Trains the model and computes the average accuracy for train and test data.
    If enabled, it also shows the loss and accuracy over the iterations.'''
    
    print('Get data ready...')
    # Create dataloader for training dataset - so we can train on multiple batches
    # Shuffle after every epoch
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True)
    
    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    # Losses & Iterations: to keep all losses during training (for plotting)
    losses = []
    iterations = []
    # Train and test accuracies: to keep their values also (for plotting)
    train_acc = []
    test_acc = []
    
    print('Training started...')
    iteration = 0
    # Train the data multiple times
    for epoch in range(num_epochs):
        
        for images, labels in iter(train_loader):
            # Set model in training mode:
            model.train()
            
            # Create log probabilities
            out = model(images)
            # Clears the gradients from previous iteration
            optimizer.zero_grad()
            # Computes loss: how far is the prediction from the actual?
            loss = criterion(out, labels)
            # Computes gradients for neurons
            loss.backward()
            # Updates the weights
            optimizer.step()
            
            # Save information after this iteration
            iterations.append(iteration)
            iteration += 1
            losses.append(loss)
            # Compute accuracy after this epoch and save
            train_acc.append(get_accuracy(model, train_data))
            test_acc.append(get_accuracy(model, test_data))
            
    
    # Show Accuracies
    # Show the last accuracy registered
    if show_acc:
        print("Final Training Accuracy: {}".format(train_acc[-1]))
        print("Final Testing Accuracy: {}".format(test_acc[-1]))
    
    # Create plots
    if show_plot:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title("Loss Curve")
        plt.plot(iterations[::20], losses[::20], label="Train", linewidth=4, color='#008C76FF')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.subplot(1,2,2)
        plt.title("Accuracy Curve")
        plt.plot(iterations[::20], train_acc[::20], label="Train", linewidth=4, color='#9ED9CCFF')
        plt.plot(iterations[::20], test_acc[::20], label="Test", linewidth=4, color='#FAA094FF')
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.show()
# Select images
mnist_data = datasets.MNIST('data', train = True, download=True, transform=transforms.ToTensor())
mnist_data = list(mnist_data)

# Training and Testing selection
mnist_train = mnist_data[:500]     # 500 training images
mnist_test = mnist_data[500:1000]  # 500 test images

# Create Model Instance
model1 = MNISTClassifier()

# Train...
train_network(model1, mnist_train, mnist_test, num_epochs=200)
# Predefined Function that shows 20 images
def show20(data, title='Default'):
    plt.figure(figsize=(10,2))
    for n, (image, label) in enumerate(data):
        if n >= 20:
            break
        plt.subplot(2, 10, n+1)
        plt.imshow(image)
        plt.suptitle(title, fontsize=15);
        

# Create original and rotated set
original_images = datasets.MNIST('data', train=True, download=True)
rotated_images = datasets.MNIST('data', train=True, download=True, 
                                transform=transforms.RandomRotation(25, fill=(0,)))

#Show images
show20(original_images, 'Original')
show20(rotated_images, 'Rotated')
# Creating a personalized transform
# First Rotates, then transforms to tensor, then normalizes the images
mytransform = transforms.Compose([transforms.RandomRotation(25, fill=(0,)),
                                  transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

# Import the MNIST data aplying the transformations
mnist_data_aug = datasets.MNIST('data', train = True, download=True, transform=mytransform)
mnist_data_aug = list(mnist_data_aug)

# We select first 500 images as our training
mnist_train_aug = mnist_data_aug[:500]

# ------ Training the model ------
# Create Model Instance
model_aug = MNISTClassifier()

# Train...
train_network(model_aug, mnist_train_aug, mnist_test, num_epochs=200)
# Create Model Instance
model2 = MNISTClassifier()

# Train...
train_network(model2, mnist_train, mnist_test, num_epochs=200, learning_rate=0.001, weight_decay=0.0005)
class MNISTClassifier_improved(nn.Module):
    def __init__(self, layer1_size=50, layer2_size=20, dropout=0.4):       # Structure of the FNN 
        super(MNISTClassifier_improved, self).__init__()
        
        self.layers = nn.Sequential(nn.Dropout(p = dropout),               # Dropout for first layer
                                    nn.Linear(28*28, layer1_size),         # From 784 neurons to layer1_size
                                    nn.ReLU(),                             # Activation Function
                                    nn.Dropout(p = dropout),               # Dropout for second layer
                                    nn.Linear(layer1_size, layer2_size),   # From layer1_size neurons to layer2_size
                                    nn.ReLU(),                             # Activation Function
                                    nn.Dropout(p = dropout),               # Dropout for last layer
                                    nn.Linear(layer2_size, 10))            # Output layer
        
    def forward(self, image):                # Taking the image through the NN
        image = image.view(-1, 28*28)        # Flatten the matrix to a vector
        out = self.layers(image)             # Log Probabilities output
        
        return out
# Training on the newly network:
# Create Model Instance
model_improved = MNISTClassifier_improved(layer1_size=80, layer2_size=50, dropout=0.5)
print(model_improved)

# Train...
train_network(model_improved, mnist_train, mnist_test, num_epochs=200)
def get_confusion_matrix(model, test_data):
    # First we make sure we disable Gradient Computing
    torch.no_grad()
    
    # Model in Evaluation Mode
    model.eval()
    
    preds, actuals = [], []

    for image, label in mnist_test:
        # Add 1 more dimension for batching
        image = image.unsqueeze(0)
        out = model_improved(image)

        prediction = torch.max(out, dim=1)[1].item()
        preds.append(prediction)
        actuals.append(label)
    
    return sklearn.metrics.confusion_matrix(preds, actuals)
plt.figure(figsize=(16, 5))
sns.heatmap(get_confusion_matrix(model_improved, mnist_test), cmap='icefire', annot=True, linewidths=0.1)
plt.title('Confusion Matrix', fontsize=15);
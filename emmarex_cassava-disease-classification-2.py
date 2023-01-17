%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from PIL import Image

from os import listdir

import pandas as pd
data_directory = '../input/train/train'

epochs = 100
train_transforms = transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.Resize(500),

        transforms.CenterCrop(500),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

train_data = datasets.ImageFolder(data_directory,transform=train_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_transforms = transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.Resize(500),

        transforms.CenterCrop(500),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

test_data = datasets.ImageFolder(data_directory,transform=test_transforms,)

testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
class_names = trainloader.dataset.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model = models.resnet50(pretrained=True)

# model = models.densenet161(pretrained=True)

model.cuda()

# print(model)
for param in model.parameters():

    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(512, 5),

                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

# criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.fc.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-4)

model.to(device)
epoch_list, train_accuracy_list, test_accuracy_list, train_loss_list, validation_loss_list = [], [], [], [], []

for epoch in range(epochs):

    train_loss = 0

    val_loss = 0

    accuracy = 0

    # Training the model

    model.train()

    counter = 0

    for inputs, labels in trainloader:

        # Move to device

        inputs, labels = inputs.to(device), labels.to(device)

        # Clear optimizers

        optimizer.zero_grad()

        # Forward pass

        output = model.forward(inputs)

        # Loss

        loss = criterion(output, labels)

        # Calculate gradients (backpropogation)

        loss.backward()

        # Adjust parameters based on gradients

        optimizer.step()

        # Add the loss to the training set's rnning loss

        train_loss += loss.item()*inputs.size(0)

        # Print the progress of our training

        counter += 1

#         print(counter, "/", len(trainloader))

    # Evaluating the model

    model.eval()

    counter = 0

    # Tell torch not to calculate gradients

    with torch.no_grad():

        for inputs, labels in testloader:

            # Move to device

            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass

            output = model.forward(inputs)

            # Calculate Loss

            valloss = criterion(output, labels)

            # Add loss to the validation set's running loss

            val_loss += valloss.item()*inputs.size(0)

            # Since our model outputs a LogSoftmax, find the real 

            # percentages by reversing the log function

            output = torch.exp(output)

            # Get the top class of the output

            top_p, top_class = output.topk(1, dim=1)

            # See how many of the classes were correct?

            equals = top_class == labels.view(*top_class.shape)

            # Calculate the mean (get the accuracy for this batch)

            # and add it to the running accuracy for this epoch

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Print the progress of our evaluation

            counter += 1

#             print(counter, "/", len(testloader))

    # Get the average loss for the entire epoch

    train_loss = train_loss/len(trainloader.dataset)

    train_loss_list.append(train_loss)

    valid_loss = val_loss/len(testloader.dataset)

    validation_loss_list.append(valid_loss)

    train_acc = accuracy/len(trainloader)

    train_accuracy_list.append(train_acc)

    test_acc = accuracy/len(testloader)

    test_accuracy_list.append(test_acc)

    epoch_list.append(epoch)

#     print(f'Epoch: {epoch}, Accuracy: {acc}, Train loss: {train_loss}, Valid loss: {valid_loss}')
max_train_accuracy = max(train_accuracy_list)

max_test_accuracy = max(test_accuracy_list)

epoch_train = epoch_list[train_accuracy_list.index(max_train_accuracy)]

epoch_test = epoch_list[test_accuracy_list.index(max_test_accuracy)]

print(f'Max Accuracy Score: {max_train_accuracy}, Epoch: {epoch_train}')

print(f'Max Accuracy Score: {max_test_accuracy}, Epoch: {epoch_test}')
epochs = range(1, len(train_accuracy_list) + 1)

#Train and validation accuracy

plt.plot(epochs, train_accuracy_list, 'b', label='Train accurarcy')

plt.plot(epochs, test_accuracy_list, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()

#Train and validation loss

plt.plot(epochs, train_loss_list, 'b', label='Training loss')

plt.plot(epochs, validation_loss_list, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
model.eval()
def process_image(image_dir):

    # Process a PIL image for use in a PyTorch model

    # tensor.numpy().transpose(1, 2, 0)

    image = Image.open(image_dir)

    preprocess = transforms.Compose([

        transforms.Resize(500),

        transforms.CenterCrop(450),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    image = preprocess(image)

    # Convert 2D image to 1D vector

    image = np.expand_dims(image, 0)

    image = torch.from_numpy(image)

    inputs = image.to(device)

    return inputs
# Using our model to predict the label

def predict(image, model):

    # Pass the image through our model

    output = model.forward(image)

    # Reverse the log function in our output

    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for

    # that class

    probs, classes = output.topk(1, dim=1)

    return probs.item(), classes.item()
test_directory = '../input/test/test/0'

predictions, test_image_fileName = [], []

try:

    test_images = listdir(test_directory)

    for images in test_images:

        test_image_fileName.append(images)

        image = process_image(f'{test_directory}/{images}')

        top_prob, top_class = predict(image, model)

        predictions.append(class_names[top_class])

except Exception as e:

    print(e)
print("[INFO] Creating pandas dataframe")

submission_data = {"Category":predictions,"Id":test_image_fileName,}

submission_data_frame = pd.DataFrame(submission_data)
submission_data_frame.head()
print("[INFO] Saving Predicition to CSV")

submission_data_frame.to_csv('sample-submission.csv',columns=["Category","Id"], index = False)
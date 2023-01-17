# Other Imports

import json

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import plotly.graph_objs as go

import plotly.offline as py

import seaborn as sns



# Pytorch Imports

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data

import torchvision

from torchvision import transforms

import torchvision.models as models



print(os.listdir('../input'))

%matplotlib inline

py.init_notebook_mode(connected=False)

sns.set(rc={'figure.figsize':(20,10)})



%env JOBLIB_TEMP_FOLDER=/tmp
CAT_TO_NAME_PATH = '../input/hackathon-blossom-flower-classification/cat_to_name.json'

TRAIN_DATA_PATH = "../input/hackathon-blossom-flower-classification/flower_data/flower_data/train"

VAL_DATA_PATH = "../input/hackathon-blossom-flower-classification/flower_data/flower_data/valid"

TEST_DATA_PATH = '../input/hackathon-blossom-flower-classification/test set/'

CHECKPOINT_PATH = '../input/model-checkpoints/'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_cat_to_name_data(file_path):

    """ Imports the cat_to_name.json file and returns a Pandas DataFrame """

    with open(file_path, 'r') as f:

        cat_to_names = json.load(f)

    return cat_to_names
cat_to_names = get_cat_to_name_data(CAT_TO_NAME_PATH)
cat_to_names
for i in cat_to_names:

    if i == '11':

        print(cat_to_names['11'])
def get_data_loaders(train_data_path, val_data_path):

    transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(256),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])



    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)

    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=4)

    val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transform)

    val_loader  = data.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=4) 

    

    train_class_names = train_data.classes

    val_class_names = val_data.classes

    

    return train_loader, val_loader, train_class_names, val_class_names
train_loader, val_loader, train_class_names, val_class_names = get_data_loaders(TRAIN_DATA_PATH, VAL_DATA_PATH)
def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

    



def visualize_images():



    dataiter = iter(train_loader)

    images, labels = dataiter.next()





    imshow(torchvision.utils.make_grid(images[:4]))

    for j in range(4):

        print("label: {}, class: {}, name: {}".format(labels[j].item(),

                                               train_class_names[labels[j].item()],

                                               cat_to_names[train_class_names[labels[j].item()]]))



    

visualize_images()
def create_model():

    model = models.densenet161(pretrained=True)



    for param in model.parameters():

        param.requires_grad = False



    num_filters = model.classifier.in_features

    model.classifier = nn.Sequential(nn.Linear(num_filters, 2048),

                               nn.ReLU(),

                               nn.Linear(2048, 512),

                               nn.ReLU(),

                               nn.Linear(512, 102),

                               nn.LogSoftmax(dim=1))



    # Move model to the device specified above

    model.to(device)

    return model
model = create_model()
criterion = nn.NLLLoss()

# Set the optimizer function using torch.optim as optim library

optimizer = optim.Adam(model.parameters())
def train(epochs):

    train_losses = []

    valid_losses = []

    

    for epoch in range(epochs):

        train_loss = 0

        val_loss = 0

        accuracy = 0



        # Training the model

        model.train()

        counter = 0

        for inputs, labels in train_loader:

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

            train_loss += loss.item()



            # Print the progress of our training

            counter += 1

            #print(counter, "/", len(train_loader))



            # Evaluating the model

        model.eval()

        counter = 0

        # Tell torch not to calculate gradients

        with torch.no_grad():

            for inputs, labels in val_loader:

                # Move to device

                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass

                output = model.forward(inputs)

                # Calculate Loss

                valloss = criterion(output, labels)

                # Add loss to the validation set's running loss

                val_loss += valloss.item()



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

                #print(counter, "/", len(val_loader))



        # Get the average loss for the entire epoch

        train_loss = train_loss/len(train_loader)

        valid_loss = val_loss/len(val_loader)



        train_losses.append(train_loss)

        valid_losses.append(valid_loss)

        # Print out the information

        print('Accuracy: ', accuracy/len(val_loader))

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))



    return train_losses, valid_losses

def save_checkpoint():

    checkpoint = {'model': model,

                  'state_dict': model.state_dict(),

                  'optimizer' : optimizer.state_dict()}



    torch.save(checkpoint, 'checkpoint1.pth')
def load_checkpoint(filepath, inference=False):

    checkpoint = torch.load(filepath + 'checkpoint1.pth')

    model = checkpoint['model']

    model.load_state_dict(checkpoint['state_dict'])

    

    if inference:

      for parameter in model.parameters():

          parameter.requires_grad = False



      model.eval()

    

    model.to(device)

    return model



model = load_checkpoint(filepath=CHECKPOINT_PATH)
def get_test_dataloaders(test_path):

    transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(256),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])



    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    test_loader  = data.DataLoader(test_data, batch_size=32, shuffle=False) 

    

    return test_loader
test_loader = get_test_dataloaders(TEST_DATA_PATH)
def predict(test_loader):

    model.eval()

    

    predictions = []

    with torch.no_grad():

        for images, _ in test_loader:

            images = images.to(device)

            output = model(images)

            ps = torch.exp(output)

            top_p, top_class = ps.topk(1, dim=1)

            predictions += [int(i) for i in list(top_class.data.cpu().numpy())]

        

    return predictions
predictions = predict(test_loader)
predictions
image_names = [image_name for image_name in os.listdir(TEST_DATA_PATH + 'test set')]
def create_pred_dataframe(image_names, predictions):

    predicted_labels = [cat_to_names[train_class_names[i]] for i in predictions]

    pred_df_with_species_name = pd.DataFrame({'image-names': image_names, 'species': predicted_labels})

    

    print('cat: {} for : train_class_name{} for pred: {}'.format([cat_to_names[train_class_names[i]] for i in predictions[:2]],

                                                [train_class_names[i] for i in predictions[:2]],

                                                                predictions[:2]))

    pred_df_with_cat_number = pd.DataFrame({'image-names': image_names, 'category': [train_class_names[i] for i in predictions]})

    return pred_df_with_species_name, pred_df_with_cat_number
pred_df_with_species_name, pred_df_with_cat_number = create_pred_dataframe(image_names, predictions)
pred_df_with_species_name = pred_df_with_species_name.sort_values(by=['image-names'])

pred_df_with_cat_number = pred_df_with_cat_number.sort_values(by=['image-names'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(pred_df_with_cat_number)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(pred_df_with_species_name)
pred_df_with_species_name.to_csv('my_predictions.csv', sep='\t', encoding='utf-8')
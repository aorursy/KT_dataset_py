import torch

import torchvision

import torchvision.transforms as transforms

import numpy as np
if torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')
device
#Loading Data

data_path = '../input/plant-seedlings-classification/train/'

batch_size=128

resized_width=224

resized_height=224

classes = ('Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent',

'Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet')



def load_split_train_test(datadir, valid_size = .2):

    train_transforms = transforms.Compose([

        transforms.Resize((resized_width,resized_height)),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))               

        ])

    test_transforms = transforms.Compose([

        transforms.Resize((resized_width,resized_height)),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))               

        ])

    train_data = torchvision.datasets.ImageFolder(

        root=datadir,  

        transform=train_transforms

    )

    test_data = torchvision.datasets.ImageFolder(

        root=datadir,

        transform=test_transforms

    )

    num_train = len(train_data)

    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(

            train_data,

            batch_size=batch_size,

            num_workers=4,

#             shuffle=True,                                # Mutually exclusive with sampler

            sampler=train_sampler

    )

    testloader = torch.utils.data.DataLoader(

            test_data,

            batch_size=batch_size,

            num_workers=4,

#             shuffle=True,                               # Mutually exclusive with sampler

            sampler=test_sampler

    )

    return trainloader, testloader



trainloader, testloader = load_split_train_test(datadir=data_path, valid_size = .2)
#Visualize images

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



import numpy as np

plt.rcParams['figure.dpi'] = 100



# functions to show an image

def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

    print(img.size())



# get some random training images

dataiter = iter(trainloader)

images, labels = dataiter.next()



# show images

imshow(torchvision.utils.make_grid(images))

# print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



# get some random validation images

testdataiter = iter(testloader)

testimages, testlabels = testdataiter.next()



# show images

imshow(torchvision.utils.make_grid(testimages))

# print labels

print(' '.join('%5s' % classes[testlabels[j]] for j in range(batch_size)))
# Define the NN

import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,kernel_size=5, stride=1,padding=(2,2))

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5, stride=1,padding=(2,2))

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=5, stride=1,padding=(2,2))

        self.fc1 = nn.Linear(32 * 28 * 28, 240)

        self.fc2 = nn.Linear(240, 120)

        self.fc3 = nn.Linear(120, 84)

        self.fc4 = nn.Linear(84, 12)

    def forward(self, x):

#         print(x.shape)

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.num_flat_features(x))

#         print(x.view(-1, self.num_flat_features(x)).shape)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        X = self.fc4(x)

        return x





    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

#         print(x.size())

#         imshow(torchvision.utils.make_grid(x))

        num_features = 1

        for s in size:

            num_features *= s

        return num_features

    

net = Net().to(device=device)
!pip install torchsummary
from torchsummary import summary

summary(net,(3,224,224))
def save_models(epoch):

    torch.save(net.state_dict(), "seedlings_{}.model".format(epoch))

    print("Chekcpoint saved")
# Define the loss function

import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# Training the NN

epochs = 50

best_accuracy = 0.0

accuracy_epoch = []

loss_epoch = []

test_accuracy_epoch = []

test_loss_epoch = []

for epoch in range(epochs):  # loop over the dataset multiple times

    

    if(epoch % 10 == 0):      # cross val every 15 iteration

        trainloader, testloader = load_split_train_test(datadir=data_path, valid_size = .2)

    # Training the model

    net.train()

    running_loss = 0.0

    correct = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        inputs = inputs.to(device=device)

        labels = labels.to(device=device)

        outputs = net(inputs)

#         print(outputs.shape)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        _, prediction = torch.max(outputs.data, 1)

        correct += torch.sum(prediction == labels.data)

#         print(correct)

        if i % 2000 == 1999:    # print every 2000 mini-batches

            print('[%d, %f\%5d] loss: %.3f' %

                  (epoch + 1,correct, i + 1, running_loss / 2000))

            running_loss = 0.0

    accuracy = 100 * correct / (len(trainloader)* batch_size)

    accuracy_epoch.append(accuracy)

    loss_epoch.append(loss)

    if(accuracy > best_accuracy):

        save_models(epoch)

        best_accuracy = accuracy

    

    # Validating the model

    net.eval()

    running_test_loss = 0.0

    test_correct = 0.0

    with torch.no_grad():

        for j , testdata in enumerate(testloader,0):

            test_inputs,test_labels = testdata

            # zero the parameter gradients

            optimizer.zero_grad()

            

            test_inputs = test_inputs.to(device = device)

            test_labels = test_labels.to(device = device)

            

            test_outputs = net(test_inputs)

            test_loss = criterion(test_outputs,test_labels)

            running_test_loss += test_loss.item()

            _, test_prediction = torch.max(test_outputs.data, 1)

            test_correct += torch.sum(test_prediction == test_labels.data)

        

        test_accuracy = 100 * test_correct / (len(testloader)*batch_size)

        test_accuracy_epoch.append(test_accuracy)

        test_loss_epoch.append(test_loss)

                       

            

    print("Epoch {:d}/{:d}, Train Loss: {:.3f}, Train Accuracy: {:.3f} Validation Loss: {:.3f}, Validation Accuracy : {:.3f}".format(epoch+1,epochs, loss.data, accuracy, test_loss.data,test_accuracy))

print('Finished Training')
plt.plot(accuracy_epoch, label='Training Accuracy')

plt.plot(loss_epoch, label='Training loss')

plt.plot(test_accuracy_epoch, label='Validation Accuracy')

plt.plot(test_loss_epoch, label='Validation loss')

plt.legend(frameon=False)

plt.rcParams['figure.dpi'] = 200

plt.show()
model_path = '../input/plantseedlingsmodels/seedlings_60.model'

def load_saved_model(model_path):

    # Loading the saved models parameters to the model Class

    model = Net()

    state_dict = torch.load(model_path,map_location=torch.device('cpu') )

    model.load_state_dict(state_dict)
from PIL import Image

from torch.autograd import Variable



def predict_image(image_path):

#     print("Prediction in progress")

    image = Image.open(image_path)



    # Define transformations for the image, should (note that imagenet models are trained with image size 224)

    transformation = transforms.Compose([

        transforms.CenterCrop((resized_width,resized_width)),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])



    # Preprocess the image

    image_tensor = transformation(image).float()



    # Add an extra batch dimension since pytorch treats all images as batches

    image_tensor = image_tensor.unsqueeze_(0)



    if torch.cuda.is_available():

        image_tensor.cuda()



    # Turn the input into a Variable

    input = Variable(image_tensor)



    # Predict the class of the image

    output = model(input)



    index = output.data.numpy().argmax()



    return index
import os

import pandas as pd

predict_img_dir = '../input/plant-seedlings-classification/test/'

def run_predictions():

    directory = os.fsencode(predict_img_dir)



    predictions = []

    for file in os.listdir(directory):

        filename = os.fsdecode(file)

        print(filename)

        prediction = predict_image(predict_img_dir+filename)

        pred = {'file':filename, 'species':classes[prediction]}

        predictions.append(pred)

    df = pd.DataFrame(predictions)

    df.to_csv('predictions.csv',index=False)
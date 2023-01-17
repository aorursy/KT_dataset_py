import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from torchvision import transforms    # For Data Augmentation

from sklearn.model_selection import train_test_split # For spliting training set

from matplotlib import pyplot as plt  # For plotting training examples and graphs  

%matplotlib inline
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

Data = train.to_numpy()  # Converting to numpy array

n_train = Data.shape[0]  # No of training examples



# Seperating labels and Pixel values from training data

labels = Data[:, 0]     

pixelVal = Data[:, 1:]



# Reshaping Train and test data

pixelVal = np.reshape(pixelVal, [-1, 1, 28, 28]).astype(np.float32())

print('Training:', pixelVal.shape)
train_data, CV_data, train_labels, CV_labels = train_test_split(pixelVal, labels, test_size = 0.1, random_state = 1)

train_data, train_labels, CV_data, CV_labels = map(torch.tensor, (train_data, train_labels, CV_data, CV_labels))



n_train = train_data.shape[0]

n_val = CV_data.shape[0]

print('Training:', train_data.shape)

print('Cross Validation:', CV_data.shape)
# Display some examples from training and test data

r = 345

print(train_labels[r])

image = train_data[r, 0, :, :]

plt.imshow(image, cmap='gray')

plt.axis('off')

plt.show()



# print(CV_labels[r])

# plt.imshow(CV_data[r, 0, :, :], cmap='gray')

# plt.axis('off')

# plt.show()



transform = transforms.Compose([

    transforms.ToPILImage(),

#     transforms.RandomAffine(degrees = 30, translate = (0.2, 0.2), scale = (0.8, 1.2)),

#     transforms.RandomErasing(p=1,scale=(0.02, 0.05))

    transforms.ToTensor()

])

transform_Image = transform(image)

plt.imshow(transform_Image[0], cmap='gray')

plt.axis('off')

plt.show()
# Checking GPU connection

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print('Device:',device)
# Defining a model

class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        

        # Convolutional Layers



        self.ConvLayers = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Dropout2d(p = 0.4),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Dropout2d(p = 0.5),

#             nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Dropout2d(p = 0.4),

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

#             nn.MaxPool2d(kernel_size=2, stride=2)

        ) 

        

        # Fully Connected Layers

        self.Classifier = nn.Sequential(

            nn.Dropout(p = 0.4),

            nn.Linear(128 * 36, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.4),

            nn.Linear(128, 10)

        )



    def forward(self, x):

        out = self.ConvLayers(x)

        out = out.view(out.size()[0], -1)

        out = self.Classifier(out)

        return out



    

model = ConvNet()  # Creating a model

optimizer = optim.Adam(model.parameters(), lr=0.003)  # Defining an optimizer to optimize the loss

criterion = nn.CrossEntropyLoss()  # Defining the cost/loss

lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, eps = 10e-9)  # Using learning rate decay



if torch.cuda.is_available():

    model = model.cuda()

    criterion = criterion.cuda()
def train_model(epochs):

    

    LOSS = []  # For saving loss at every epoch

    ACC = []   # For saving accuracy at every epoch

    

    ValData = TensorDataset(CV_data/255, CV_labels)

    CVloader = DataLoader(ValData, batch_size = 256)

    

    for epoch in range(epochs):

        model.train()

        trainLoss = 0.0

        trainAcc = 0.0

        

        AugData = torch.zeros([n_train, 1, 28, 28], dtype=torch.float32)

#         for m in range(int(n_train/4)):

#             AugData[m] = transform(train_data[m])

#         TrainData = TensorDataset(AugData/255, train_labels)

        TrainData = TensorDataset(train_data/255, train_labels)

        trainloader = DataLoader(TrainData, batch_size=128, shuffle=True, num_workers=4)



        for data, label in trainloader:

            # Converting for using GPU

            data = data.to(device)

            label = label.to(device)

            

            # Calculating Loss/Cost

            optimizer.zero_grad()

            out = model(data.float())

            loss = criterion(out, label)

            trainLoss += loss

            

            # Calculating Accuracy

            preds = torch.argmax(out, dim = 1)

            trainAcc += 100.0 * (preds == label).sum() / len(label)

            

            # Backpropagation

            loss.backward()

            optimizer.step()

            

        lr_decay.step(trainLoss / len(trainloader))  # Learning rate decay

        

        trainLoss = trainLoss / len(trainloader)

        trainAcc = trainAcc / len(trainloader)

        LOSS.append(trainLoss)

        ACC.append(trainAcc)

        

        model.eval()

        valLoss = 0.0

        valAcc = 0.0

        

        with torch.no_grad():

            for data, label in CVloader:

                # Converting for using GPU

                data = data.to(device)

                label = label.to(device)

                

                out = model(data.float())

                valLoss += criterion(out, label)

                

                preds = torch.argmax(out, dim = 1)

                valAcc += 100.0 * (preds == label).sum() / len(label)

        

        valLoss = valLoss / len(CVloader)

        valAcc = valAcc / len(CVloader)

                

        # Printing loss and accuracy

        if epoch % 3 == 2 or epoch == 0:

            print('Epochs:', epoch + 1)

            print('Traning Loss = {}  Traning Accuracy = {} '.format(trainLoss, trainAcc))

            print('CrossValLoss = {}  CrossValAccuracy = {}'.format(valLoss, valAcc))

    print("\nTraining Finished")

    print('Training Accuracy = ', trainAcc)

    print('Validation Accuracy = ', valAcc)
train_model(30)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test = test.to_numpy()

test = np.reshape(test, [-1, 1, 28, 28]).astype(np.float32())

test = test / 255

test = torch.from_numpy(test)



with torch.no_grad():

    test = test.to(device)

    out = model(test.float())

    preds = torch.argmax(out, dim = 1)

    preds = preds.cpu().numpy()

preds = np.reshape(preds, preds.shape[0])



sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

sub['Label'] = preds

sub.to_csv('submission.csv',index=False)
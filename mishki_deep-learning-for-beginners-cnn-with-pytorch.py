import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import DataLoader, Dataset



import pandas as pd



from torchvision import datasets, transforms

from torchvision.utils import make_grid



import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv',sep=',')

test_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv', sep = ',')



class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']
print("train_data dimensions: ", train_data.shape)

train_data.head(3)
print("train_data dimensions: ", test_data.shape)

test_data.head(3)
#let's look at the first example in our training dataset 

label = train_data.iloc[0,0]

image = train_data.iloc[0,1:]

print('Shape:', image.shape, '\nLabel:', label, '\nClass:', class_names[label])



torch_tensor = torch.tensor(image.values)

plt.figure(figsize=(1,1))

plt.imshow(torch_tensor.reshape((28,28)), cmap="gray");
class FashionMNISTDataset(Dataset):

    def __init__(self, file_path):

        data = pd.read_csv(file_path,sep=',')

        labels = torch.tensor(data['label'].values)

        images = torch.tensor(data.drop(columns=['label']).values)

        

        #convert the pandas DataFrames to torch tensors

        #separate tensors for labels and for features (our image pixels)

        self.labels = torch.tensor(labels)

        self.images = torch.reshape(images,(-1,1, 28,28))

    

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self, idx):

        return self.images[idx],self.labels[idx]
train_data = FashionMNISTDataset('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True) 
for images,labels in train_loader: 

    break

    

#check our data format

print("Shape of our chunk of images: ",images.shape)

    

# Print the labels

print('Labels:', labels.numpy())

print('Classes:', *np.array([class_names[i] for i in labels]))



img = make_grid(images, nrow=10) 



plt.figure(figsize=(12,6))

plt.imshow(np.transpose(img.numpy(), (1, 2, 0))); #we need to transpose our images because plt.imshow expects another shape
test_data = FashionMNISTDataset('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

test_loader = DataLoader(test_data, batch_size = 10, shuffle = False)  
class myCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1,6,3,1)

        self.conv2 = nn.Conv2d(6,16,3,1)

        self.fc1 = nn.Linear(5*5*16,120)

        self.fc2 = nn.Linear(120,84)

        self.fc3 = nn.Linear(84,10)

        

    def forward(self, X):

        X = F.relu(self.conv1(X))

        X = F.max_pool2d(X,2,2)

        X = F.relu(self.conv2(X))

        X = F.max_pool2d(X,2,2)

        X = X.view(-1, 5*5*16) # we need to flatten our data to 1D before inputting it to our fully connected layers. 

        X = F.relu(self.fc1(X))

        X = F.relu(self.fc2(X))

        return F.log_softmax(X, dim=1)

    

torch.manual_seed(101)

model = myCNN()

model
def param_count(model):

    params = [p.numel() for p in model.parameters() if p.requires_grad]

    for item in params:

        print(item)

    print('Total: ', sum(params))

    

param_count(model)
eval_crit = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
type(len(train_data))
epochs = 10



import time

start = time.time()



"""

Let's keep track of the training losses by appending them to a vector

At the end of training, we will plot the losses on the training set and on the test set for each epoch

This will help us see if there is a potential to improve results if we train for more epochs.

"""

train_losses = []

test_losses = []



#Let's also keep track of the number of correct classifications in each epoch

train_correct = []

test_correct = []





for i in range(epochs):

    

    train_correct_pred = 0

    test_correct_pred = 0

    

    for X_train, y_train in train_loader:

        #apply our model

        y_pred = model(X_train.float())

        

        #compute the loss

        loss = eval_crit(y_pred, y_train)



        """

        Count the number of correctly predicted items:

        y_pred contains the probability for the training examples to belong to each of the 10 possible classes

        therefore, in order to get the predicted class for each example, we need to do this for each row:

        find the maximum probabiliy and return its column number 

        """

        predicted_classes = torch.max(y_pred.data, 1)[1]

        train_correct_pred += (predicted_classes == y_train).sum() #add the number of correctly predicted items in this batch

        

        #update the model parameters

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

    train_losses.append(loss)

    train_correct.append(train_correct_pred.item()*100/len(train_data))

    

    print(f'Epoch: {i+1} Loss: {loss:7.5} Accuracy: {train_correct_pred.item()*100/len(train_data):4.2f} ({train_correct_pred} correct out of {len(train_data)})')

    

    #Evaluate the performance of the currect model on the test set

    #we first set our model into eval mode

    with torch.no_grad():

        for X_test, y_test in test_loader:

            #apply our model

            y_pred = model(X_test.float())

            

            predicted_classes = torch.max(y_pred.data, 1)[1]

            test_correct_pred += (predicted_classes == y_test).sum()

   

        loss = eval_crit(y_pred, y_test)

        test_losses.append(loss)

        test_correct.append(test_correct_pred.item()*100/len(test_data))

    



print(f'Duration of training (min): {(time.time()-start)/60}')
plt.figure(figsize=(6,4))

plt.plot(train_losses, label='training loss')

plt.plot(test_losses, label='test set loss')

plt.title('Loss at the end of each epoch')

plt.ylim(0,1)

plt.legend();
plt.figure(figsize=(6,4))

plt.plot(train_correct, label='training accuracy')

plt.plot(test_correct, label='test set accuracy')

plt.title('Accuracy (%) at the end of each epoch')

plt.ylim(50,100)

plt.legend();
torch.save(model.state_dict(), 'FashionMNIST-CNN-Model-Batch10.pt')
def train_model(batch_size=10):



    train_data = FashionMNISTDataset('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True) 



    test_data = FashionMNISTDataset('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True) 



    torch.manual_seed(101)

    model = myCNN()



    eval_crit = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    epochs = 10



    start = time.time()



    """

    Let's keep track of the training losses by appending them to a vector

    At the end of training, we will plot the losses on the training set and on the test set for each epoch

    This will help us see if there is a potential to improve results if we train for more epochs.

    """

    train_losses = []

    test_losses = []



    #Let's also keep track of the number of correct classifications in each epoch

    train_correct = []

    test_correct = []





    for i in range(epochs):



        train_correct_pred = 0

        test_correct_pred = 0



        for X_train, y_train in train_loader:

            #apply our model

            y_pred = model(X_train.float())



            #compute the loss

            loss = eval_crit(y_pred, y_train)



            """

            Count the number of correctly predicted items:

            y_pred contains the probability for the training examples to belong to each of the 10 possible classes

            therefore, in order to get the predicted class for each example, we need to do this for each row:

            find the maximum probabiliy and return its column number 

            """

            predicted_classes = torch.max(y_pred.data, 1)[1]

            train_correct_pred += (predicted_classes == y_train).sum() #add the number of correctly predicted items in this batch



            #update the model parameters

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



        train_losses.append(loss)

        train_correct.append(train_correct_pred.item()*100/len(train_data))



        print(f'Epoch: {i+1} Loss: {loss:7.5} Accuracy: {train_correct_pred.item()*100/len(train_data):4.2f} ({train_correct_pred} correct out of {len(train_data)})')



        #Evaluate the performance of the currect model on the test set

        #we first set our model into eval mode

        model.eval()

        with torch.no_grad():

            for X_test, y_test in test_loader:

                #apply our model

                y_pred = model(X_test.float())



                predicted_classes = torch.max(y_pred.data, 1)[1]

                test_correct_pred += (predicted_classes == y_test).sum()



            loss = eval_crit(y_pred, y_test)

            test_losses.append(loss)

            test_correct.append(test_correct_pred.item()*100/len(test_data))

        model.train()

    

    print(f'Duration of training (min): {(time.time()-start)/60}')

    

    plt.figure(figsize=(6,4))

    plt.plot(train_losses, label='training loss')

    plt.plot(test_losses, label='test set loss')

    plt.title('Loss at the end of each epoch')

    plt.ylim(0,0.8)

    plt.legend();

    

    plt.figure(figsize=(6,4))    

    plt.plot(train_correct, label='training accuracy')

    plt.plot(test_correct, label='test set accuracy')

    plt.title('Accuracy (%) at the end of each epoch')

    plt.ylim(50,100)

    plt.legend();
train_model(100)

torch.save(model.state_dict(), 'FashionMNIST-CNN-Model-Batch100.pt')
train_model(4)

torch.save(model.state_dict(), 'FashionMNIST-CNN-Model-Batch4.pt')
#instantiate the model

final_model = myCNN()



#load the saved parameters

final_model.load_state_dict(torch.load('FashionMNIST-CNN-Model-Batch10.pt'))
final_model.eval() #sets our model into evaluation mode <-- check PyTorch documentation

with torch.no_grad(): #prevents adjusting the network parameters by mistake

    corr_pred = 0

    for X_test, y_test in test_loader:

        y_pred = final_model(X_test.float())

        # for each test example, the prediction is a vector of 10 items. Item number k represents the probability of this

        # example belonging to class k. To obtain the class label from this vector, we extract the position of the item

        # with the highest value in the 10-item array. 

        labels_pred = torch.max(y_pred,1)[1] #get the labels for the predictions

        corr_pred += (labels_pred == y_test).sum()

        

print(f'Accuracy: {corr_pred.item()*100/(len(test_data))}%')
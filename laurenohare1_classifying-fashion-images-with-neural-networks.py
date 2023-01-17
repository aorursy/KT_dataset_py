import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader



import matplotlib.pyplot as plt



import numpy as np



from sklearn.metrics import confusion_matrix



import os



import random



print("Setup complete!")
random.seed(1)

torch.manual_seed(1)

torch.cuda.manual_seed(1)

np.random.seed(1)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False
print_images = datasets.FashionMNIST('data', train=True, download=True)



for k, (image, label) in enumerate(print_images):

    if k >= 18:

        break

    plt.subplot(3, 6, k+1)

    plt.imshow(image)
# transform the images to a tensor

fashion_mnist_data = datasets.FashionMNIST("data", train = True , download = True, transform = transforms.ToTensor())



fashion_mnist_data = list(fashion_mnist_data)
# check the length of the list to see how many images we have in the data

print(len(fashion_mnist_data))
# For this example I want 80% of the images for my training (48,000)

# I want the remaining 20% for testing (12,000)



fashion_mnist_train = fashion_mnist_data[0:48000]

fashion_mnist_test = fashion_mnist_data[48000 : 60000]



print(len(fashion_mnist_train))

print(len(fashion_mnist_test))
first_img, first_lab = fashion_mnist_train[0]

print(first_img.size())

print(first_lab)
# pytorch uses batches of images - lets use 64

trainloader = torch.utils.data.DataLoader(fashion_mnist_train,

                                          batch_size=64,

                                          num_workers = 0,

                                          shuffle = True)



testloader = torch.utils.data.DataLoader(fashion_mnist_test,

                                          batch_size=64,

                                          num_workers = 0,

                                          shuffle = True)
class FCN_Model(nn.Module):

    def __init__(self):

        super(FCN_Model, self).__init__()

        self.fc1 = nn.Linear(1 * 28 * 28, 50) #input layer connected to first hidden layer

        self.fc2 = nn.Linear(50, 20) #first hidden layer connected to second hidden layer

        self.fc3 = nn.Linear(20, 10) #second hidden layer connected to output layer

        

    def forward(self, img):

        

        #flatten the image

        flatten = img.view(-1, 28 * 28) #flatten the image dimensions

        

        #activation layers

        acti1 = F.relu(self.fc1(flatten)) #use the relu activation

        acti2 = F.relu(self.fc2(acti1))   #introduces non linearity

        output = self.fc3(acti2)

        return output
def train(model, train, test, batch_size=64, num_iters=1, learn_rate=0.01, weight_decay=0):

    train_loader = torch.utils.data.DataLoader(train,

                                               batch_size=batch_size,

                                               shuffle=True) # shuffle after every epoch

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=weight_decay)



    iters, losses, train_acc, test_acc = [], [], [], []



    # training

    n = 0 # the number of iterations

    while True:

        if n >= num_iters:

            break

        for imgs, labels in iter(train_loader):

            model.train() 

            out = model(imgs)             # forward pass

            loss = criterion(out, labels) # compute the total loss

            loss.backward()               # backward pass (compute parameter updates)

            optimizer.step()              # make the updates for each parameter

            optimizer.zero_grad()         # a clean up step for PyTorch



            # save the current training information

            if n % 10 == 9:

                iters.append(n)

                losses.append(float(loss)/batch_size)        # compute *average* loss

                train_acc.append(get_accuracy(model, train)) # compute training accuracy 

                test_acc.append(get_accuracy(model, test))   # compute testing accuracy

            n += 1



    # plotting

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)

    plt.title("Training Curve")

    plt.plot(iters, losses, label="Train")

    plt.xlabel("Iterations")

    plt.ylabel("Loss")



    plt.subplot(1,2,2)

    plt.title("Training Curve")

    plt.plot(iters, train_acc, label="Train")

    plt.plot(iters, test_acc, label="Testing")

    plt.xlabel("Iterations")

    plt.ylabel("Training Accuracy")

    plt.legend(loc='best')

    plt.show()



    print("Final Training Accuracy: {}".format(train_acc[-1]))

    print("Final Testing Accuracy: {}".format(test_acc[-1]))

    

def get_accuracy(model, data):

    correct = 0

    total = 0

    model.eval()

    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64):

        output = model(imgs)

        pred = output.max(1, keepdim=True)[1] # get the index of the max logit

        correct += pred.eq(labels.view_as(pred)).sum().item()

        total += imgs.shape[0]

    return correct / total
# create an instance and start training!

model1 = FCN_Model()

train(model1 , fashion_mnist_train, fashion_mnist_test, num_iters= 700)
model1.eval() # model1 is the name given to my model. In this case a fully connected neural network as defined above.



true_class = []

predicted_class = []



for data , target in testloader: #image and label from the testing images

    for label in target.cpu().data.numpy():  #this is to turn the tensor into numpy array

        true_class.append(label)

    for prediction in model1.cpu()(data).data.numpy().argmax(1): #this is to turn the tensor into numpy array + position of 

        predicted_class.append(prediction)                      # largest value

        

confusion_matrix(true_class, predicted_class)
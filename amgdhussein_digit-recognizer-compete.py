import os

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import torchvision
root = Path("../input/digit-recognizer")
os.listdir(root)
train_file, test_file, submission_file = 'train.csv', 'test.csv', 'sample_submission.csv'
def load_data(root, file_name):

    return pd.read_csv(root/file_name)



digits = load_data(root, train_file)
digits.head()
digits.shape
features, labels = digits.drop(columns = ["label"], axis = 1), digits.label.to_numpy()
import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

with sns.axes_style(style = "whitegrid"):

    sns.countplot(x = labels)

def show_image(image, label):

    plt.figure(figsize = (2, 2))

    plt.axis("off")

    plt.title("%s"%label)

    plt.imshow(X = image, cmap = "gray")

    plt.show()
image, label = features.iloc[10].to_numpy().reshape(28, 28), labels[10]

show_image(image, label)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(features)
def clean(features):

    features = features.astype(float)

    features = scaler.transform(features)

    features = torch.Tensor(features).view(len(features), 28, 28)

    return features
features, labels = clean(features), torch.from_numpy(labels)
from torch.utils.data import TensorDataset

from torch.utils.data import random_split
dataset = TensorDataset(features, labels)
lenghts = [34000, 8000]

train_set, valid_set = random_split(dataset, lenghts)
from torch.utils.data.dataloader import DataLoader
def get_data_loader(dataset, batch_size):

    

    data_loader = DataLoader(

        dataset = dataset,

        batch_size= batch_size,

        shuffle=True,

        num_workers = 2,

        pin_memory=True,

    )

    

    return data_loader
batch_size = 128



train_loader = get_data_loader(train_set, batch_size)

valid_loader = get_data_loader(valid_set, batch_size)
def show_batch(batch):

    images, labels = batch

    fig = plt.figure(figsize=(18,24))



    for i in range(1, len(images)+1):

        fig.add_subplot(16,16,i)

        plt.axis("off")

        plt.imshow(images[i-1],cmap='gray')

        plt.title(labels[i-1].item())

        

    plt.show()

    
batch = next(iter(train_loader))
show_batch(batch)
classes = list(range(10))
batch[0].shape
from torchvision import models

from torch import nn

from torch.nn.functional import cross_entropy, softmax

from torch.optim import Adam

from torch.optim.lr_scheduler import ExponentialLR

import time
class Learner(object):

    def __init__(self, model, train_dl, valid_dl = None, opt_fn = Adam, loss_fn=cross_entropy):

        

        self.model = model

        self.train_dl = train_dl

        self.valid_dl = valid_dl

        self.loss_fn = loss_fn

        self.optimizer = opt_fn



        self.metrics = {

            "train_loss": [],

            "val_loss" : [],

            "val_acc" : [],

        }

    

    def fit(self, epochs = 10, lr = 0.1, rate_decay = 0.0, weight_decay = 0.0):

        

        optimizer = self.optimizer(params = self.model.parameters(), lr = lr, weight_decay = weight_decay,)

        lr_scheduler = ExponentialLR(optimizer = optimizer, gamma = rate_decay)



        if self.valid_dl:

            print("epoch     train_loss    valid_loss    accuracy    time(in minute)")

        for epoch in range(0, epochs):



            start_time = time.time()

            for images, labels in self.train_dl:



                optimizer.zero_grad()

                probs = self.model(images)      

                loss = self.loss_fn(probs, labels)            

                loss.backward()            

                optimizer.step()            

                

            lr_scheduler.step() 

            end_time = time.time()

            

            

            if self.valid_dl:

                val_acc, val_loss = self.evaluate()



                self.metrics["val_acc"].append(val_acc)

                self.metrics["val_loss"].append(val_loss)

                self.metrics["train_loss"].append(loss.item())



                print(self.metrics_format(epoch, loss.item(), val_loss, val_acc, (end_time - start_time)/60))



    def metrics_format(self, epoch, loss, val_loss, val_acc, time):

        return f"{epoch:3}       {loss:0.6f}      {val_loss:0.6f}      {val_acc:0.4f}      {time:0.2f}"



    @torch.no_grad()

    def evaluate(self):

        size = len(self.valid_dl)

        acc, loss = np.zeros(size, dtype = float), np.zeros(size, dtype = float)



        for i, batch in enumerate(self.valid_dl):



            images, labels = batch

            probs = self.model(images)

            acc[i] = self.accuracy(probs, labels)

            loss[i] = self.loss_fn(probs, labels).item()



        return acc.mean(), loss.mean()



    

    def accuracy(self, out, target):

        

        probs = softmax(input = out, dim = 1)

        _, predictions = torch.max(probs, dim = 1)



        return torch.sum(predictions == target).item() / len(target)

            

    def model_complexity_graph(self):

    

        val_loss = self.metrics["val_loss"]

        train_loss = self.metrics["train_loss"]

        val_accuracy = self.metrics["val_acc"]

        x = range(len(val_loss))



        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 6))



        ax1.set_title('Validation Accuracy')



        ax1.plot(

            x, val_accuracy, marker='o', markerfacecolor='darkred', markersize=8, 

            color='lightcoral', linewidth=2

        )





        ax2.set_title('Model Complexity Graph')



        ax2.plot(

            x, val_loss, marker='o', markerfacecolor='darkred', markersize=8, 

            color='lightcoral', linewidth=2, label = "val loss"

        )



        ax2.plot(

            x, train_loss, marker='o', markerfacecolor='blue', markersize=8,

            color='skyblue', linewidth=2, label = "train loss"

        )



        ax2.legend()



        plt.show()

        

    
class Network(nn.Module):

    def __init__(self, in_features, classes_size):

        super(Network, self).__init__()



        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(in_features = in_features, out_features = 512, bias=True),

            nn.ReLU(),

            nn.Linear(in_features = 512, out_features = 128, bias=True),

            nn.ReLU(),

            nn.Linear(in_features = 128, out_features = classes_size, bias=True),

        )

        

    def forward(self, xb):

        return self.classifier(xb)
class DeviceDataLoader(object):

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def batch_to_device(self, features, labels, device):

        return features.to(device, non_blocking = True), labels.to(device, non_blocking = True)



    def __iter__(self):

        for features, labels in self.dl: 

            yield self.batch_to_device(features, labels, self.device)



    def __len__(self):

        return len(self.dl)
default_device = torch.device('cpu')



if torch.cuda.is_available():

    default_device = torch.device('cuda')

    print("Running on a GPU")
train_dl = DeviceDataLoader(train_loader, default_device)

valid_dl = DeviceDataLoader(valid_loader, default_device)
model = Network(

    

    in_features = 28*28,

    classes_size = len(classes),

    

).to(default_device)
learner = Learner(model, train_dl, valid_dl)
epochs = 30

learning_rate = 0.0001

weight_decay = 1e-4

rate_decay = 0.96
%%time

learner.fit(epochs = epochs, lr = learning_rate, rate_decay = rate_decay, weight_decay = weight_decay)
learner.model_complexity_graph()
set_features = load_data(root, test_file)

digit_test = clean(set_features)
random_indices = [np.random.randint(0, len(digit_test)) for i in range(10)]

samples = [digit_test[index] for index in random_indices]
def predict(model, x):

    probs = model(x.view(1, 28, 28))

    _, prediction = torch.max(probs, dim = 1)

    return prediction.item()
for image in samples:

    pred = predict(model, image.to(default_device))

    show_image(image, f"Predicted: {classes[pred]}")
probs = model(digit_test.to(default_device))

_, outputs = torch.max(probs, dim = 1)

predictions = [output.item() for output in outputs]
submission = load_data(root, submission_file)

submission.head()
submission.Label = predictions

submission.to_csv("submission.csv", index = False)
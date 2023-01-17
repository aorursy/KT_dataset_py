import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from tqdm import tqdm

import torch.optim as optim
print("Tensor = [[1, 2], [3, 4]]")

print("Dimension 0: ")

print("{}\n".format(F.softmax(torch.Tensor([[1,2],[3,4]]), dim=0)))

print("Dimension 1: ")

print("{}".format(F.softmax(torch.Tensor([[1,2],[3,4]]), dim=1)))
# File paths



train_path = r"../input/fashionmnist/fashion-mnist_train.csv"

test_path = r"../input/fashionmnist/fashion-mnist_test.csv"

train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)
# Training data dataframe to numpy

y_data = train_data['label'].values

X_data = train_data.drop(['label'], axis=1).values



# Testing data dataframe to numpy

y_test_data = test_data['label'].values

X_test_data = test_data.drop(['label'], axis=1).values
# Load training data and convert it to tensors



X_train = torch.tensor(X_data, dtype=torch.float32)

X_train = X_train/255

y_train = torch.tensor(y_data, dtype=torch.int64)



#Loading testing data

y_test = torch.tensor(y_test_data, dtype=torch.int64)

X_test = torch.tensor(X_test_data, dtype=torch.float32)
# Build the neural network class



class Net(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)

        self.fc2 = nn.Linear(64, 80)

        self.fc3 = nn.Linear(80, 120)

        self.fc4 = nn.Linear(120, 10)



        

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return F.log_softmax(x, dim=1)
if torch.cuda.is_available():

    device = torch.device("cuda:0")

    print("Running on a GPU")

else:

    device = torch.device("cpu")

    print("Running on a CPU")
model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
EPOCHS = 15

BATCH_SIZE = 20



for epoch in range(EPOCHS):

    for i in tqdm(range(0, len(X_train), BATCH_SIZE)):

        batch_X = X_train[i:i+BATCH_SIZE].view(-1, 28*28)

        batch_y = y_train[i:i+BATCH_SIZE]

        

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        

        model.zero_grad()

        output = model(batch_X)

        loss = F.nll_loss(output, batch_y)

        loss.backward() # Backpropagate the loss with respect to weights and biases

        optimizer.step() # Update weights and biases base on the loss

        

    print("Epoch {}: Loss: {} - Accuracy: {}".format(epoch+1, loss, 1.0-loss))
correct = 0

total = 0

TEST_BATCH = 20





with torch.no_grad():

    for i in tqdm(range(0, len(X_test), TEST_BATCH)):

        batch_X = X_test[i:i+TEST_BATCH].view(-1,28*28).to(device)

        batch_y = y_test[i:i+TEST_BATCH]

        preds = model(batch_X)

        

        out_maxes = [torch.argmax(i) for i in preds]

        

        for idx, j in enumerate(out_maxes):

            if j == batch_y[idx]:

                correct+=1

            total+=1

        

    print("val accuracy-{}".format(correct/total))
import matplotlib.pyplot as plt
with torch.no_grad():

    predictions = model(X_test[0:5].view(-1,28*28).to(device))

    pred_class = [torch.argmax(i).tolist() for i in predictions]
class_label = {0: 'T-shirt/top',

            1:  'Trouser',

            2: 'Pullover',

            3:  'Dress',

            4: 'Coat',

            5: 'Sandal',

            6: 'Shirt',

            7: 'Sneaker',

            8: 'Bag',

            9: 'Ankle boot'}

fig, axes = plt.subplots(1,5, figsize=(20,5))

axes[0].imshow(X_test[0].view(28,28))

axes[0].set_title(class_label[pred_class[0]])



axes[1].imshow(X_test[1].view(28,28))

axes[1].set_title(class_label[pred_class[1]])



axes[2].imshow(X_test[2].view(28,28))

axes[2].set_title(class_label[pred_class[2]])



axes[3].imshow(X_test[3].view(28,28))

axes[3].set_title(class_label[pred_class[3]])



axes[4].imshow(X_test[4].view(28,28))

axes[4].set_title(class_label[pred_class[4]])
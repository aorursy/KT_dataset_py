import random

import numpy as np

import pandas

import csv

import keras # really just want for utils not keras itself to one-hot

			 # some categorical labels because I am lazy



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
######################

## Data Preparation ##

######################

training_data_fpath = "../input/train.csv"

test_data_fpath = "../input/test.csv"



training_data = pandas.read_csv(training_data_fpath)

test_data = pandas.read_csv(test_data_fpath)



# Training data as data matrices

y_train = np.array(training_data["label"])

x_train = np.array(training_data.drop(labels = ["label"], axis=1))

x_train = x_train.astype("float32")



# Test data as data matrix

x_test = np.array(test_data)

x_test = x_test.astype("float32")



# Change labels into binary categorical vectors

y_train = keras.utils.to_categorical(y_train, 10)



# Create observation and label pairs for each of the datasets

training_data = list(zip(x_train,y_train))

test_data = x_test
def create_mini_batches(data, batch_size):

	"""

	Returns batched dataset

	"""

	batches = [data[k:k+batch_size] for k in range(0, len(data), batch_size)]

	torch_batches = []

	for mini_batch in batches:

		mini_batch_obs = []

		mini_batch_labels = []

		for pair in mini_batch:

			x,y = pair

			mini_batch_obs.append(x)

			mini_batch_labels.append(y)

		torch_batches.append((torch.from_numpy(np.array(mini_batch_obs)), torch.from_numpy(np.array(mini_batch_labels))))

	return torch_batches
#############################

## Architecture Definition ##

#############################

class SimpleNet(nn.Module):

	def __init__(self):

		super(SimpleNet, self).__init__()

		self.fc1 = nn.Linear(784, 256)

		self.fc2 = nn.Linear(256, 64)

		self.fc3 = nn.Linear(64, 10)



	def forward(self, x):

		x = F.sigmoid(self.fc1(x))

		x = F.sigmoid(self.fc2(x))

		x = self.fc3(x)

		return x
# Instantiate Network, define loss function, and optimizer for instance of SimpleNet

net = SimpleNet()

criterion = nn.MSELoss(reduction="mean")

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

batch_size = 128

epochs = 500
############################

## Training the SimpleNet ##

############################

for epoch in range(epochs):

	running_loss = 0.0

	random.shuffle(training_data)

	batched_training_data = create_mini_batches(training_data, batch_size)

	for x_batch, y_batch in batched_training_data:

		# forward pass

		preds = net(x_batch)



		# backward pass

		loss = criterion(preds, y_batch)

		optimizer.zero_grad()

		loss.backward()



		# Update parameters

		optimizer.step()



		# Print progress

		running_loss += loss.item()

	if epoch % 10 == 0:

		print("Epoch [%d] Loss: %.3f" % (epoch, running_loss))



print ("\n ### Finished Training ### \n")
#######################

## Work on Test Data ##

#######################

# Make test_data compliant with our network

torch_test_data = []

for x in test_data:

	torch_test_data.append(torch.from_numpy(x))

            

with torch.no_grad():

    imgid = []

    labels = []

    image_num = 1

    for x in torch_test_data:

        prediction = net(x)

        max_value, predicted_label = torch.max(prediction.data,0)

        imgid.append(image_num)

        labels.append(int(predicted_label))

        image_num += 1

    my_submission = pandas.DataFrame({'ImageID': imgid, 'Label': labels})

    # you could use any filename. We choose submission here

    my_submission.to_csv('submission.csv', index=False)
import os

import random

import cv2

from os import listdir



os.makedirs('training')

os.makedirs('training/airplane')

os.makedirs('training/car')

os.makedirs('training/cat')

os.makedirs('training/dog')

os.makedirs('training/flower')

os.makedirs('training/fruit')

os.makedirs('training/motorbike')

os.makedirs('training/person')



os.makedirs('testing')

os.makedirs('testing/airplane')

os.makedirs('testing/car')

os.makedirs('testing/cat')

os.makedirs('testing/dog')

os.makedirs('testing/flower')

os.makedirs('testing/fruit')

os.makedirs('testing/motorbike')

os.makedirs('testing/person')



imagePath = r'/kaggle/input/natural-images/data/natural_images'

rate = 0.8

fList = listdir(imagePath)



for imgClass in fList:

	imageList = listdir(imagePath + '/' + imgClass)

	random.shuffle(imageList)

	count = 0

	for f in imageList:

		splitData = 'training'

		if (count > len(imageList) * rate):

			splitData = 'testing'



		cv2.imwrite(splitData + '/' + imgClass + '/' + f,

			cv2.imread(imagePath + '/' + imgClass + '/' + f))



		count += 1

import torch

from torchvision import datasets, models, transforms 

import urllib

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from torch.autograd import Variable

from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader

import cv2

from os import listdir

from torchvision.datasets import ImageFolder

import sys

import time

import torchvision



preprocess = transforms.Compose([		

	transforms.Resize((256,256)),

	transforms.ToTensor(),

	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])



trainloader = DataLoader(dataset= ImageFolder(r'training', transform=preprocess),

                         batch_size=1, shuffle=True)

# define loss function

loss_fn = torch.nn.MSELoss()



# define total number of epoch that all data feed into network to be trained.

totalEpoch = 10

outputClasses = 8



# define model

model = models.alexnet(pretrained=True)

for param in model.parameters():

	param.requires_grad = False



model.classifier[4] = nn.Linear(4096, 1024)

model.classifier[6] = nn.Linear(1024, outputClasses)



optimizer = torch.optim.AdamW(model.classifier.parameters())

model.cuda()
start_time = time.time()

for epoch in range(totalEpoch):

	lossTotal = 0

	for i, data in enumerate(trainloader, 0):        

		# Zero gradients, perform a backward pass, and update the weights.

		optimizer.zero_grad()



		# Forward pass: Compute predicted y by passing x to the model

		y_pred = model(data[0].cuda())



		# Compute and print loss				

		temp = torch.nn.functional.one_hot(data[1], outputClasses).type(torch.FloatTensor).cuda()           

		loss = loss_fn(y_pred, temp)								

		lossTotal += loss



		# perform a backward pass (backpropagation)

		loss.backward()



		# Update the parameters

		optimizer.step()        

    

	print("epoch : ", epoch , ", ",time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

	print("loss : ", lossTotal / i)



print("Finish training")
testloader = DataLoader(dataset=ImageFolder(r'testing', transform=preprocess),

                        batch_size=1, shuffle=True)

correct = 0

total = 0

with torch.no_grad():

	for i, data in enumerate(testloader, 0):		

		images, labels = data

		outputs = model(images.cuda())

		_, predicted = torch.max(outputs.data, 1)

		total += labels.size(0)



		if predicted == labels.cuda():

			correct += 1



	print('Accuracy of the network on the {} test images: {} , (error = {})'.format(total,

				100 * correct / total, total - correct))
#inporting the necessary libraries

import numpy as np

import torch

from torch import nn

from torchvision import transforms, datasets, models

import math

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision as tv
#check if CUDA is available

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:

    print("CUDA is available. Training on GPU!")

else:

    print("CUDA is not available. Training on CPU.")
#time to prepare the data



batch_size = 32

test_size = 0.20

valid_size = 0.20



mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]





#define the transforms

train_transform  = transforms.Compose([

                                        transforms.Resize((400,400)),

                                        transforms.RandomRotation(359),

                                        transforms.RandomHorizontalFlip(0.2),

                                        transforms.RandomVerticalFlip(0.2),

                                        transforms.ToTensor(),

                                        transforms.Normalize(mean=mean, std=std)])







train_data = datasets.ImageFolder("/kaggle/input/single-cell-morphological-dataset-of-leukocytes/blood_smear_images_for_aml_diagnosis_MOD/AML-Cytomorphology_LMU_MOD",

                                  transform = train_transform)



#obtain training indicies that will be used as testing and validation



num_train = len(train_data)

indicies = list(range(num_train))

np.random.shuffle(indicies)

test_split = int(np.floor(test_size * num_train))

valid_split = int(np.floor(valid_size * num_train))



train_idx, valid_idx, test_idx = indicies[test_split+valid_split:], indicies[:valid_split], indicies[valid_split:test_split+valid_split]



#define samplers for obtainig the trainig, testing and validation set

train_sampler = SubsetRandomSampler(train_idx)

test_sampler = SubsetRandomSampler(test_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,

                                           sampler = train_sampler)

test_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,

                                          sampler = test_sampler)



valid_loader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,

                                          sampler = valid_sampler)





#  PROMYELOCYTE (PMB Promyelocyte (bilobled))

# PMO Promyelocyte), MYELOCYTE (MYB Myelocyte, MYO Myeloblast)ARE FOUND ON LEUKEMIA PATIENTS 

classes = ['BAS', 'EBO', 'EOS', 'KSC','LYT','MON', 'MYO', 'NGB', 'NGS', 'PMO']



%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torchvision.transforms.functional as F



def imshow(img):

  img = img /2+0.5 #unormalize the images

  plt.imshow(np.transpose(img, (1, 2, 0))) #convert it back from tensor to image



#get one batch of training images

dataiter = iter(train_loader) #now contains the first batch

images, labels = dataiter.next() #images=the first batch of images, labels= the first batch of labels

images = images.numpy() #convert the images to display them



#plot the imahes in the batch along with the corresponding labels

fig = plt.figure(figsize=(25,6))



for idx in np.arange(20):

  ax = fig.add_subplot(1, 20, idx+1, xticks=[], yticks=[]) #(rows, cols, index, .., ..)

  imshow(images[idx])

  ax.set_title(classes[labels[idx]])
#Load AlexNet pretrained model

model = models.vgg16(pretrained=True)

model
#freeze the model calssifier

for param in model.features.parameters():

  param.requires_grad = False



from collections import OrderedDict



classifier = nn.Sequential(OrderedDict([

                            

                          ('fc1', nn.Linear(4096, 1024)),

                          ('relu', nn.ReLU()),

                          ('fc2', nn.Linear(1024, 10))]))

#('output', nn.LogSoftmax(dim=1)



model.classifier[6] = classifier

model
import torch.optim as optim

#Loss function and optmixation function

criterion = nn.CrossEntropyLoss()

# the optimizer accepts only the trainable parameters

optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)



if train_on_gpu:

    model.cuda()



model
# number of epochs to train the model

import numpy as np

n_epochs = 20



valid_loss_min = np.Inf # track change in validation loss



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for images, labels in train_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(images)

        # calculate the batch loss (comapre the values of the output model to the actual labels)

        loss = criterion(output, labels)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item()*images.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval()

    for images, labels in valid_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(images)

        # calculate the batch loss

        loss = criterion(output, labels)

        # update average validation loss 

        valid_loss += loss.item()*images.size(0)

    

    # calculate average losses

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        valid_loss_min = valid_loss

        # print the decremnet in the validation

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, train_loss, valid_loss))

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min, 

        valid_loss))

        torch.save(model.state_dict(), 'model_AML_classifier.pt')

        

    if epoch % 10 == 0:    

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, train_loss, valid_loss))
model.load_state_dict(torch.load('model_AML_classifier.pt'))
#initialize the test loss

test_loss = 0.0



class_correct = list(0. for i in range(10))

class_total = list (0. for i in range(10))



#set the model to test and validation mode (no gradient descent needed)

model.eval()



for data, target in test_loader:

  #move the tensor to GPU ig available

  if train_on_gpu:

    data, target = data.cuda(), target.cuda()

  #forward pass: compute prediction output by passing the first batch of test data

  output = model(data)

  #calculate the batch size

  loss = criterion(output, target)

  #update the test loss

  test_loss += loss.item()*data.size(0)

  #convert output probabilities to output class

  _, pred = torch.max(output, 1)

  #compare the prediction to true label

  correct_tensor = pred.eq(target.data.view_as(pred))

  #conveert to numpy array and remove the extra dimention and get only the result

  correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())



  #calculate test accuracy for each object class

  for i in range(batch_size):

    try:

      label = target.data[i] #get the corresponding label from the object

      class_correct[label] += correct[i].item()

      class_total[label] += 1

    except IndexError:

      break

  

# average test loss

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(10):

  if class_total[i] > 0:

     print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

  else:

       print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))

   





#Move model inputs to cuda

if train_on_gpu:

    images = images.cuda()



#get sample outputs

output = model(images)

#convert probabilties to prediction class

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())



# plot the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(10):

    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])

    imshow(images.cpu()[idx])

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))
# Experiment parameters

lr_find_epochs = 2

start_lr = 1e-7

end_lr = 0.1
# Set up the model, optimizer and loss function for the experiment



optimizer = torch.optim.SGD(model.classifier.parameters(), lr=start_lr)

criterion = nn.CrossEntropyLoss()



# y = a.e(-bt) 

# end_lr = start_lr . e(b.t)

# (end_lr - start_lr) = e(b.t)

# ln(end_lr - start_lr) = b.t

# b = ln(end_lr - start_lr) / t
# LR function lambda

from torch.optim.lr_scheduler import LambdaLR

lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len( train_loader)))

scheduler = LambdaLR(optimizer, lr_lambda)

# move model to GPU

if train_on_gpu:

    model.cuda()

model
# Run the experiment

lr_find_loss = []

lr_find_lr = []



iter = 0



smoothing = 0.05

for i in range(lr_find_epochs):

  print("epoch {}".format(i))

  model.train()

  for inputs, labels in train_loader:

    # move tensors to GPU if CUDA is available

    if train_on_gpu:

      inputs, labels = inputs.cuda(), labels.cuda()



    optimizer.zero_grad()

    

    # Get outputs to calc loss

    outputs = model(inputs)

    loss = criterion(outputs, labels)



    # Backward pass

    loss.backward()

    optimizer.step()



    # Update LR

    scheduler.step()

    lr_step = optimizer.state_dict()["param_groups"][0]["lr"]

    lr_find_lr.append(lr_step)



    # smooth the loss

    if iter==0:

      lr_find_loss.append(loss)

    else:

      loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]

      lr_find_loss.append(loss)

     

    iter += 1
plt.ylabel("loss")

plt.xlabel("learning rate")

plt.xscale("log")

plt.plot(lr_find_lr, lr_find_loss)

plt.show()
# As concluded above

lr_max = 3e-3
def cyclical_lr(stepsize, min_lr=5e-4, max_lr=3e-3):



    # Scaler: we can adapt this if we do not want the triangular CLR

    scaler = lambda x: 1.



    # Lambda function to calculate the LR

    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)



    # Additional function to see where on the cycle we are

    def relative(it, stepsize):

        cycle = math.floor(1 + it / (2 * stepsize))

        x = abs(it / stepsize - 2 * cycle + 1)

        return max(0, (1 - x)) * scaler(cycle)



    return lr_lambda
from torch import optim



#Parameters

factor = 6

end_lr = lr_max

iter = 0

total_logs = []



#Loss function and optmixation function

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1.)

step_size = 4*len(train_loader)

clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])



if train_on_gpu:

    model.cuda()



model
# number of epochs to train the model

import numpy as np

n_epochs = 20



valid_loss_min = np.Inf # track change in validation loss



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for images, labels in train_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(images)

        # calculate the batch loss (comapre the values of the output model to the actual labels)

        loss = criterion(output, labels)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        

        scheduler.step() # > Where the magic happens

        lr_sched_test = scheduler.get_last_lr()

        # update training loss

        train_loss += loss.item()*images.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval()

    

    for images, labels in valid_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(images)

        # calculate the batch loss

        loss = criterion(output, labels)

        # update average validation loss 

        valid_loss += loss.item()*images.size(0)

    

    # calculate average losses

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        # print the decremnet in the validation

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tLearning rate: {}'.format(

            epoch, train_loss, valid_loss, lr_sched_test))

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min, 

        valid_loss))

        torch.save(model.state_dict(), 'model_AML_classifier.pt')

        valid_loss_min = valid_loss

    if epoch % 10 == 0:    

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tLearning rate: {}'.format(

            epoch, train_loss, valid_loss, lr_sched_test))   
model.load_state_dict(torch.load('model_AML_classifier.pt'))
#initialize the test loss

test_loss = 0.0



class_correct = list(0. for i in range(10))

class_total = list (0. for i in range(10))



#set the model to test and validation mode (no gradient descent needed)

model.eval()



for data, target in test_loader:

  #move the tensor to GPU ig available

  if train_on_gpu:

    data, target = data.cuda(), target.cuda()

  #forward pass: compute prediction output by passing the first batch of test data

  output = model(data)

  #calculate the batch size

  loss = criterion(output, target)

  #update the test loss

  test_loss += loss.item()*data.size(0)

  #convert output probabilities to output class

  _, pred = torch.max(output, 1)

  #compare the prediction to true label

  correct_tensor = pred.eq(target.data.view_as(pred))

  #conveert to numpy array and remove the extra dimention and get only the result

  correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())



  #calculate test accuracy for each object class

  for i in range(batch_size):

    try:

      label = target.data[i] #get the corresponding label from the object

      class_correct[label] += correct[i].item()

      class_total[label] += 1

    except IndexError:

      break

  

# average test loss

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(10):

  if class_total[i] > 0:

     print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

  else:

       print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))

   


#Move model inputs to cuda

if train_on_gpu:

    images = images.cuda()



#get sample outputs

output = model(images)

#convert probabilties to prediction class

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())



# plot the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(10):

    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])

    imshow(images.cpu()[idx])

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))
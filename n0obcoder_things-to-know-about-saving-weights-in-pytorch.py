# In this notebook, we will try to understand 2 of the most popular ways of saving weights in PyTorch-

# 1. Saving the weights of the model using state_dict()

# 2. Saving the whole model(including the architecture as well as the weights)



# Please follow the below mentioned blog link for detailed explaination of this notebook.

# https://medium.com/@animesh7pointer/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de
# Importing the necessary libraries



import torch

import torch.nn as nn
# Defining a CNN based model



class NeuralNet(nn.Module):

    def __init__(self):

        super(NeuralNet, self).__init__()

        

        self.sequential = nn.Sequential(nn.Conv2d(1, 32, 5), 

                                        nn.Conv2d(32, 64, 5), 

                                        nn.Dropout(0.3))

        self.layer1 = nn.Conv2d(64, 128, 5)

        self.layer2 = nn.Conv2d(128, 256, 5)

        self.fc = nn.Linear(256*34*34, 128)

    

    def forward(self, x):

        

        output = self.sequential(x)

        output = self.layer1(output)

        output = self.layer2(output)

        output = output.view(output.size()[0], -1)

        output = self.fc(output)

        

        return output
# Initializing and printing the model to see what's inside it



model = NeuralNet()



print(model)
# Printing all the parameters of the model



for name, param in model.named_parameters():

    print('name: ', name)

    print(type(param))

    print('param.shape: ', param.shape)

    print('param.requires_grad: ', param.requires_grad)

    print('=====')
# Making every parameter non-learnable or non-trainable except 'fc.weight' and 'fc.bias'



for name, param in model.named_parameters():

    if name in ['fc.weight', 'fc.bias']:

        param.requires_grad = True

    else:

        param.requires_grad = False
# Verifying if the desired changes have been made successfully



for name, param in model.named_parameters():

    print(name, ':', param.requires_grad)
#####

# Saving model's weights using model.state_dict()

#####
# print(model.state_dict())
# Printing the following shows us the 'model' is an instance of nn.Module



# help(model)
# Same can be verified by using python's isinstance function



print(isinstance(model, nn.Module))
# Is 'model.fc' also an instance of nn.Module ?



print(isinstance(model.fc, nn.Module))
# We can see what all nn.Module objects lie under model



for name, child in model.named_children():

    print('name: ', name)

    print('isinstance({}, nn.Module): '.format(name), isinstance(child, nn.Module))

    print('=====')
# Let’s now try the state_dict() function on the ‘fc’ layer of the model.



for key in model.fc.state_dict():

    print('key: ', key)

    param = model.fc.state_dict()[key]

    print('param.shape: ', param.shape)

    print('param.requires_grad: ', param.requires_grad)

    print('param.shape, param.requires_grad: ', param.shape, param.requires_grad)

    print('isinstance(param, nn.Module) ', isinstance(param, nn.Module))

    print('isinstance(param, nn.Parameter) ', isinstance(param, nn.Parameter))

    print('isinstance(param, torch.Tensor): ', isinstance(param, torch.Tensor))

    print('=====')
# We will now save the state_dict of the model



torch.save(model.state_dict(), 'weights_only.pth')



# This makes a ‘weights_only.pth’ file in the working directory and it holds, 

# in an ordered dictionary, the torch.Tensor objects of all the layers of the model.
# Next step would be to load the saved weights.



# But before we do that, we need to define the model architecture first. It makes

# sense to define the model first and then to load the weights in it because the 

# saved information is just the weights and not the model architecture.



model_new = NeuralNet()

model_new.load_state_dict(torch.load('weights_only.pth'))
# Checking the requires_grad attribute of all the loaded parameters



for name, param in model_new.named_parameters():

    print(name, ':', param.requires_grad)
# Wait ! What ?



# What happened to all the requires_grad flags that we had set for all the 

# different layers ? It seems like all the requires_grad flags have been

# reset to True.
# Saving Entire Model
# Yes we have this second way of saving things, in which we can save the entire model

# too. By entire model, I mean the architecture of the model as well as it’s weights.



# So we will resume from the point where we had frozen all but the last 

# layer (the ‘fc’ layer) of the model and save the entire model.
# Saving the entire model



torch.save(model, 'entire_model.pth')



# This makes a ‘entire_model.pth’ file in the working directory and it 

# contains the model architecture as well as the saved weights.
# We will try to load the saved model now. And this time, we do not need 

# to define the model architecture as the information about the model architecture 

# is already stored in the saved file.



model_new = torch.load('entire_model.pth')
# Once the model is loaded, let’s check the requires_grad attribute 

# of all the layers of model_new.



for name, param in model_new.named_parameters():

    print(name, ':', param.requires_grad)
# That is exactly what we wanted to see, isn’t it ? :D
#####



# CONCLUSIONS:



# 1. Saving a nn.Module object’s state_dict only saves the weights 

# of the various parameters of that object and not the model architecture. 

# Neither does it involve the requires_grad attribute of the weights. So before 

# loading the state_dict, one must define the model first.



# 2. Entire model (nn.Module object) can also be saved which would include the

# model architecture as well as its weights. Since we are saving the nn.Module 

# object, the requires_grad attribute is also saved this way. Also we don’t need

# to define the model architecture before loading the saved file since the saved

# file already has the model architecture saved in it



#####
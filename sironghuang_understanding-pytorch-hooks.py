import numpy as np

import torch

import torch.nn as nn

from IPython.display import Image
Image("../input/fig1.png",width=900,height=400)
class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(2,2)

        self.s1 = nn.Sigmoid()

        self.fc2 = nn.Linear(2,2)

        self.s2 = nn.Sigmoid()

        self.fc1.weight = torch.nn.Parameter(torch.Tensor([[0.15,0.2],[0.250,0.30]]))

        self.fc1.bias = torch.nn.Parameter(torch.Tensor([0.35]))

        self.fc2.weight = torch.nn.Parameter(torch.Tensor([[0.4,0.45],[0.5,0.55]]))

        self.fc2.bias = torch.nn.Parameter(torch.Tensor([0.6]))

        

    def forward(self, x):

        x= self.fc1(x)

        x = self.s1(x)

        x= self.fc2(x)

        x = self.s2(x)

        return x



net = Net()

print(net)
# parameters: weight and bias

print(list(net.parameters()))

# input data

weight2 = list(net.parameters())[2]

data = torch.Tensor([0.05,0.1])
# output of last layer

out = net(data)

target = torch.Tensor([0.01,0.99])  # a dummy target, for example

criterion = nn.MSELoss()

loss = criterion(out, target); loss
# A simple hook class that returns the input and output of a layer during forward/backward pass

class Hook():

    def __init__(self, module, backward=False):

        if backward==False:

            self.hook = module.register_forward_hook(self.hook_fn)

        else:

            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        self.input = input

        self.output = output

    def close(self):

        self.hook.remove()
# register hooks on each layer

hookF = [Hook(layer[1]) for layer in list(net._modules.items())]

hookB = [Hook(layer[1],backward=True) for layer in list(net._modules.items())]

# run a data batch

out=net(data)

# backprop once to get the backward hook results

out.backward(torch.tensor([1,1],dtype=torch.float),retain_graph=True)

#! loss.backward(retain_graph=True)  # doesn't work with backward hooks, 

#! since it's not a network layer but an aggregated result from the outputs of last layer vs target 



print('***'*3+'  Forward Hooks Inputs & Outputs  '+'***'*3)

for hook in hookF:

    print(hook.input)

    print(hook.output)

    print('---'*17)

print('\n')

print('***'*3+'  Backward Hooks Inputs & Outputs  '+'***'*3)

for hook in hookB:             

    print(hook.input)          

    print(hook.output)         

    print('---'*17)
# Confirm the calculations with the print result above

# the 4th layer - sigmoid

forward_output = np.array([0.7514, 0.7729]) 

grad_in = np.array([1,1])  # sigmoid layer

# grad of sigmoid(x) wrt x is: sigmoid(x)(1-sigmoid(x))

grad_out = grad_in*(forward_output*(1-forward_output)); grad_out 
# the 3th layer - linear

print([0.1868, 0.1755])  # grad_input * (grad of Wx+b = (w1*x1+w2*x2)+b wrt W) 

print(0.1868 + 0.1755)   # grad of Wx+b wrt b o



grad_in = torch.Tensor(grad_out)

grad_in.view(1,-1) @ weight2;grad_out  # grad of layer output wrt input: wx+b => w
# the 2nd layer - sigmoid

forward_output=np.array([0.5933, 0.5969])

grad_in=np.array([0.1625, 0.1806])

grad_in*(forward_output*(1-forward_output)) # grad * (grad of sigmoid(x) wrt x)
# gradient of loss wrt prarameters

net.zero_grad()

loss.backward(retain_graph=True)

[print(p.grad) for p in net.parameters()]
class Guided_backprop():

    """

        Visualize CNN activation maps with guided backprop.

        

        Returns: An image that represent what the network learnt for recognizing 

        the given image. 

        

        Methods: First layer input that minimize the error between the last layers output,

        for the given class, and the true label(=1). 

        

        ! Call visualize(image) to get the image representation

    """

    def __init__(self,model):

        self.model = model

        self.image_reconstruction = None

        self.activation_maps = []

        # eval mode

        self.model.eval()

        self.register_hooks()

    

    def register_hooks(self):

        

        def first_layer_hook_fn(module, grad_out, grad_in):

            """ Return reconstructed activation image"""

            self.image_reconstruction = grad_out[0] 

            

        def forward_hook_fn(module, input, output):

            """ Stores the forward pass outputs (activation maps)"""

            self.activation_maps.append(output)

            

        def backward_hook_fn(module, grad_out, grad_in):

            """ Output the grad of model output wrt. layer (only positive) """

            

            # Gradient of forward_output wrt. forward_input = error of activation map:

                # for relu layer: grad of zero = 0, grad of identity = 1

            grad = self.activation_maps[-1] # corresponding forward pass output 

            grad[grad>0] = 1 # grad of relu when > 0

            

            # set negative output gradient to 0 #!???

            positive_grad_out = torch.clamp(input=grad_out[0],min=0.0)

            

            # backward grad_out = grad_out * (grad of forward output wrt. forward input)

            new_grad_out = positive_grad_out * grad

            

            del self.forward_outputs[-1] 

            

            # For hook functions, the returned value will be the new grad_out

            return (new_grad_out,)

            

        # !!!!!!!!!!!!!!!! change the modules !!!!!!!!!!!!!!!!!!

        # only conv layers, no flattened fc linear layers

        modules = list(self.model.features._modules.items())

        

        # register hooks to relu layers

        for name, module in modules:

            if isinstance(module, nn.ReLU):

                module.register_forward_hook(forward_hook_fn)

                module.register_backward_hook(backward_hook_fn)

        

        # register hook to the first layer 

        first_layer = modules[0][1] 

        first_layer.register_backward_hook(first_layer_hook_fn)

        

    def visualize(self, input_image, target_class):

        # last layer output

        model_output = self.model(input_image)

        self.model.zero_grad()

        

        # only calculate gradients wrt. target class 

        # set the other classes to 0: eg. [0,0,1]

        grad_target_map = torch.zeros(model_output.shape,

                                     dtype=torch.float)

        grad_target_map[0][target_class] = 1

        

        model_output.backward(grad_target_map)

        

        # Convert Pytorch variable to numpy array

        # [0] to get rid of the first channel (1,3,224,224)

        result = self.image_reconstruction.data.numpy()[0] 

        return result